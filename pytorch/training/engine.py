"""Fused Triton training engine mirroring `fast_inference/engine.py`, separated from the HF reference model.

Wraps a live `LoomFormerForCausalLM` without copying weights so DDP, AdamW, and `hf_model.save_pretrained()`
operate directly on the standard HuggingFace parameters. Each decoder layer is encapsulated in a self-contained
module (`FastLoomLayer` or `FastMHALayer`) with in-layer `MRead` and `MWrite` (Triton-only fast path).
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from loomformer.configuration_loomformer import LoomFormerConfig
from loomformer.modeling_loomformer import (
    LoomFormerDecoderLayer,
    LoomFormerForCausalLM,
    RotaryEmbedding,
    gelu_approx,
    rms_norm_unit,
)
from .triton_ops import (
    can_use_triton_loom,
    causal_attention_train,
    loom_read_triton,
    loom_write_triton,
    strict_causal_attention_add_train,
)


_ARMS = ("q", "k", "v", "o")


def _pad_rows(weights: list, multiple: int = 8) -> int:
  """Append zero rows so fused GEMM width is a multiple of `multiple`; returns rows added."""
  pad = -sum(w.shape[0] for w in weights) % multiple
  if pad:
    weights.append(weights[0].new_zeros((pad, weights[0].shape[1])))
  return pad


def _swiglu_act(gate: torch.Tensor, up: torch.Tensor, beta1: float = 0.0, beta2: float = 0.0) -> torch.Tensor:
  if beta1 > 0.0 and beta2 > 0.0:
    gate_f32, up_f32 = gate.float(), up.float()
    return (
        (beta1 * torch.tanh(gate_f32 / beta1) * torch.sigmoid(gate_f32))
        * (beta2 * torch.tanh(up_f32 / beta2))
    ).to(gate.dtype)
  return F.silu(gate) * up


class _FastDecoderLayerBase(nn.Module):
  """Shared base for fused decoder layers referencing live HF `LoomFormerDecoderLayer` weights."""

  def __init__(self, layer: LoomFormerDecoderLayer, config: LoomFormerConfig):
    super().__init__()
    self.layer = layer
    self.num_heads = config.num_attention_heads
    self.beta1 = float(config.situ_beta1)
    self.beta2 = float(config.situ_beta2)

  def ffn_x(self, hidden: torch.Tensor) -> torch.Tensor:
    """Fused gate+up SwiGLU / SiTU-GLU token FFN (zero-pads odd intermediate widths to a multiple of 8)."""
    ffn = self.layer.ffn_x
    inter = ffn.gate.weight.shape[0]
    pad = -inter % 8
    if pad:
      zeros = ffn.gate.weight.new_zeros((pad, ffn.gate.weight.shape[1]))
      fused_weight = torch.cat((ffn.gate.weight, zeros, ffn.up.weight, zeros), dim=0)
      gate, up = F.linear(hidden, fused_weight).chunk(2, dim=-1)
      return F.linear(_swiglu_act(gate, up, self.beta1, self.beta2), F.pad(ffn.down.weight, (0, pad)))
    fused_weight = torch.cat((ffn.gate.weight, ffn.up.weight), dim=0)
    gate, up = F.linear(hidden, fused_weight).chunk(2, dim=-1)
    return ffn.down(_swiglu_act(gate, up, self.beta1, self.beta2))


class FastMHALayer(_FastDecoderLayerBase):
  """Fused standard Transformer decoder layer (`loom_enabled=False`) referencing live HF parameters."""

  def __init__(self, layer: LoomFormerDecoderLayer, config: LoomFormerConfig):
    super().__init__(layer, config)
    self.head_dim = config.head_dim
    self.nope_dim = config.mha_nope_dim
    self.scale = self.head_dim ** -0.5

  def _self_attn(self, x_norm: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    b, t, hidden_size = x_norm.shape
    attn = self.layer.self_attn
    qkv_weight = torch.cat((attn.query.weight, attn.key.weight, attn.value.weight), dim=0)
    q, k, v = F.linear(x_norm, qkv_weight).view(b, t, 3, self.num_heads, self.head_dim).unbind(2)
    if self.nope_dim > 0:
      q = torch.cat((q[..., :self.nope_dim], RotaryEmbedding.apply(q[..., self.nope_dim:], cos, sin)), dim=-1)
      k = torch.cat((k[..., :self.nope_dim], RotaryEmbedding.apply(k[..., self.nope_dim:], cos, sin)), dim=-1)
    else:
      q = RotaryEmbedding.apply(q, cos, sin)
      k = RotaryEmbedding.apply(k, cos, sin)
    attn_out = causal_attention_train(q * self.scale, k, v)
    return attn.out(attn_out.reshape(b, t, hidden_size))

  def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x = x + self._self_attn(self.layer.ln_1(x), cos, sin)
    x = x + self.ffn_x(self.layer.ln_2(x))
    return x


class FastLoomLayer(_FastDecoderLayerBase):
  """Self-contained Triton Loom decoder layer (in-layer `MRead` -> Attention -> FFN -> `MWrite`)."""

  def __init__(self, layer: LoomFormerDecoderLayer, config: LoomFormerConfig, cache_read_fb: bool = False):
    super().__init__(layer, config)
    self.head_dim = config.head_dim
    self.d_k = config.d_k
    self.rope_dim = config.rope_dim
    self.d_r = config.d_r
    self._cache_read_fb = cache_read_fb

  def _loom_mha(
      self, x_norm: torch.Tensor, M: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, b: int, t: int
  ):
    """Fused input projections + Triton MShift/MNorm/MRead/RoPE + Strictly Causal Attention."""
    assert can_use_triton_loom(x_norm, self.head_dim), (
        f"FastLoomLayer requires CUDA BF16 and head_dim >= 16 "
        f"(got device={x_norm.device}, dtype={x_norm.dtype}, head_dim={self.head_dim})"
    )
    num_heads, head_dim, d_k, rope_dim = self.num_heads, self.head_dim, self.d_k, self.rope_dim
    attn = self.layer.loom_mha

    proj_weights = [attn.W_sg.weight, attn.W_q.weight, attn.W_k.weight]
    proj_weights += [getattr(attn, f"W_rg_{arm}").weight for arm in _ARMS]
    proj_weights += [getattr(attn, f"W_rg_bar_{arm}").weight for arm in _ARMS]
    proj_weights += [attn.W_wg.weight, attn.W_r1.weight, attn.W_w1.weight] if attn.write_M else [attn.W_r1.weight]
    proj_pad = _pad_rows(proj_weights)
    proj = F.linear(x_norm, torch.cat(proj_weights, dim=0))

    if attn.write_M:
      shift, std, gates, write_gate, down_proj, _ = proj.split(
          (d_k, 2 * num_heads * rope_dim, 8 * num_heads, num_heads, 2 * self.d_r, proj_pad), dim=-1
      )
      read_down_act, write_down_act = gelu_approx(down_proj).chunk(2, dim=-1)
    else:
      shift, std, gates, down_proj, _ = proj.split(
          (d_k, 2 * num_heads * rope_dim, 8 * num_heads, self.d_r, proj_pad), dim=-1
      )
      write_gate = write_down_act = None
      read_down_act = gelu_approx(down_proj)

    up_weight = torch.cat([getattr(attn, f"W_r2_{arm}").weight for arm in _ARMS], dim=0)
    up_bias = torch.cat([getattr(attn, f"W_r2_{arm}").bias for arm in _ARMS], dim=0)
    up = F.linear(read_down_act, up_weight, up_bias)

    static_key = torch.cat([getattr(attn, f"q_bar_{arm}") for arm in _ARMS], dim=0)
    read_biases = torch.cat(
        [attn.W_sg.bias]
        + [getattr(attn, f"W_rg_{arm}").bias for arm in _ARMS]
        + [getattr(attn, f"W_rg_bar_{arm}").bias for arm in _ARMS],
        dim=0,
    )

    q_sub, k_sub, v_sub, o_m = loom_read_triton(
        M, shift, std, up, gates, static_key, read_biases, attn.c_qk, attn.c_vo, cos, sin,
        b, num_heads, head_dim, d_k, rope_dim,
        attn.eps, attn.key_eps, attn.key_scale, 1.1 / attn.static_init, 1.0 / attn.logit_scale,
        cache_fb=self._cache_read_fb,
    )
    attn_heads = strict_causal_attention_add_train(q_sub, k_sub, v_sub, o_m, b, t, num_heads, head_dim)
    dx = attn.W_o(attn_heads)
    attn_write = (attn_heads, write_gate, write_down_act) if attn.write_M else None
    return dx, attn_write

  def _ffn_and_write(
      self, hidden: torch.Tensor, M: torch.Tensor, attn_write
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused token FFN (`ffn_x`) + memory FFN (`ffn_m` + write gate/address) + in-layer Triton `MWrite`."""
    if attn_write is None:
      return self.ffn_x(hidden), M
    attn_heads, write_gate, write_down_act = attn_write
    attn = self.layer.loom_mha
    ffn_x, ffn_m = self.layer.ffn_x, self.layer.ffn_m

    x_inter = ffn_x.gate.weight.shape[0]
    x_pad = -x_inter % 8
    x_width = x_inter + x_pad
    mem_inter = ffn_m.gate.weight.shape[0]
    mem_pad = -mem_inter % 8
    mem_width = mem_inter + mem_pad

    if self.beta1 == 0.0 and self.beta2 == 0.0:
      # Pair-fuse `[x_gate, mem_gate]` and `[x_up, mem_up]` instead of concatenating all 6 projections:
      # keeps each backward GEMM input contiguous while avoiding a 6-way Inductor `cat` kernel.
      g_weights, u_weights = [ffn_x.gate.weight], [ffn_x.up.weight]
      if x_pad:
        z_x = ffn_x.gate.weight.new_zeros((x_pad, ffn_x.gate.weight.shape[1]))
        g_weights.append(z_x)
        u_weights.append(z_x)
      g_weights.append(ffn_m.gate.weight)
      u_weights.append(ffn_m.up.weight)
      if mem_pad:
        z_m = ffn_m.gate.weight.new_zeros((mem_pad, ffn_m.gate.weight.shape[1]))
        g_weights.append(z_m)
        u_weights.append(z_m)
      x_gate, mem_gate = F.linear(hidden, torch.cat(g_weights, dim=0)).split((x_width, mem_width), dim=-1)
      x_up, mem_up = F.linear(hidden, torch.cat(u_weights, dim=0)).split((x_width, mem_width), dim=-1)
      x_act = F.silu(x_gate) * x_up
      dx_ffn = F.linear(x_act, F.pad(ffn_x.down.weight, (0, x_pad))) if x_pad else ffn_x.down(x_act)
      mem_act = F.silu(mem_gate) * mem_up
      mem_content = F.linear(mem_act, F.pad(ffn_m.down.weight, (0, mem_pad))) if mem_pad else ffn_m.down(mem_act)
    else:
      dx_ffn = self.ffn_x(hidden)
      if mem_pad:
        z_m = ffn_m.gate.weight.new_zeros((mem_pad, ffn_m.gate.weight.shape[1]))
        mem_w = torch.cat((ffn_m.gate.weight, z_m, ffn_m.up.weight, z_m), dim=0)
        mem_gate, mem_up = F.linear(hidden, mem_w).chunk(2, dim=-1)
        mem_content = F.linear(F.silu(mem_gate) * mem_up, F.pad(ffn_m.down.weight, (0, mem_pad)))
      else:
        mem_gate, mem_up = F.linear(hidden, torch.cat((ffn_m.gate.weight, ffn_m.up.weight), dim=0)).chunk(2, dim=-1)
        mem_content = ffn_m.down(F.silu(mem_gate) * mem_up)

    if self.layer.ffn_divisor != 1.0:
      mem_content = (mem_content.float() / self.layer.ffn_divisor).to(mem_content.dtype)

    aux_weights = [self.layer.ffn_W_wg.weight, self.layer.ffn_W_w1.weight]
    aux_pad = _pad_rows(aux_weights)
    ffn_write_gate, ffn_addr_down, _ = F.linear(hidden, torch.cat(aux_weights, dim=0)).split(
        (self.num_heads, self.d_r, aux_pad), dim=-1
    )

    addr_attn = attn.W_w2(write_down_act)
    addr_ffn = self.layer.ffn_W_w2(gelu_approx(ffn_addr_down))
    write_biases = torch.cat((attn.W_wg.bias, self.layer.ffn_W_wg.bias), dim=0)
    M = loom_write_triton(
        M, attn_heads, write_gate, addr_attn, mem_content, ffn_write_gate, addr_ffn, write_biases,
        self.num_heads, self.head_dim, self.d_k, self.layer.write_eps,
    )
    return dx_ffn, M

  def forward(
      self, x_flat: torch.Tensor, M: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, b: int, t: int
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    dx, attn_write = self._loom_mha(self.layer.ln_1(x_flat), M, cos, sin, b, t)
    x_flat = x_flat + dx
    dx_ffn, M = self._ffn_and_write(self.layer.ln_2(x_flat), M, attn_write)
    x_flat = x_flat + dx_ffn
    return x_flat, M


class FastTrainModel(nn.Module):
  """Fused Triton training wrapper around a live `LoomFormerForCausalLM` (zero weight copies)."""

  def __init__(self, hf_model: LoomFormerForCausalLM, grad_checkpoint: bool = False):
    super().__init__()
    self.hf_model = hf_model
    self.config = hf_model.config
    self.loom_enabled = self.config.loom_enabled
    self.grad_checkpoint = grad_checkpoint
    if self.loom_enabled:
      self.layers = nn.ModuleList(
          FastLoomLayer(layer, self.config, cache_read_fb=grad_checkpoint) for layer in hf_model.model.layers
      )
    else:
      self.layers = nn.ModuleList(FastMHALayer(layer, self.config) for layer in hf_model.model.layers)

    # Precompute RoPE cos/sin table so `torch.compile` slices a static buffer instead of recomputing trig per layer
    device = next(hf_model.parameters()).device
    rotary = hf_model.model.layers[0].loom_mha.rotary if self.loom_enabled else hf_model.model.layers[0].self_attn.rotary
    cos_3d, sin_3d = rotary.table(torch.arange(self.config.max_position_embeddings, device=device), torch.float32)
    self.register_buffer("_rope_cos", cos_3d.squeeze(1).contiguous(), persistent=False)
    self.register_buffer("_rope_sin", sin_3d.squeeze(1).contiguous(), persistent=False)

  def _rope_tables(self, t: int, dtype: torch.dtype, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    if t <= self._rope_cos.shape[0]:
      cos, sin = self._rope_cos[:t].to(dtype), self._rope_sin[:t].to(dtype)
    else:
      attn0 = self.hf_model.model.layers[0].loom_mha if self.loom_enabled else self.hf_model.model.layers[0].self_attn
      cos_3d, sin_3d = attn0.rotary.table(torch.arange(t, device=device), dtype)
      cos, sin = cos_3d.squeeze(1), sin_3d.squeeze(1)
    return cos.contiguous(), sin.contiguous()

  def _embed_write(self, x_flat: torch.Tensor) -> torch.Tensor:
    cfg = self.config
    embed_write = self.hf_model.model.embed_write
    num_tokens, dtype = x_flat.shape[0], x_flat.dtype
    num_heads, head_dim, d_k = cfg.num_attention_heads, cfg.head_dim, cfg.d_k
    fused_weight = torch.cat((embed_write.W_e.weight, embed_write.W_w1.weight, embed_write.W_wg.weight), dim=0)
    content, addr_down, gate_logits = F.linear(x_flat, fused_weight).split((num_heads * head_dim, cfg.d_r, num_heads), dim=-1)
    if embed_write.divisor:
      content = (content.float() / embed_write.divisor).to(dtype).view(num_tokens, num_heads, head_dim)
    else:
      content = rms_norm_unit(content.view(num_tokens, num_heads, head_dim), embed_write.eps)
    gate = torch.sigmoid(gate_logits + embed_write.W_wg.bias.to(dtype))
    addr = rms_norm_unit(embed_write.W_w2(gelu_approx(addr_down)).view(num_tokens, num_heads, d_k), embed_write.eps)
    return torch.matmul((gate[..., None] * content).transpose(-1, -2), addr).reshape(num_tokens, head_dim * d_k)

  def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
    backbone = self.hf_model.model
    x = backbone.embed_tokens(input_ids)
    if x.is_cuda and torch.is_autocast_enabled("cuda"):
      x = x.to(torch.get_autocast_dtype("cuda"))
    b, t, hidden_size = x.shape

    use_ckpt = self.grad_checkpoint and self.training and torch.is_grad_enabled()
    cos, sin = self._rope_tables(t, x.dtype, x.device)

    if not self.loom_enabled:
      cos, sin = cos[None, :, None, :], sin[None, :, None, :]
      for layer in self.layers:
        x = checkpoint(layer, x, cos, sin, use_reentrant=False) if use_ckpt else layer(x, cos, sin)
      return self.hf_model.compute_logits(backbone.norm(x))

    x_flat = x.reshape(b * t, hidden_size)
    M = self._embed_write(x_flat)
    for layer in self.layers:
      if use_ckpt:
        x_flat, M = checkpoint(layer, x_flat, M, cos, sin, b, t, use_reentrant=False)
      else:
        x_flat, M = layer(x_flat, M, cos, sin, b, t)
    return self.hf_model.compute_logits(backbone.norm(x_flat.view(b, t, hidden_size)))


def build_train_model(hf_model: LoomFormerForCausalLM, grad_checkpoint: bool = False) -> FastTrainModel:
  """Build a fused Triton training wrapper sharing live parameters with `hf_model`."""
  return FastTrainModel(hf_model, grad_checkpoint=grad_checkpoint)
