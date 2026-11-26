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


def _pad_rows(weights: list, multiple: int = 8) -> int:
  """Append zero rows so fused GEMM width is a multiple of `multiple`; returns rows added."""
  pad = -sum(w.shape[0] for w in weights) % multiple
  if pad:
    weights.append(weights[0].new_zeros((pad, weights[0].shape[1])))
  return pad


def _swiglu_act(g: torch.Tensor, u: torch.Tensor, beta1: float = 0.0, beta2: float = 0.0) -> torch.Tensor:
  if beta1 > 0.0 and beta2 > 0.0:
    gf, uf = g.float(), u.float()
    return ((beta1 * torch.tanh(gf / beta1) * torch.sigmoid(gf)) * (beta2 * torch.tanh(uf / beta2))).to(g.dtype)
  return F.silu(g) * u


class FastMHALayer(nn.Module):
  """Fused standard Transformer decoder layer (`loom_enabled=False`) referencing live HF parameters."""

  def __init__(self, layer: LoomFormerDecoderLayer, config: LoomFormerConfig):
    super().__init__()
    self.layer = layer
    self.n = config.num_attention_heads
    self.hd = config.head_dim
    self.nope = config.mha_nope_dim
    self.scale = self.hd ** -0.5
    self.beta1 = float(config.situ_beta1)
    self.beta2 = float(config.situ_beta2)

  def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    b, t, d_model = x.shape
    dt = x.dtype
    a = self.layer.self_attn

    xn = self.layer.ln_1(x)
    qkv_w = torch.cat((a.query.weight, a.key.weight, a.value.weight), dim=0).to(dt)
    q, k, v = F.linear(xn, qkv_w).view(b, t, 3, self.n, self.hd).unbind(2)
    if self.nope > 0:
      q = torch.cat((q[..., :self.nope], RotaryEmbedding.apply(q[..., self.nope:], cos, sin)), dim=-1)
      k = torch.cat((k[..., :self.nope], RotaryEmbedding.apply(k[..., self.nope:], cos, sin)), dim=-1)
    else:
      q = RotaryEmbedding.apply(q, cos, sin)
      k = RotaryEmbedding.apply(k, cos, sin)
    y = causal_attention_train(q * self.scale, k, v)
    x = x + F.linear(y.reshape(b, t, d_model), a.out.weight.to(dt))

    h = self.layer.ln_2(x)
    gu_w = torch.cat((self.layer.ffn_x.gate.weight, self.layer.ffn_x.up.weight), dim=0).to(dt)
    xg, xu = F.linear(h, gu_w).chunk(2, dim=-1)
    mlp = F.linear(_swiglu_act(xg, xu, self.beta1, self.beta2), self.layer.ffn_x.down.weight.to(dt))
    return x + mlp


class FastLoomLayer(nn.Module):
  """Self-contained Triton Loom decoder layer (in-layer `MRead` -> Attention -> FFN -> `MWrite`)."""

  def __init__(self, layer: LoomFormerDecoderLayer, config: LoomFormerConfig):
    super().__init__()
    self.layer = layer
    self.n = config.num_attention_heads
    self.k = config.head_dim
    self.v = config.d_k
    self.s = config.rope_dim
    self.d_r = config.d_r
    self.beta1 = float(config.situ_beta1)
    self.beta2 = float(config.situ_beta2)

  def forward(
      self, x_flat: torch.Tensor, M: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, batch: int, seq_len: int
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    assert can_use_triton_loom(x_flat, self.k), (
        f"FastLoomLayer requires CUDA BF16 and head_dim >= 64 (got device={x_flat.device}, dtype={x_flat.dtype}, d={self.k})"
    )
    dt = x_flat.dtype
    n, k, v, s = self.n, self.k, self.v, self.s
    a = self.layer.loom_mha

    # 1. Fused input projection of `ln_1(x)` (all shift, RoPE, read/write gates & bottlenecks in one GEMM)
    xn = self.layer.ln_1(x_flat)
    xs = [a.W_sg.weight, a.W_q.weight, a.W_k.weight]
    xs += [getattr(a, f"W_rg_{arm}").weight for arm in ("q", "k", "v", "o")]
    xs += [getattr(a, f"W_rg_bar_{arm}").weight for arm in ("q", "k", "v", "o")]
    if a.write_M:
      xs.append(a.W_wg.weight)
    xs.append(a.W_r1.weight)
    if a.write_M:
      xs.append(a.W_w1.weight)
    z_pad = _pad_rows(xs)
    z = F.linear(xn, torch.cat(xs, dim=0).to(dt))

    if a.write_M:
      shift, std, gates, write_gate, z_down, _ = z.split((v, 2 * n * s, 8 * n, n, 2 * self.d_r, z_pad), dim=-1)
      read_down_act, write_down_act = gelu_approx(z_down).chunk(2, dim=-1)
    else:
      shift, std, gates, z_down, _ = z.split((v, 2 * n * s, 8 * n, self.d_r, z_pad), dim=-1)
      write_gate = None
      read_down_act = gelu_approx(z_down)
      write_down_act = None

    # 2. Fused 4-arm read up-projection + Triton MShift/MNorm/MRead/RoPE + Strictly Causal Attention
    up_w = torch.cat([getattr(a, f"W_r2_{arm}").weight for arm in ("q", "k", "v", "o")], dim=0).to(dt)
    up_b = torch.cat([getattr(a, f"W_r2_{arm}").bias for arm in ("q", "k", "v", "o")], dim=0).to(dt)
    up = F.linear(read_down_act, up_w, up_b)

    static_key = torch.cat([getattr(a, f"q_bar_{arm}") for arm in ("q", "k", "v", "o")], dim=0).to(dt)
    xb = torch.cat(
        [a.W_sg.bias]
        + [getattr(a, f"W_rg_{arm}").bias for arm in ("q", "k", "v", "o")]
        + [getattr(a, f"W_rg_bar_{arm}").bias for arm in ("q", "k", "v", "o")],
        dim=0,
    ).to(dt)

    q_sub, k_sub, v_sub, o_m = loom_read_triton(
        M, shift, std, up, gates, static_key, xb, a.c_qk.to(dt), a.c_vo.to(dt), cos, sin,
        batch, n, k, v, s, a.eps, a.key_eps, a.key_scale, 1.1 / a.static_init, 1.0 / a.logit_scale,
    )
    o_head = strict_causal_attention_add_train(q_sub, k_sub, v_sub, o_m, batch, seq_len, n, k)
    x_flat = x_flat + F.linear(o_head, a.W_o.weight.to(dt))

    # 3. Fused FFN (`ffn_x` + `ffn_m` + write gates/addresses) and in-layer Triton MWrite
    h = self.layer.ln_2(x_flat)
    inter = self.layer.ffn_x.gate.weight.shape[0]
    if self.layer.write_M:
      mem_inter = self.layer.ffn_m.gate.weight.shape[0]
      hs = [
          self.layer.ffn_x.gate.weight, self.layer.ffn_m.gate.weight,
          self.layer.ffn_x.up.weight, self.layer.ffn_m.up.weight,
          self.layer.ffn_W_wg.weight, self.layer.ffn_W_w1.weight,
      ]
      h_pad = _pad_rows(hs)
      gu, mgate, adown, _ = F.linear(h, torch.cat(hs, dim=0).to(dt)).split(
          (2 * (inter + mem_inter), n, self.d_r, h_pad), dim=-1
      )
      g_all, u_all = gu.chunk(2, dim=-1)
      if self.beta1 == 0.0 and self.beta2 == 0.0:
        act_x, act_m = (F.silu(g_all) * u_all).split((inter, mem_inter), dim=-1)
      else:
        xg, mg = g_all.split((inter, mem_inter), dim=-1)
        xu, mu = u_all.split((inter, mem_inter), dim=-1)
        act_x, act_m = _swiglu_act(xg, xu, self.beta1, self.beta2), F.silu(mg) * mu
      mlp = F.linear(act_x, self.layer.ffn_x.down.weight.to(dt))
      mem = F.linear(act_m, self.layer.ffn_m.down.weight.to(dt))
      if self.layer.ffn_divisor != 1.0:
        mem = (mem.float() / self.layer.ffn_divisor).to(dt)
      addr_a = F.linear(write_down_act, a.W_w2.weight.to(dt), a.W_w2.bias.to(dt))
      addr_f = F.linear(gelu_approx(adown), self.layer.ffn_W_w2.weight.to(dt), self.layer.ffn_W_w2.bias.to(dt))
      wb = torch.cat((a.W_wg.bias, self.layer.ffn_W_wg.bias), dim=0).to(dt)
      M = loom_write_triton(M, o_head, write_gate, addr_a, mem, mgate, addr_f, wb, n, k, v, self.layer.write_eps)
    else:
      hw = torch.cat((self.layer.ffn_x.gate.weight, self.layer.ffn_x.up.weight), dim=0).to(dt)
      xg, xu = F.linear(h, hw).chunk(2, dim=-1)
      mlp = F.linear(_swiglu_act(xg, xu, self.beta1, self.beta2), self.layer.ffn_x.down.weight.to(dt))

    return x_flat + mlp, M


class FastTrainModel(nn.Module):
  """Fused Triton training wrapper around a live `LoomFormerForCausalLM` (zero weight copies)."""

  def __init__(self, hf_model: LoomFormerForCausalLM, grad_checkpoint: bool = False):
    super().__init__()
    self.hf_model = hf_model
    self.config = hf_model.config
    self.loom_enabled = self.config.loom_enabled
    self.grad_checkpoint = grad_checkpoint
    layer_cls = FastLoomLayer if self.loom_enabled else FastMHALayer
    self.layers = nn.ModuleList(layer_cls(layer, self.config) for layer in hf_model.model.layers)

    # Precompute RoPE cos/sin table so `torch.compile` slices a static buffer instead of recomputing trig per layer
    dev = next(hf_model.parameters()).device
    rotary = hf_model.model.layers[0].loom_mha.rotary if self.loom_enabled else hf_model.model.layers[0].self_attn.rotary
    cos_3d, sin_3d = rotary.table(torch.arange(self.config.max_position_embeddings, device=dev), torch.float32)
    self.register_buffer("_rope_cos", cos_3d.squeeze(1).contiguous(), persistent=False)
    self.register_buffer("_rope_sin", sin_3d.squeeze(1).contiguous(), persistent=False)

  def _embed_write(self, x_flat: torch.Tensor) -> torch.Tensor:
    c = self.config
    ew = self.hf_model.model.embed_write
    rows, dt = x_flat.shape[0], x_flat.dtype
    n, k, v = c.num_attention_heads, c.head_dim, c.d_k
    ew_w = torch.cat((ew.W_e.weight, ew.W_w1.weight, ew.W_wg.weight), dim=0).to(dt)
    ew_b = torch.cat((
        ew.W_e.weight.new_zeros(n * k, dtype=dt),
        ew.W_w1.weight.new_zeros(c.d_r, dtype=dt),
        ew.W_wg.bias.to(dt),
    ), dim=0)
    e_data, e_hid_in, e_gate_in = F.linear(x_flat, ew_w, ew_b).split((n * k, c.d_r, n), dim=-1)
    if ew.divisor:
      e_data = (e_data.float() / ew.divisor).to(dt)
    else:
      e_data = rms_norm_unit(e_data.view(rows, n, k), ew.eps).view(rows, n * k)
    e_gate = torch.sigmoid(e_gate_in)
    e_addr = rms_norm_unit(F.linear(gelu_approx(e_hid_in), ew.W_w2.weight.to(dt), ew.W_w2.bias.to(dt)).view(rows, n, v), ew.eps)
    return torch.matmul((e_gate[..., None] * e_data.view(rows, n, k)).transpose(-1, -2), e_addr).reshape(rows, k * v)

  def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
    m = self.hf_model.model
    x = m.embed_tokens(input_ids)
    if x.is_cuda and torch.is_autocast_enabled("cuda"):
      x = x.to(torch.get_autocast_dtype("cuda"))
    b, t, d_model = x.shape

    use_ckpt = self.grad_checkpoint and self.training and torch.is_grad_enabled()

    if not self.loom_enabled:
      if t <= self._rope_cos.shape[0]:
        cos = self._rope_cos[:t].to(x.dtype)[None, :, None, :]
        sin = self._rope_sin[:t].to(x.dtype)[None, :, None, :]
      else:
        pos = torch.arange(t, device=x.device)[None]
        cos, sin = m.layers[0].self_attn.rotary.table(pos, x.dtype)
      for layer in self.layers:
        if use_ckpt:
          x = checkpoint(layer, x, cos, sin, use_reentrant=False)
        else:
          x = layer(x, cos, sin)
      return self.hf_model.compute_logits(m.norm(x))

    if t <= self._rope_cos.shape[0]:
      cos = self._rope_cos[:t].to(x.dtype).contiguous()
      sin = self._rope_sin[:t].to(x.dtype).contiguous()
    else:
      cos_3d, sin_3d = m.layers[0].loom_mha.rotary.table(torch.arange(t, device=x.device), x.dtype)
      cos, sin = cos_3d.squeeze(1).contiguous(), sin_3d.squeeze(1).contiguous()
    x_flat = x.reshape(b * t, d_model)
    M = self._embed_write(x_flat)
    for layer in self.layers:
      if use_ckpt:
        x_flat, M = checkpoint(layer, x_flat, M, cos, sin, b, t, use_reentrant=False)
      else:
        x_flat, M = layer(x_flat, M, cos, sin, b, t)
    return self.hf_model.compute_logits(m.norm(x_flat.view(b, t, d_model)))


def build_train_model(hf_model: LoomFormerForCausalLM, grad_checkpoint: bool = False) -> FastTrainModel:
  """Build a fused Triton training wrapper sharing live parameters with `hf_model`."""
  return FastTrainModel(hf_model, grad_checkpoint=grad_checkpoint)
