"""HuggingFace PyTorch implementation of LoomFormer (Llama RMSNorm/SwiGLU backbone, Section 3.2 & Section 5.1).

Self-contained module supporting `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`.
Tensor axis order (`M` in `[B, T, d_k, d]`, `q_bar_*` in `[n, d_k]`) and parameter names match Paper Section 3.2,
`loomformer_pseudo.py`, and `jax/MaxText/layers/loom.py`.
"""
import importlib
import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import GenerationMixin, PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

try:
  from .configuration_loomformer import LoomFormerConfig
except ImportError:
  LoomFormerConfig = importlib.import_module("configuration_loomformer").LoomFormerConfig


def rms_norm_unit(x: torch.Tensor, eps: float) -> torch.Tensor:
  """Parameter-free RMSNorm over the last axis with FP32 statistics."""
  x_f32 = x.float()
  return (x_f32 * torch.rsqrt(x_f32.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)


def gelu_approx(x: torch.Tensor) -> torch.Tensor:
  return F.gelu(x, approximate="tanh")


def split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
  return x.unflatten(-1, (num_heads, -1))


class RMSNorm(nn.Module):
  """RMSNorm with direct learned multiplier `weight`."""

  def __init__(self, dim: int, eps: float):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(dim))
    self.eps = eps

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if x.is_cuda and hasattr(F, "rms_norm"):
      return F.rms_norm(x, (x.shape[-1],), self.weight.to(x.dtype), self.eps).to(x.dtype)
    return rms_norm_unit(x, self.eps) * self.weight.to(x.dtype)


class RotaryEmbedding(nn.Module):
  """Half-split Rotary Position Embedding over `dim` coordinates."""

  def __init__(self, dim: int, min_timescale: float = 1.0, max_timescale: float = 10000.0):
    super().__init__()
    self.dim = dim
    self.min_timescale = min_timescale
    self.max_timescale = max_timescale

  def table(self, positions: torch.Tensor, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    fraction = 2.0 * torch.arange(self.dim // 2, dtype=torch.float32, device=positions.device) / self.dim
    timescale = self.min_timescale * (self.max_timescale / self.min_timescale) ** fraction
    angle = positions[..., None, None].float() / timescale
    return angle.cos().to(dtype), angle.sin().to(dtype)

  @staticmethod
  def apply(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    first_half, second_half = x.chunk(2, dim=-1)
    return torch.cat((first_half * cos - second_half * sin, second_half * cos + first_half * sin), dim=-1)


def masked_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
  """Causal / strictly-causal SDPA where rows with no allowed keys return zero."""
  empty = ~allowed.any(-1)
  mask = (allowed | empty[..., None])[:, None]
  attn_out = F.scaled_dot_product_attention(
      q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask, scale=1.0
  ).transpose(1, 2)
  return attn_out.masked_fill(empty[..., None, None], 0)


class SwiGLU(nn.Module):
  """SwiGLU FFN with optional bounded SiTU limits (`situ_beta1`, `situ_beta2`; Paper Appendix D.1)."""

  def __init__(self, hidden_size: int, intermediate_size: int, situ_beta1: float = 0.0, situ_beta2: float = 0.0):
    super().__init__()
    self.gate = nn.Linear(hidden_size, intermediate_size, bias=False)
    self.up = nn.Linear(hidden_size, intermediate_size, bias=False)
    self.down = nn.Linear(intermediate_size, hidden_size, bias=False)
    self.situ_beta1 = float(situ_beta1)
    self.situ_beta2 = float(situ_beta2)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    gate, up = self.gate(x), self.up(x)
    if self.situ_beta1 > 0.0 and self.situ_beta2 > 0.0:
      gate_f32, up_f32 = gate.float(), up.float()
      act = (self.situ_beta1 * torch.tanh(gate_f32 / self.situ_beta1) * torch.sigmoid(gate_f32)) * (
          self.situ_beta2 * torch.tanh(up_f32 / self.situ_beta2)
      )
      return self.down(act.to(x.dtype))
    return self.down(F.silu(gate) * up)


class LoomEmbedWrite(nn.Module):
  """Initializes `M_0` `[B, T, d_k, d]` from token embeddings `x_0` via `MWrite` (Paper Eq. 10)."""

  def __init__(self, config: LoomFormerConfig):
    super().__init__()
    hidden_size, num_heads, head_dim = config.hidden_size, config.num_attention_heads, config.head_dim
    d_k, d_r = config.d_k, config.d_r
    self.num_heads = num_heads
    self.eps = config.write_eps
    self.divisor = config.embed_write_divisor
    self.W_e = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
    self.W_w1 = nn.Linear(hidden_size, d_r, bias=False)
    self.W_w2 = nn.Linear(d_r, num_heads * d_k, bias=True)
    self.W_wg = nn.Linear(hidden_size, num_heads, bias=True)

  def forward(self, x_0: torch.Tensor) -> torch.Tensor:
    content_raw = split_heads(self.W_e(x_0), self.num_heads)
    content_scaled = (
        (content_raw.float() / self.divisor).to(x_0.dtype)
        if self.divisor
        else rms_norm_unit(content_raw, self.eps)
    )
    write_key = rms_norm_unit(split_heads(self.W_w2(gelu_approx(self.W_w1(x_0))), self.num_heads), self.eps)
    gate = torch.sigmoid(self.W_wg(x_0))[..., None]
    return torch.einsum("btnk,btnd->btkd", write_key, gate * content_scaled)


class StandardMHA(nn.Module):
  """Standard Transformer MHA baseline (`loom_enabled=False`), supporting odd `head_dim=75` via `mha_nope_dim=1`."""

  def __init__(self, config: LoomFormerConfig, layer_idx: int):
    super().__init__()
    hidden_size, num_heads, head_dim = config.hidden_size, config.num_attention_heads, config.head_dim
    self.layer_idx = layer_idx
    self.num_heads = num_heads
    self.head_dim = head_dim
    self.nope_dim = config.mha_nope_dim
    self.query = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
    self.key = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
    self.value = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
    self.out = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
    self.rotary = RotaryEmbedding(head_dim - self.nope_dim, config.rope_min_timescale, config.rope_max_timescale)

  def forward(self, x: torch.Tensor, rope: Tuple[torch.Tensor, torch.Tensor], allowed: torch.Tensor,
              past_key_value: Optional[Cache] = None) -> torch.Tensor:
    cos, sin = rope
    rot = lambda proj: torch.cat(
        (proj[..., :self.nope_dim], RotaryEmbedding.apply(proj[..., self.nope_dim:], cos, sin)), dim=-1
    )
    q = rot(split_heads(self.query(x), self.num_heads))
    k = rot(split_heads(self.key(x), self.num_heads))
    v = split_heads(self.value(x), self.num_heads)
    q = q / torch.tensor(math.sqrt(self.head_dim), dtype=q.dtype)
    if past_key_value is not None:
      k, v = (
          cached.transpose(1, 2)
          for cached in past_key_value.update(k.transpose(1, 2), v.transpose(1, 2), self.layer_idx)
      )
    return self.out(masked_attention(q, k, v, allowed).flatten(-2))


class LoomMHA(nn.Module):
  """Loom Multi-Head Attention sub-layer on Llama backbone (Paper Section 3.2 Eqs. 5-8)."""

  def __init__(self, config: LoomFormerConfig, layer_idx: int):
    super().__init__()
    hidden_size, num_heads = config.hidden_size, config.num_attention_heads
    head_dim, d_k, d_r, rope_dim = config.head_dim, config.d_k, config.d_r, config.rope_dim
    self.layer_idx = layer_idx
    self.num_heads, self.head_dim, self.d_k, self.rope_dim = num_heads, head_dim, d_k, rope_dim
    self.eps = config.rms_norm_eps
    self.key_eps = config.read_key_eps
    self.key_scale = config.read_key_scale
    self.write_eps = config.write_eps
    self.static_init = config.static_read_gate_init

    self.W_sg = nn.Linear(hidden_size, d_k, bias=True)
    self.W_q = nn.Linear(hidden_size, num_heads * rope_dim, bias=False)
    self.W_k = nn.Linear(hidden_size, num_heads * rope_dim, bias=False)
    self.W_r1 = nn.Linear(hidden_size, d_r, bias=False)
    for arm in ("q", "k", "v", "o"):
      setattr(self, f"W_r2_{arm}", nn.Linear(d_r, num_heads * d_k, bias=True))
      setattr(self, f"W_rg_{arm}", nn.Linear(hidden_size, num_heads, bias=True))
      setattr(self, f"q_bar_{arm}", nn.Parameter(torch.zeros(num_heads, d_k)))
      setattr(self, f"W_rg_bar_{arm}", nn.Linear(hidden_size, num_heads, bias=True))
    self.c_qk = nn.Parameter(torch.ones(1))
    self.c_vo = nn.Parameter(torch.ones(1))
    self.rotary = RotaryEmbedding(rope_dim, config.rope_min_timescale, config.rope_max_timescale)
    self.W_o = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
    self.logit_scale = math.sqrt(head_dim)

    self.write_M = config.has_write(layer_idx)
    if self.write_M:
      self.W_w1 = nn.Linear(hidden_size, d_r, bias=False)
      self.W_w2 = nn.Linear(d_r, num_heads * d_k, bias=True)
      self.W_wg = nn.Linear(hidden_size, num_heads, bias=True)

  def _m_shift_and_norm(self, x: torch.Tensor, M: torch.Tensor, M_prev: torch.Tensor, shift_valid: torch.Tensor):
    shift_gate = torch.sigmoid(self.W_sg(x).float()).to(M.dtype)[..., None]
    mixed = (1.0 - shift_gate) * M + shift_gate * M_prev
    M_shifted = torch.where(shift_valid[..., None, None], mixed, M)
    M_f32 = M_shifted.float()
    return (M_f32 * torch.rsqrt(M_f32.pow(2).mean((-2, -1), keepdim=True) + self.eps)).to(M.dtype)

  def _m_read(self, M_tilde: torch.Tensor, arm: str, x: torch.Tensor, read_down: torch.Tensor, scale: torch.Tensor):
    read_key = rms_norm_unit(split_heads(getattr(self, f"W_r2_{arm}")(read_down), self.num_heads), self.key_eps)
    dyn_gate = (self.key_scale * torch.sigmoid(getattr(self, f"W_rg_{arm}")(x)))[..., None]
    read_dyn = torch.einsum("btnk,btkd->btnd", dyn_gate * read_key, M_tilde)
    stat_gate = ((1.1 / self.static_init) * torch.sigmoid(getattr(self, f"W_rg_bar_{arm}")(x)))[..., None]
    q_bar = getattr(self, f"q_bar_{arm}").to(M_tilde.dtype)
    read_stat = torch.einsum("nk,btkd->btnd", q_bar, M_tilde) * stat_gate
    return ((read_dyn + read_stat).float() * scale.float()).to(x.dtype)

  def forward(self, x, M, M_prev, shift_valid, rope, allowed, past_key_value: Optional[Cache] = None):
    M_tilde = self._m_shift_and_norm(x, M, M_prev, shift_valid)
    read_down = gelu_approx(self.W_r1(x))
    cos, sin = rope

    q_tilde = self._m_read(M_tilde, "q", x, read_down, self.c_qk)
    k_tilde = self._m_read(M_tilde, "k", x, read_down, self.c_qk)
    v = self._m_read(M_tilde, "v", x, read_down, self.c_vo)
    o_m = self._m_read(M_tilde, "o", x, read_down, self.c_vo)

    q_rope = RotaryEmbedding.apply(split_heads(self.W_q(x), self.num_heads), cos, sin)
    k_rope = RotaryEmbedding.apply(split_heads(self.W_k(x), self.num_heads), cos, sin)
    q = torch.cat((q_tilde, q_rope), dim=-1) / torch.tensor(self.logit_scale, dtype=x.dtype)
    k = torch.cat((k_tilde, k_rope), dim=-1)

    if past_key_value is not None:
      k_all, v_all = (
          cached.transpose(1, 2)
          for cached in past_key_value.update(k.transpose(1, 2), v.transpose(1, 2), self.layer_idx)
      )
    else:
      k_all, v_all = k, v

    attn_out = masked_attention(q, k_all, v_all, allowed) + o_m
    dx = self.W_o(attn_out.flatten(-2))
    if not self.write_M:
      return dx, None
    content = torch.sigmoid(self.W_wg(x))[..., None] * rms_norm_unit(attn_out, self.write_eps)
    write_key = rms_norm_unit(split_heads(self.W_w2(gelu_approx(self.W_w1(x))), self.num_heads), self.write_eps)
    return dx, (write_key, content)


class LoomFormerDecoderLayer(nn.Module):
  """Serial pre-LN LoomFormer / Transformer block (Paper Eq. 11)."""

  def __init__(self, config: LoomFormerConfig, layer_idx: int):
    super().__init__()
    hidden_size, num_heads = config.hidden_size, config.num_attention_heads
    self.loom_enabled = config.loom_enabled
    self.num_heads = num_heads
    self.write_eps = config.write_eps
    self.ln_1 = RMSNorm(hidden_size, config.rms_norm_eps)
    if self.loom_enabled:
      self.loom_mha = LoomMHA(config, layer_idx)
    else:
      self.self_attn = StandardMHA(config, layer_idx)
    self.ln_2 = RMSNorm(hidden_size, config.rms_norm_eps)
    self.ffn_x = SwiGLU(
        hidden_size, config.ffn_x_dim_of(layer_idx), situ_beta1=config.situ_beta1, situ_beta2=config.situ_beta2
    )
    self.write_M = config.has_write(layer_idx)
    if self.write_M:
      self.ffn_m = SwiGLU(hidden_size, config.ffn_m_dim)
      self.ffn_W_w1 = nn.Linear(hidden_size, config.d_r, bias=False)
      self.ffn_W_w2 = nn.Linear(config.d_r, num_heads * config.d_k, bias=True)
      self.ffn_W_wg = nn.Linear(hidden_size, num_heads, bias=True)
      self.ffn_divisor = config.ffn_write_divisor

  def _ffn_m(self, hidden: torch.Tensor, M: torch.Tensor, attn_write) -> torch.Tensor:
    """Memory FFN + memory write (`M_{l+1} = M_l + dM_attn + dM_ffn`)."""
    if attn_write is None:
      return M
    write_key_attn, content_attn = attn_write
    dM = torch.einsum("btnk,btnd->btkd", write_key_attn, content_attn)
    mem_content = split_heads((self.ffn_m(hidden).float() / self.ffn_divisor).to(hidden.dtype), self.num_heads)
    content_ffn = torch.sigmoid(self.ffn_W_wg(hidden))[..., None] * mem_content
    write_key_ffn = rms_norm_unit(
        split_heads(self.ffn_W_w2(gelu_approx(self.ffn_W_w1(hidden))), self.num_heads), self.write_eps
    )
    dM = dM + torch.einsum("btnk,btnd->btkd", write_key_ffn, content_ffn)
    return M + dM

  def forward(self, x, M, M_prev, shift_valid, rope, allowed, past_key_value=None):
    if not self.loom_enabled:
      x = x + self.self_attn(self.ln_1(x), rope, allowed, past_key_value)
      return x + self.ffn_x(self.ln_2(x)), None

    dx, attn_write = self.loom_mha(self.ln_1(x), M, M_prev, shift_valid, rope, allowed, past_key_value)
    x = x + dx
    hidden = self.ln_2(x)
    M = self._ffn_m(hidden, M, attn_write)
    x = x + self.ffn_x(hidden)
    return x, M


def _logit(prob: float) -> float:
  return math.log(prob / (1.0 - prob))


class _BF16LinearFP32Out(torch.autograd.Function):
  """Lossless FP32 logit accumulation with BF16 operands in forward & backward (matches JAX `bf16_dot_fp32_output`)."""

  @staticmethod
  def forward(ctx, hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    ctx.save_for_backward(hidden, weight)
    flat = hidden.reshape(-1, hidden.shape[-1])
    return torch.mm(flat, weight.t(), out_dtype=torch.float32).view(*hidden.shape[:-1], weight.shape[0])

  @staticmethod
  def backward(ctx, grad_out: torch.Tensor):
    hidden, weight = ctx.saved_tensors
    grad_flat = grad_out.to(hidden.dtype).reshape(-1, weight.shape[0])
    hidden_flat = hidden.reshape(-1, hidden.shape[-1])
    grad_hidden = torch.mm(grad_flat, weight).view_as(hidden)
    grad_weight = torch.mm(grad_flat.t(), hidden_flat)
    return grad_hidden, grad_weight


class LoomFormerPreTrainedModel(PreTrainedModel):
  config_class = LoomFormerConfig
  base_model_prefix = "model"
  _no_split_modules = ["LoomFormerDecoderLayer"]
  _supports_cache_class = True
  supports_gradient_checkpointing = False

  def _init_weights(self, module):
    cfg = self.config

    def _skip(param: torch.Tensor) -> bool:
      return bool(getattr(param, "_is_hf_initialized", False))

    if isinstance(module, (nn.Linear, nn.Embedding)):
      if not _skip(module.weight):
        module.weight.data.normal_(0.0, cfg.initializer_range)
      if getattr(module, "bias", None) is not None and not _skip(module.bias):
        module.bias.data.zero_()
    elif isinstance(module, RMSNorm):
      if not _skip(module.weight):
        module.weight.data.fill_(1.0)
    elif isinstance(module, LoomEmbedWrite):
      if not _skip(module.W_wg.bias):
        module.W_wg.bias.data.fill_(_logit(cfg.write_gate_init))
    elif isinstance(module, LoomMHA):
      if not _skip(module.W_sg.weight):
        module.W_sg.weight.data.zero_()
      if not _skip(module.W_sg.bias):
        module.W_sg.bias.data.fill_(_logit(cfg.m_shift_init))
      read_bias = _logit(cfg.read_gate_init)
      stat_bias = math.log(cfg.static_read_gate_init / (1.1 - cfg.static_read_gate_init))
      for arm in ("q", "k", "o", "v"):
        read_gate = getattr(module, f"W_rg_{arm}")
        static_read_gate = getattr(module, f"W_rg_bar_{arm}")
        if not _skip(read_gate.weight):
          read_gate.weight.data.zero_()
        if not _skip(read_gate.bias):
          read_gate.bias.data.fill_(read_bias)
        if not _skip(static_read_gate.weight):
          static_read_gate.weight.data.zero_()
        if not _skip(static_read_gate.bias):
          static_read_gate.bias.data.fill_(stat_bias)
        q_bar = getattr(module, f"q_bar_{arm}")
        if not _skip(q_bar):
          if arm == "v":
            q_bar.data.normal_(0.0, cfg.d_k ** -0.5)
          else:
            q_bar.data.zero_()
      if cfg.clone_vo_init and not _skip(module.W_r2_v.weight):
        module.W_r2_v.weight.data.copy_(module.W_r2_o.weight.data)
      if not _skip(module.c_qk):
        module.c_qk.data.fill_(1.0)
      if not _skip(module.c_vo):
        module.c_vo.data.fill_(1.0)
      if module.write_M and not _skip(module.W_wg.bias):
        module.W_wg.bias.data.fill_(_logit(cfg.write_gate_init))
    elif isinstance(module, LoomFormerDecoderLayer):
      if module.write_M and not _skip(module.ffn_W_wg.bias):
        module.ffn_W_wg.bias.data.fill_(_logit(cfg.write_gate_init))


class LoomFormerModel(LoomFormerPreTrainedModel):
  """LoomFormer / Transformer decoder stack."""

  def __init__(self, config: LoomFormerConfig):
    super().__init__(config)
    self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
    self.embed_write = LoomEmbedWrite(config) if config.loom_enabled else None
    self.layers = nn.ModuleList(LoomFormerDecoderLayer(config, i) for i in range(config.num_hidden_layers))
    self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
    self.post_init()

  def get_input_embeddings(self):
    return self.embed_tokens

  def set_input_embeddings(self, value):
    self.embed_tokens = value

  def forward(
      self,
      input_ids=None,
      attention_mask=None,
      position_ids=None,
      past_key_values=None,
      inputs_embeds=None,
      use_cache=None,
      output_hidden_states=None,
      return_dict=None,
      **kwargs,
  ):
    use_cache = self.config.use_cache if use_cache is None else use_cache
    loom_on = self.config.loom_enabled
    x = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
    if x.is_cuda and torch.is_autocast_enabled("cuda"):
      x = x.to(torch.get_autocast_dtype("cuda"))
    b, t = x.shape[:2]
    if use_cache and past_key_values is None:
      past_key_values = DynamicCache()
    cache = past_key_values if use_cache else None
    past = cache.get_seq_length() if cache is not None else 0

    if position_ids is None:
      if attention_mask is not None:
        position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)[:, -t:]
      else:
        position_ids = torch.arange(past, past + t, device=x.device)[None].expand(b, t)

    s = past + t
    keys, queries = torch.arange(s, device=x.device)[None, :], torch.arange(past, s, device=x.device)[:, None]
    allowed = ((keys < queries) if loom_on else (keys <= queries))[None]
    if attention_mask is not None:
      allowed = allowed & attention_mask[:, None, -s:].bool()

    attn0 = self.layers[0].loom_mha if loom_on else self.layers[0].self_attn
    rope = attn0.rotary.table(position_ids, x.dtype)
    hidden_states = (x,) if output_hidden_states else None

    if not loom_on:
      for layer in self.layers:
        x, _ = layer(x, None, None, None, rope, allowed, cache)
        if output_hidden_states:
          hidden_states += (x,)
      return BaseModelOutputWithPast(
          last_hidden_state=self.norm(x), past_key_values=cache, hidden_states=hidden_states
      )

    prev_m = getattr(cache, "loom_m", None) if past else None
    prev_pos = cache.loom_pos[:, None] if prev_m is not None else position_ids[:, :1]
    prev_pos = torch.cat((prev_pos, position_ids[:, :-1]), dim=1)
    shift_valid = position_ids == prev_pos + 1
    if prev_m is None:
      shift_valid[:, 0] = False

    M = self.embed_write(x)
    last_m = []
    for i, layer in enumerate(self.layers):
      first = prev_m[i][:, None] if prev_m is not None else M[:, :1]
      M_prev = torch.cat((first, M[:, :-1]), dim=1)
      last_m.append(M[:, -1])
      x, M = layer(x, M, M_prev, shift_valid, rope, allowed, cache)
      if output_hidden_states:
        hidden_states += (x,)
    if cache is not None:
      cache.loom_m = last_m
      cache.loom_pos = position_ids[:, -1]
    x = self.norm(x)
    return BaseModelOutputWithPast(last_hidden_state=x, past_key_values=cache, hidden_states=hidden_states)


class LoomFormerForCausalLM(LoomFormerPreTrainedModel, GenerationMixin):
  _tied_weights_keys = []

  def __init__(self, config: LoomFormerConfig):
    super().__init__(config)
    self.model = LoomFormerModel(config)
    self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    self.post_init()

  def get_input_embeddings(self):
    return self.model.embed_tokens

  def set_input_embeddings(self, value):
    self.model.embed_tokens = value

  def get_output_embeddings(self):
    return self.lm_head

  def set_output_embeddings(self, new_embeddings):
    self.lm_head = new_embeddings

  def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
    weight = self.lm_head.weight
    if hidden.dtype == torch.float32 or not hidden.is_cuda:
      return F.linear(hidden.float(), weight.float())
    weight_cast = weight.to(hidden.dtype)
    if torch.is_grad_enabled() and (hidden.requires_grad or weight_cast.requires_grad):
      return _BF16LinearFP32Out.apply(hidden, weight_cast)
    flat = hidden.reshape(-1, hidden.shape[-1])
    return torch.mm(flat, weight_cast.t(), out_dtype=torch.float32).view(*hidden.shape[:-1], -1)

  def forward(
      self,
      input_ids=None,
      attention_mask=None,
      position_ids=None,
      past_key_values=None,
      inputs_embeds=None,
      labels=None,
      use_cache=None,
      output_hidden_states=None,
      return_dict=None,
      logits_to_keep: int = 0,
      **kwargs,
  ) -> Union[Tuple, CausalLMOutputWithPast]:
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_hidden_states=output_hidden_states,
    )
    hidden = outputs.last_hidden_state
    logits = self.compute_logits(hidden[:, -logits_to_keep:] if logits_to_keep else hidden)
    loss = None
    if labels is not None:
      loss = F.cross_entropy(logits[:, :-1].flatten(0, 1), labels[:, 1:].flatten(), ignore_index=-100)
    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
    )

  def prepare_inputs_for_generation(
      self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, cache_position=None, **kwargs
  ):
    past = past_key_values.get_seq_length() if past_key_values is not None else 0
    if past > 0:
      input_ids = input_ids[:, past:]
    position_ids = None
    if attention_mask is not None:
      position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)[:, -input_ids.shape[1]:]
    model_inputs = (
        {"inputs_embeds": inputs_embeds} if inputs_embeds is not None and past == 0 else {"input_ids": input_ids}
    )
    model_inputs.update(
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=kwargs.get("use_cache", True),
    )
    return model_inputs

  def _reorder_cache(self, past_key_values, beam_idx):
    past_key_values.reorder_cache(beam_idx)
    if getattr(past_key_values, "loom_m", None) is not None:
      past_key_values.loom_m = [mem.index_select(0, beam_idx.to(mem.device)) for mem in past_key_values.loom_m]
      past_key_values.loom_pos = past_key_values.loom_pos.index_select(0, beam_idx.to(past_key_values.loom_pos.device))
    return past_key_values
