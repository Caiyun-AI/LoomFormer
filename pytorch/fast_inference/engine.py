"""Fast inference wrappers (`build(hf_model)`) using fused Triton kernels and static CUDA-graph-safe caches.

Supports:
  - `LoomPythiaForCausalLM` (`model_type == "loompythia"`) -> `FastLoomPythia`
  - `LoomFormerForCausalLM` (`model_type == "loomformer"`, `loom_enabled=True`) -> `FastLoomFormer`
  - `LoomFormerForCausalLM` (`model_type == "loomformer"`, `loom_enabled=False`) -> `FastMHA`
  - `GPTNeoXForCausalLM` (`model_type == "gpt_neox"`, stock Pythia baseline) -> `FastPythia`
"""
import math
import os
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from . import kernels_llama, kernels_pythia
from .cache import PaddedState, State
from .decode_attention import attend_and_append, decode_attention
from .prefill_attention import prefill_sdpa, strict_attention_cudnn, strict_causal_attention

SHARED_GEMV = os.environ.get("LOOM_SHARED_GEMV", "0") == "1"
USE_KERNELS = os.environ.get("LOOM_KERNELS", "1") == "1"


def _pad_rows(w: torch.Tensor, multiple: int = 64) -> torch.Tensor:
  pad = -w.shape[0] % multiple
  if not pad:
    return w
  return F.pad(w, (0, 0, 0, pad)) if w.dim() == 2 else F.pad(w, (0, pad))


def _cat_linear(linears, multiple: int = 64):
  w = torch.cat([l.weight for l in linears], dim=0)
  b = torch.cat([l.bias if l.bias is not None else l.weight.new_zeros(l.weight.shape[0]) for l in linears], dim=0)
  return nn.Parameter(_pad_rows(w, multiple), requires_grad=False), nn.Parameter(_pad_rows(b, multiple), requires_grad=False)


def _p(t: torch.Tensor) -> nn.Parameter:
  return nn.Parameter(t.detach().clone(), requires_grad=False)


def _ln(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
  return F.layer_norm(x.float(), (x.shape[-1],), w, b, eps).to(x.dtype)


def _rmsw(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
  f = x.float()
  return (f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps) * w).to(x.dtype)


def _rms(x: torch.Tensor, eps: float) -> torch.Tensor:
  f = x.float()
  return f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps)


def _rope_tables(dim: int, min_ts: float, max_ts: float, max_pos: int, device, dtype):
  frac = 2.0 * torch.arange(dim // 2, dtype=torch.float32) / dim
  timescale = min_ts * (max_ts / min_ts) ** frac
  ang = torch.arange(max_pos * 4, dtype=torch.float32)[:, None] / timescale
  return ang.cos().to(dtype).to(device), ang.sin().to(dtype).to(device), (1.0 / timescale).to(device)


def _rope_tail(x: torch.Tensor, start: int, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
  h = (x.shape[-1] - start) // 2
  a, b = x[..., start : start + h], x[..., start + h :]
  return torch.cat((x[..., :start], a * cos - b * sin, b * cos + a * sin), dim=-1)


def _swiglu(gu: torch.Tensor, inter: int, beta1: float = 0.0, beta2: float = 0.0) -> torch.Tensor:
  g, u = gu[..., :inter], gu[..., inter : 2 * inter]
  if beta1 > 0.0 and beta2 > 0.0:
    gf, uf = g.float(), u.float()
    return ((beta1 * torch.tanh(gf / beta1) * torch.sigmoid(gf)) * (beta2 * torch.tanh(uf / beta2))).to(gu.dtype)
  return F.silu(g) * u


def _padded_ffn(gate: nn.Linear, up: nn.Linear, down: nn.Linear, multiple: int = 64):
  pad = -gate.weight.shape[0] % multiple
  return (
      F.pad(gate.weight, (0, 0, 0, pad)),
      F.pad(up.weight, (0, 0, 0, pad)),
      F.pad(down.weight, (0, pad)),
      gate.weight.shape[0] + pad,
  )


@torch.library.custom_op("loom_llama::gemv_cols", mutates_args=())
def gemv_cols(
    srcs: list[torch.Tensor], starts: list[int], ws: list[torch.Tensor], bs: list[Optional[torch.Tensor]],
    acts: list[int], br: int, bk: int, warps: int,
) -> list[torch.Tensor]:
  xs = [x[:, a : a + w.shape[1]] for x, a, w in zip(srcs, starts, ws)]
  return kernels_pythia.gemv(xs, ws, bs, acts, br, bk, warps)


@gemv_cols.register_fake
def _(srcs, starts, ws, bs, acts, br, bk, warps):
  return [x.new_empty((1, w.shape[0])) for x, w in zip(srcs, ws)]


# ==============================================================================
# 1. Pythia Baseline (`GPTNeoXForCausalLM`) & `FastLoomPythia` (`LoomPythiaForCausalLM`)
# ==============================================================================

class FastPythia(nn.Module):
  """Stock HuggingFace `GPTNeoXForCausalLM` (parallel residual, partial NeoX RoPE)."""

  def __init__(self, hf):
    super().__init__()
    c = hf.config
    self.config = c
    m = hf.gpt_neox
    self.d, self.n, self.hd = c.hidden_size, c.num_attention_heads, c.hidden_size // c.num_attention_heads
    rope_params = getattr(c, "rope_parameters", None) or {}
    pct = rope_params.get("partial_rotary_factor", getattr(c, "rotary_pct", 0.25))
    theta = rope_params.get("rope_theta", getattr(c, "rotary_emb_base", 10000.0))
    self.rot = int(self.hd * pct)
    self.eps = c.layer_norm_eps
    self.embed = _p(m.embed_in.weight)
    self.lm_head = _p(hf.get_output_embeddings().weight)
    self.norm_w, self.norm_b = _p(m.final_layer_norm.weight.float()), _p(m.final_layer_norm.bias.float())
    inv = 1.0 / (theta ** (torch.arange(0, self.rot, 2, dtype=torch.float32) / self.rot))
    self.register_buffer("inv_freq", inv.to(self.embed.device), persistent=False)
    ang = torch.arange(c.max_position_embeddings * 4, dtype=torch.float32)[:, None] * inv
    self.register_buffer("rope_cos", ang.cos().to(self.embed.dtype).to(self.embed.device), persistent=False)
    self.register_buffer("rope_sin", ang.sin().to(self.embed.dtype).to(self.embed.device), persistent=False)
    self.layers = nn.ModuleList()
    for layer in m.layers:
      p = nn.Module()
      a = layer.attention
      w = a.query_key_value.weight.view(self.n, 3, self.hd, self.d).transpose(0, 1).reshape(3 * self.d, self.d)
      b = a.query_key_value.bias.view(self.n, 3, self.hd).transpose(0, 1).reshape(-1)
      p.qkv_w, p.qkv_b = _p(w), _p(b)
      p.dense_w, p.dense_b = _p(a.dense.weight), _p(a.dense.bias)
      p.fc_in_w, p.fc_in_b = _p(layer.mlp.dense_h_to_4h.weight), _p(layer.mlp.dense_h_to_4h.bias)
      p.fc_out_w, p.fc_out_b = _p(layer.mlp.dense_4h_to_h.weight), _p(layer.mlp.dense_4h_to_h.bias)
      p.out_cat_w = _p(torch.cat((p.dense_w, p.fc_out_w), dim=1))
      p.out_cat_b = _p(p.dense_b + p.fc_out_b)
      for name, ln in (("ln1", layer.input_layernorm), ("ln2", layer.post_attention_layernorm)):
        setattr(p, name + "_w", _p(ln.weight.float()))
        setattr(p, name + "_b", _p(ln.bias.float()))
      self.layers.append(p)

  def new_state(self, batch: int, max_len: int, layout: str = "rows") -> State:
    return State(len(self.layers), batch, self.n, max_len, self.hd, self.hd, dtype=self.embed.dtype,
                 device=self.embed.device, layout=layout)

  def _rope(self, x, cos, sin):
    r, h = self.rot, self.rot // 2
    a, b = x[..., :h], x[..., h:r]
    return torch.cat((a * cos - b * sin, b * cos + a * sin, x[..., r:]), dim=-1)

  def _forward(self, ids, state, decode: bool, last_only: bool = True):
    b, t = ids.shape
    x = F.embedding(ids, self.embed)
    pos = state.pos if state is not None else torch.zeros((), dtype=torch.long, device=ids.device)
    lengths = (pos + 1).to(torch.int32).expand(b).contiguous() if decode else None
    positions = pos + torch.arange(t, device=ids.device)
    cos, sin = self.rope_cos[positions][:, None], self.rope_sin[positions][:, None]
    scale = self.hd ** -0.5
    gemv = decode and b == 1 and SHARED_GEMV
    for i, p in enumerate(self.layers):
      if gemv:
        qkv, hpre = kernels_pythia.gemv(
            [x[:, 0], x[:, 0]], [p.qkv_w, p.fc_in_w], [p.qkv_b, p.fc_in_b], [0, 0], 8, 512, 4,
            ln_ws=[p.ln1_w, p.ln2_w], ln_bs=[p.ln1_b, p.ln2_b], ln_eps=self.eps,
        )
      else:
        qkv = F.linear(_ln(x, p.ln1_w, p.ln1_b, self.eps), p.qkv_w, p.qkv_b)
      q, k, v = qkv.view(b, t, 3, self.n, self.hd).unbind(2)
      q, k = self._rope(q, cos, sin), self._rope(k, cos, sin)
      if decode:
        y = attend_and_append((q * scale)[:, 0], state.k(i), state.v(i), k[:, 0], v[:, 0], pos, lengths, False, state.layout)[:, None]
      else:
        if state is not None:
          state.store_prefill(i, k.transpose(1, 2), v.transpose(1, 2))
        y = prefill_sdpa(q * scale, k, v, strict=False)
      if gemv:
        (x,) = kernels_pythia.gemv(
            [y.reshape(1, self.d)], [p.out_cat_w], [p.out_cat_b], [2], 8, 512, 8,
            x2s=[hpre], ress=[x[:, 0]], act_froms=[self.d],
        )
        x = x[:, None]
        continue
      attn = F.linear(y.reshape(b, t, self.d), p.dense_w, p.dense_b)
      h = _ln(x, p.ln2_w, p.ln2_b, self.eps)
      mlp = F.linear(F.gelu(F.linear(h, p.fc_in_w, p.fc_in_b)), p.fc_out_w, p.fc_out_b)
      x = x + attn + mlp
    if state is not None:
      state.pos.add_(t)
    x = _ln(x[:, -1:] if last_only else x, self.norm_w, self.norm_b, self.eps)
    return F.linear(x, self.lm_head)

  def prefill(self, ids, state=None, last_only: bool = True):
    return self._forward(ids, state, decode=False, last_only=last_only)

  def decode(self, ids, state):
    return self._forward(ids, state, decode=True)


class FastLoomPythia(nn.Module):
  """Fused Triton inference wrapper for `LoomPythiaForCausalLM` (e.g. `LoomPythia-2.8B`)."""

  def __init__(self, hf):
    super().__init__()
    c = hf.config
    self.config = c
    m = hf.model
    self.d, self.n = c.hidden_size, c.num_attention_heads
    self.kd, self.vd, self.s = c.head_dim, c.d_k, c.rope_dim
    self.eps, self.key_eps, self.write_eps = c.layer_norm_eps, c.read_key_eps, c.write_eps
    self.key_scale = c.read_key_scale
    self.static_scale = 1.1 / c.static_read_gate_init
    self.embed = _p(m.embed_tokens.weight)
    self.lm_head = _p(hf.lm_head.weight)
    self.norm_w, self.norm_b = _p(m.norm.weight.float()), _p(m.norm.bias.float())

    ew = m.embed_write
    self.e_w, self.e_b = _cat_linear([ew.W_e, ew.W_w1, ew.W_wg])
    self.e_up_w, self.e_up_b = _p(ew.W_w2.weight), _p(ew.W_w2.bias)
    self.e_tau, self.e_bottleneck = ew.divisor, ew.W_w1.weight.shape[0]

    cos, sin, inv_ts = _rope_tables(self.s, c.rope_min_timescale, c.rope_max_timescale, c.max_position_embeddings,
                                    self.embed.device, self.embed.dtype)
    self.register_buffer("inv_timescale", inv_ts, persistent=False)
    self.register_buffer("rope_cos", cos, persistent=False)
    self.register_buffer("rope_sin", sin, persistent=False)

    # Specialized 80x64 Triton fused_read requires (head_dim=80, d_k=64, rope_dim=20, BF16);
    # mini test configs automatically run the packed PyTorch fast path (`self.kernels = False`).
    self.kernels = (
        USE_KERNELS
        and self.embed.dtype == torch.bfloat16
        and (self.kd, self.vd, self.s) == (80, 64, 20)
        and self.n >= 16
        and (self.n & (self.n - 1)) == 0
    )

    self.layers = nn.ModuleList()
    for layer in m.layers:
      a = layer.loom_mha
      p = nn.Module()
      p.has_write = a.write_M
      p.has_mlp_write = layer.write_M
      xs = [a.W_sg, a.W_q, a.W_k, a.W_r1] + ([a.W_w1] if a.write_M else [])
      xs += [getattr(a, f"W_rg_{arm}") for arm in ("q", "k", "v", "o")]
      xs += [getattr(a, f"W_rg_bar_{arm}") for arm in ("q", "k", "v", "o")]
      xs += [a.W_wg] if a.write_M else []
      p.x_w, p.x_b = _cat_linear(xs)
      p.split_x = [l.weight.shape[0] for l in xs]

      ups = [getattr(a, f"W_r2_{arm}") for arm in ("q", "k", "v", "o")]
      p.up_w, p.up_b = _cat_linear(ups)
      # `q_bar_{arm}` in LoomPythiaMHA is already `[n, d_k]` -> stacked `[4, n, d_k]`
      p.static_key = _p(torch.stack([getattr(a, f"q_bar_{arm}").float() for arm in ("q", "k", "v", "o")]))
      p.static_key_rows = _p(p.static_key.flatten(0, 1).to(self.embed.dtype).contiguous())
      p.qk_scale, p.vo_scale = float(a.c_qk.float().item()), float(a.c_vo.float().item())
      p.q_norm, p.k_norm = _p(a.q_norm.weight), _p(a.k_norm.weight)
      p.norm_w = _p(torch.stack((a.q_norm.weight, a.k_norm.weight)).float())

      offsets = [0]
      for w in p.split_x:
        offsets.append(offsets[-1] + w)
      names = (
          ["shift", "q_std", "k_std", "read_down"]
          + (["write_down"] if a.write_M else [])
          + ["gates"] * 8
          + (["write_gate"] if a.write_M else [])
      )
      p.off_x = {}
      for name, o, w in zip(names, offsets, p.split_x):
        p.off_x[name] = (p.off_x[name][0], o + w) if name in p.off_x else (o, o + w)

      p.out_w, p.out_b = _p(a.W_o.weight), _p(a.W_o.bias)
      if a.write_M:
        p.wup_w, p.wup_b = _p(a.W_w2.weight), _p(a.W_w2.bias)

      hs = [layer.ffn_x.fc_in]
      if layer.write_M:
        hs += [layer.ffn_m.fc_in, layer.ffn_W_wg, layer.ffn_W_w1]
        p.mem_out_w = _p(layer.ffn_m.fc_out.weight)
        p.addr_up_w, p.addr_up_b = _p(layer.ffn_W_w2.weight), _p(layer.ffn_W_w2.bias)
        p.tau = layer.ffn_divisor
      p.h_w, p.h_b = _cat_linear(hs)
      p.split_h = [l.weight.shape[0] for l in hs]
      p.fc_out_w, p.fc_out_b = _p(layer.ffn_x.fc_out.weight), _p(layer.ffn_x.fc_out.bias)
      for name, ln in (("ln1", layer.ln_1), ("ln2", layer.ln_2)):
        setattr(p, name + "_w", _p(ln.weight.float()))
        setattr(p, name + "_b", _p(ln.bias.float()))
      self.layers.append(p)

  def new_state(self, batch: int, max_len: int, layout: Optional[str] = None) -> State:
    if layout is None:
      layout = "kt" if self.kernels else "rows"
    # Internal physical M buffer uses `[kd, vd]` (`[d, d_k]`) so inner stride `d_k=64` is 128-byte cache-line aligned
    return State(
        len(self.layers), batch, self.n, max_len, self.kd + self.s, self.kd, (self.kd, self.vd),
        dtype=self.embed.dtype, device=self.embed.device, layout=layout,
    )

  def _heads(self, x):
    return x.unflatten(-1, (self.n, -1))

  def _embedding_write(self, e):
    n, kd = self.n, self.kd
    z = F.linear(e, self.e_w, self.e_b)
    data = (z[..., : n * kd].float() / self.e_tau).to(e.dtype)
    hid = F.gelu(z[..., n * kd : n * kd + self.e_bottleneck], approximate="tanh")
    gate = torch.sigmoid(z[..., n * kd + self.e_bottleneck : n * kd + self.e_bottleneck + n])
    address = _rms(self._heads(F.linear(hid, self.e_up_w, self.e_up_b)), self.write_eps).to(e.dtype)
    content = gate[..., None] * self._heads(data)  # [B, T, n, d]
    return torch.matmul(content.transpose(-1, -2), address)  # [B, T, d, d_k] physical scratch

  def _rope(self, x, cos, sin):
    kd, h = self.kd, self.s // 2
    a, b = x[..., kd : kd + h], x[..., kd + h :]
    return torch.cat((x[..., :kd], a * cos - b * sin, b * cos + a * sin), dim=-1)

  def _layer(self, i, p, x, M, M_prev, valid, cos, sin, state, lengths, decode: bool):
    n, kd, vd = self.n, self.kd, self.vd
    xn = _ln(x, p.ln1_w, p.ln1_b, self.eps)
    z = F.linear(xn, p.x_w, p.x_b)
    parts = list(torch.split(z[..., : sum(p.split_x)], p.split_x, dim=-1))
    shift, q_std, k_std, read_down = parts[:4]
    rest = parts[4:]
    write_down = rest.pop(0) if p.has_write else None
    gates = torch.stack(rest[:4], dim=-2)
    static_gates = torch.stack(rest[4:8], dim=-2)
    write_gate = rest[8] if p.has_write else None

    up = F.linear(F.gelu(read_down, approximate="tanh"), p.up_w, p.up_b)[..., : 4 * n * vd].unflatten(-1, (4, n, vd))
    keys = (self.key_scale * torch.sigmoid(gates.float()))[..., None] * _rms(up, self.key_eps)
    keys = keys + (self.static_scale * torch.sigmoid(static_gates.float()))[..., None] * p.static_key
    s = torch.sigmoid(shift.float()).to(M.dtype)[..., None, :]
    mixed = torch.where(valid[..., None, None], (1.0 - s) * M + s * M_prev, M)
    f = mixed.float()
    inv = torch.rsqrt(f.pow(2).mean((-2, -1)) + self.eps)
    reads = torch.matmul(mixed, keys.to(M.dtype).flatten(2, 3).transpose(-1, -2))
    reads = (reads.float() * inv[..., None, None]).transpose(-1, -2).unflatten(-2, (4, n))
    q_loc, k_loc, value, local_o = reads.unbind(2)
    dt = x.dtype
    q = torch.cat(((q_loc * p.qk_scale).to(dt), self._heads(q_std)), dim=-1)
    k = torch.cat(((k_loc * p.qk_scale).to(dt), self._heads(k_std)), dim=-1)
    q = _rms(q, self.eps).to(dt) * p.q_norm
    k = _rms(k, self.eps).to(dt) * p.k_norm
    q = self._rope(q, cos, sin) / torch.tensor(math.sqrt(kd + self.s), dtype=dt)
    k = self._rope(k, cos, sin)
    value = (value * p.vo_scale).to(dt)
    local_o = (local_o * p.vo_scale).to(dt)
    if decode:
      y = attend_and_append(q[:, 0], state.k(i), state.v(i), k[:, 0], value[:, 0], state.pos, lengths, True, state.layout)[:, None]
    else:
      if state is not None:
        state.store_prefill(i, k.transpose(1, 2), value.transpose(1, 2))
      y = prefill_sdpa(q, k, value, strict=True)
    o_head = y + local_o
    x = x + F.linear(o_head.flatten(-2), p.out_w, p.out_b)
    h = _ln(x, p.ln2_w, p.ln2_b, self.eps)
    hz = F.linear(h, p.h_w, p.h_b)
    hparts = torch.split(hz[..., : sum(p.split_h)], p.split_h, dim=-1)
    mlp = F.linear(F.gelu(hparts[0]), p.fc_out_w, p.fc_out_b)
    if p.has_write:
      content = torch.sigmoid(write_gate)[..., None] * _rms(o_head, self.write_eps).to(dt)
      address = _rms(self._heads(F.linear(F.gelu(write_down, approximate="tanh"), p.wup_w, p.wup_b)), self.write_eps).to(dt)
      if p.has_mlp_write:
        mem_in, mgate, adown = hparts[1:]
        memory = self._heads(F.linear(F.gelu(mem_in), p.mem_out_w))
        if p.tau != 1.0:
          memory = (memory.float() / p.tau).to(dt)
        mlp_content = torch.sigmoid(mgate)[..., None] * memory
        mlp_address = _rms(self._heads(F.linear(F.gelu(adown, approximate="tanh"), p.addr_up_w, p.addr_up_b)), self.write_eps).to(dt)
        content = torch.cat((content, mlp_content), dim=-2)
        address = torch.cat((address, mlp_address), dim=-2)
      M = M + torch.matmul(content.transpose(-1, -2), address)
    return x + mlp, M

  def _forward(self, ids, state, decode: bool, last_only: bool = True):
    b, t = ids.shape
    x = F.embedding(ids, self.embed)
    pos = state.pos if state is not None else torch.zeros((), dtype=torch.long, device=ids.device)
    lengths = pos.to(torch.int32).expand(b).contiguous() if decode else None
    positions = pos + torch.arange(t, device=ids.device)
    ang = positions[:, None].float() * self.inv_timescale
    cos, sin = ang.cos().to(x.dtype)[:, None], ang.sin().to(x.dtype)[:, None]
    if decode:
      valid = (pos > 0).expand(b, t)
    else:
      valid = torch.ones(b, t, dtype=torch.bool, device=ids.device)
      valid[:, 0] = False
    M = self._embedding_write(x)
    for i, p in enumerate(self.layers):
      M_in = M
      M_prev = state.m(i)[:, None] if decode else torch.cat((M[:, :1], M[:, :-1]), dim=1)
      x, M = self._layer(i, p, x, M, M_prev, valid, cos, sin, state, lengths, decode)
      if state is not None:
        state.m(i).copy_(M_in[:, -1])
    if state is not None:
      state.pos.add_(t)
    x = _ln(x[:, -1:] if last_only else x, self.norm_w, self.norm_b, self.eps)
    if x.dtype == torch.float32:
      return F.linear(x, self.lm_head)
    return torch.mm(x.flatten(0, 1), self.lm_head.t(), out_dtype=torch.float32).view(b, x.shape[1], -1)

  def _forward_kernels(self, ids, state, decode: bool, last_only: bool = True):
    b, t = ids.shape
    rows = b * t
    n, kd, vd, d = self.n, self.kd, self.vd, self.d
    x = F.embedding(ids, self.embed).view(rows, d)
    pos = state.pos if state is not None else torch.zeros((), dtype=torch.long, device=ids.device)
    prev_valid = (pos > 0).to(torch.int32)
    lengths = pos.to(torch.int32).expand(b).contiguous() if decode else None
    M = self._embedding_write(x).reshape(rows, kd * vd)
    pending = None
    for i, p in enumerate(self.layers):
      small = decode and b == 1
      gemv = small and SHARED_GEMV
      if gemv:
        (z,) = kernels_pythia.gemv([x], [p.x_w], [p.x_b], [0], 8, 512, 4, ln_ws=[p.ln1_w], ln_bs=[p.ln1_b], ln_eps=self.eps)
      else:
        z = F.linear(_ln(x, p.ln1_w, p.ln1_b, self.eps), p.x_w, p.x_b)
      o = p.off_x
      rd = z[:, o["read_down"][0] : o["read_down"][1]]
      if small and p.has_write:
        up, wup = kernels_pythia.gemv(
            [rd, z[:, o["write_down"][0] : o["write_down"][1]]], [p.up_w, p.wup_w], [p.up_b, p.wup_b], [1, 1], 32, 128, 4
        )
      elif small:
        (up,) = kernels_pythia.gemv([rd], [p.up_w], [p.up_b], [1], 32, 128, 4)
        wup = None
      elif decode and p.has_write:
        assert o["write_down"][0] == o["read_down"][1]
        g = F.gelu(z[:, o["read_down"][0] : o["write_down"][1]], approximate="tanh")
        kr = p.up_w.shape[1]
        up, wup = kernels_pythia.grouped_gemm(
            [g[:, :kr], g[:, kr:]], [p.up_w, p.wup_w], [p.up_b, p.wup_b], [0, 0], None, 128, 64, 4 if rows <= 16 else 8, 3
        )
      else:
        up = F.linear(F.gelu(rd, approximate="tanh"), p.up_w, p.up_b)
        wup = None

      write = pending if pending is not None else (M,) * 6
      m_prev = state.m(i) if decode else M
      m_new, qk, vo = kernels_pythia.fused_read(
          M, *write, m_prev, prev_valid, pos, z[:, : o["shift"][1]], z[:, o["q_std"][0] : o["k_std"][1]], up,
          z[:, o["gates"][0] : o["gates"][1]], p.static_key_rows, p.norm_w, self.rope_cos, self.rope_sin,
          b, pending is not None, decode, self.write_eps, self.eps, self.key_eps, self.key_scale,
          self.static_scale, p.qk_scale, p.vo_scale, 1.0 / math.sqrt(kd + self.s), 1 if decode else 8,
          state.k(i) if decode else None, state.v(i) if decode else None, state.tail(i) if decode else None,
      )
      if pending is not None:
        M = m_new
      if state is not None:
        state.m(i).copy_(M.view(b, t, kd * vd)[:, -1].view(b, kd, vd))
      q, k = (qk[:, j, :, : kd + self.s].view(b, t, n, kd + self.s) for j in (0, 1))
      value, local_o = vo[:, 0].view(b, t, n, kd), vo[:, 1].view(b, t, n, kd)
      if decode:
        y = decode_attention(q[:, 0, :, None], state.k(i), state.v(i), lengths, state.layout, state.tail(i))[:, None, :, 0]
        o_head = (y + local_o).view(rows, n * kd)
      else:
        if state is not None:
          state.store_prefill(i, k.transpose(1, 2), value.transpose(1, 2))
        o_head = strict_causal_attention(q, k, value, local_o).view(rows, n * kd)

      sh = p.split_h
      if gemv:
        (x,) = kernels_pythia.gemv([o_head], [p.out_w], [p.out_b], [0], 4, 1024, 2, ress=[x])
        (hz,) = kernels_pythia.gemv([x], [p.h_w], [p.h_b], [0], 16, 256, 4, ln_ws=[p.ln2_w], ln_bs=[p.ln2_b], ln_eps=self.eps)
        if p.has_write:
          assert p.has_mlp_write and p.tau == 1.0
          o2, o3 = sh[0] + sh[1], sh[0] + sh[1] + sh[2]
          x, mem, addr = kernels_pythia.gemv(
              [hz[:, : sh[0]], hz[:, sh[0] : o2], hz[:, o3 : o3 + sh[3]]],
              [p.fc_out_w, p.mem_out_w, p.addr_up_w], [p.fc_out_b, None, p.addr_up_b], [2, 2, 1], 8, 1024, 8,
              ress=[x, None, None],
          )
          pending = (o_head, z[:, o["write_gate"][0] : o["write_gate"][1]], wup, mem, hz[:, o2:o3], addr)
        else:
          (x,) = kernels_pythia.gemv([hz[:, : sh[0]]], [p.fc_out_w], [p.fc_out_b], [2], 8, 1024, 8, ress=[x])
          pending = None
        continue

      x = x + F.linear(o_head, p.out_w, p.out_b)
      h = _ln(x, p.ln2_w, p.ln2_b, self.eps)
      if small:
        hz = F.linear(h, p.h_w, p.h_b)
        mlp = F.linear(F.gelu(hz[:, : sh[0]]), p.fc_out_w, p.fc_out_b)
        if p.has_write:
          assert p.has_mlp_write and p.tau == 1.0
          o2, o3 = sh[0] + sh[1], sh[0] + sh[1] + sh[2]
          mem, addr = kernels_pythia.gemv(
              [hz[:, sh[0] : o2], hz[:, o3 : o3 + sh[3]]], [p.mem_out_w, p.addr_up_w], [None, p.addr_up_b], [2, 1], 16, 1024, 8
          )
          pending = (o_head, z[:, o["write_gate"][0] : o["write_gate"][1]], wup, mem, hz[:, o2:o3], addr)
        else:
          pending = None
        x = x + mlp
        continue

      hz = F.linear(h, p.h_w, p.h_b)
      mlp = F.linear(F.gelu(hz[:, : sh[0]]), p.fc_out_w, p.fc_out_b)
      if p.has_write:
        assert p.has_mlp_write and p.tau == 1.0
        o2, o3 = sh[0] + sh[1], sh[0] + sh[1] + sh[2]
        mem_in, adown = F.gelu(hz[:, sh[0] : o2]), F.gelu(hz[:, o3 : o3 + sh[3]], approximate="tanh")
        if decode:
          mem, addr = kernels_pythia.grouped_gemm(
              [mem_in, adown], [p.mem_out_w, p.addr_up_w], [None, p.addr_up_b], [0, 0], None,
              32 if rows <= 16 else 64, 128, 4, 3,
          )
        else:
          mem, addr = F.linear(mem_in, p.mem_out_w), F.linear(adown, p.addr_up_w, p.addr_up_b)
        if wup is None:
          wd = z[:, o["write_down"][0] : o["write_down"][1]]
          wup = F.linear(F.gelu(wd, approximate="tanh"), p.wup_w, p.wup_b)
        pending = (o_head, z[:, o["write_gate"][0] : o["write_gate"][1]], wup, mem, hz[:, o2:o3], addr)
      else:
        pending = None
      x = x + mlp

    if state is not None:
      state.pos.add_(t)
    x = x.view(b, t, d)
    x = _ln(x[:, -1:] if last_only else x, self.norm_w, self.norm_b, self.eps)
    return torch.mm(x.flatten(0, 1), self.lm_head.t(), out_dtype=torch.float32).view(b, x.shape[1], -1)

  def prefill(self, ids, state=None, last_only: bool = True):
    if self.kernels:
      return self._forward_kernels(ids, state, decode=False, last_only=last_only)
    return self._forward(ids, state, decode=False, last_only=last_only)

  def decode(self, ids, state):
    if self.kernels:
      return self._forward_kernels(ids, state, decode=True)
    return self._forward(ids, state, decode=True)


# ==============================================================================
# 2. Llama-Backbone `FastMHA` & `FastLoomFormer` (`LoomFormerForCausalLM`)
# ==============================================================================

class _LlamaBase(nn.Module):

  def __init__(self, hf):
    super().__init__()
    c = hf.config
    self.config = c
    m = hf.model
    self.d, self.n, self.hd, self.eps = c.hidden_size, c.num_attention_heads, c.head_dim, c.rms_norm_eps
    self.beta1, self.beta2 = float(getattr(c, "situ_beta1", 0.0)), float(getattr(c, "situ_beta2", 0.0))
    self.embed = _p(m.embed_tokens.weight)
    self.lm_head = _p(hf.lm_head.weight)
    self.norm_w = _p(m.norm.weight.float())

  def _logits(self, x, b: int):
    x = _rmsw(x, self.norm_w, self.eps)
    if x.dtype == torch.float32:
      return F.linear(x, self.lm_head)
    return torch.mm(x.flatten(0, 1), self.lm_head.t(), out_dtype=torch.float32).view(b, x.shape[1], -1)

  def _mlp(self, p, h):
    return F.linear(_swiglu(F.linear(h, p.gu_w), p.inter, self.beta1, self.beta2), p.down_w)


class FastMHA(_LlamaBase):
  """Standard Transformer MHA baseline (`LoomFormerForCausalLM` with `loom_enabled=False`)."""

  def __init__(self, hf):
    super().__init__(hf)
    c = hf.config
    self.nope = c.mha_nope_dim
    cos, sin, _ = _rope_tables(self.hd - self.nope, c.rope_min_timescale, c.rope_max_timescale,
                               c.max_position_embeddings, self.embed.device, self.embed.dtype)
    self.register_buffer("rope_cos", cos, persistent=False)
    self.register_buffer("rope_sin", sin, persistent=False)
    self.layers = nn.ModuleList()
    for layer in hf.model.layers:
      a, p = layer.self_attn, nn.Module()
      p.qkv_w = _p(torch.cat((a.query.weight, a.key.weight, a.value.weight), dim=0))
      p.out_w = _p(a.out.weight)
      g, u, dw, p.inter = _padded_ffn(layer.ffn_x.gate, layer.ffn_x.up, layer.ffn_x.down)
      p.gu_w, p.down_w = _p(torch.cat((g, u), dim=0)), _p(dw)
      p.ln1_w, p.ln2_w = _p(layer.ln_1.weight.float()), _p(layer.ln_2.weight.float())
      self.layers.append(p)

  def new_state(self, batch: int, max_len: int, layout: str = "rows") -> PaddedState:
    return PaddedState(len(self.layers), batch, self.n, max_len, self.hd, self.hd, dtype=self.embed.dtype,
                       device=self.embed.device, layout=layout)

  def _forward(self, ids, state, decode: bool, last_only: bool = True):
    b, t = ids.shape
    n, hd = self.n, self.hd
    x = F.embedding(ids, self.embed)
    pos = state.pos if state is not None else torch.zeros((), dtype=torch.long, device=ids.device)
    lengths = (pos + 1).to(torch.int32).expand(b).contiguous() if decode else None
    positions = pos + torch.arange(t, device=ids.device)
    cos, sin = self.rope_cos[positions][:, None], self.rope_sin[positions][:, None]
    scale = hd ** -0.5
    for i, p in enumerate(self.layers):
      qkv = F.linear(_rmsw(x, p.ln1_w, self.eps), p.qkv_w)
      q, k, v = qkv.view(b, t, 3, n, hd).unbind(2)
      q, k = _rope_tail(q, self.nope, cos, sin), _rope_tail(k, self.nope, cos, sin)
      if decode:
        y = attend_and_append((q * scale)[:, 0], state.k(i), state.v(i), k[:, 0], v[:, 0], pos, lengths, False, state.layout)[:, None]
      else:
        if state is not None:
          state.store_prefill(i, k.transpose(1, 2), v.transpose(1, 2))
        y = prefill_sdpa(q * scale, k, v, strict=False)
      x = x + F.linear(y.reshape(b, t, n * hd), p.out_w)
      x = x + self._mlp(p, _rmsw(x, p.ln2_w, self.eps))
    if state is not None:
      state.pos.add_(t)
    return self._logits(x[:, -1:] if last_only else x, b)

  def prefill(self, ids, state=None, last_only: bool = True):
    return self._forward(ids, state, decode=False, last_only=last_only)

  def decode(self, ids, state):
    return self._forward(ids, state, decode=True)


class FastLoomFormer(_LlamaBase):
  """Fused Triton inference wrapper for `LoomFormerForCausalLM` (Medium / Large / XL / XL-SiTU)."""

  def __init__(self, hf):
    super().__init__(hf)
    c = hf.config
    m = hf.model
    self.kd, self.vd, self.s = c.head_dim, c.d_k, c.rope_dim
    self.key_eps, self.write_eps, self.key_scale = c.read_key_eps, c.write_eps, c.read_key_scale
    self.static_scale = 1.1 / c.static_read_gate_init

    ew = m.embed_write
    self.e_w, self.e_b = _cat_linear([ew.W_e, ew.W_w1, ew.W_wg])
    self.e_up_w, self.e_up_b = _p(ew.W_w2.weight), _p(ew.W_w2.bias)
    self.e_tau, self.e_bottleneck = ew.divisor, ew.W_w1.weight.shape[0]

    cos, sin, _ = _rope_tables(self.s, c.rope_min_timescale, c.rope_max_timescale,
                               c.max_position_embeddings, self.embed.device, self.embed.dtype)
    self.register_buffer("rope_cos", cos, persistent=False)
    self.register_buffer("rope_sin", sin, persistent=False)

    # Generic Triton kernel requires head_dim >= 16 and BF16
    self.kernels = USE_KERNELS and self.embed.dtype == torch.bfloat16 and self.kd >= 16

    self.layers = nn.ModuleList()
    for layer in m.layers:
      a, p = layer.loom_mha, nn.Module()
      p.has_write, p.has_mlp_write = a.write_M, layer.write_M
      xs = [a.W_sg, a.W_q, a.W_k, a.W_r1] + ([a.W_w1] if a.write_M else [])
      xs += [getattr(a, f"W_rg_{arm}") for arm in ("q", "k", "v", "o")]
      xs += [getattr(a, f"W_rg_bar_{arm}") for arm in ("q", "k", "v", "o")]
      xs += [a.W_wg] if a.write_M else []
      p.x_w, p.x_b = _cat_linear(xs)
      p.split_x = [l.weight.shape[0] for l in xs]
      names = ["shift", "std", "std", "read_down"] + (["write_down"] if a.write_M else []) + ["gates"] * 8
      names += ["write_gate"] if a.write_M else []
      p.off_x, o = {}, 0
      for name, w in zip(names, p.split_x):
        p.off_x[name] = (p.off_x.get(name, (o,))[0], o + w)
        o += w

      p.up_w, p.up_b = _cat_linear([getattr(a, f"W_r2_{arm}") for arm in ("q", "k", "v", "o")])
      p.static_key = _p(torch.stack([getattr(a, f"q_bar_{arm}").float() for arm in ("q", "k", "v", "o")]))
      p.static_key_rows = _p(p.static_key.flatten(0, 1).to(self.embed.dtype).contiguous())
      p.qk_scale, p.vo_scale = float(a.c_qk.float().item()), float(a.c_vo.float().item())
      p.out_w = _p(a.W_o.weight)
      if a.write_M:
        p.wup_w, p.wup_b = _p(a.W_w2.weight), _p(a.W_w2.bias)

      g, u, dw, p.inter = _padded_ffn(layer.ffn_x.gate, layer.ffn_x.up, layer.ffn_x.down)
      hs = [g, u]
      hb = [g.new_zeros(2 * p.inter)]
      if layer.write_M:
        mg, mu, mdw, p.mem_inter = _padded_ffn(layer.ffn_m.gate, layer.ffn_m.up, layer.ffn_m.down)
        hs += [mg, mu, layer.ffn_W_wg.weight, layer.ffn_W_w1.weight]
        hb += [g.new_zeros(2 * p.mem_inter), layer.ffn_W_wg.bias, g.new_zeros(layer.ffn_W_w1.weight.shape[0])]
        p.mem_down_w = _p(mdw)
        p.addr_up_w, p.addr_up_b = _p(layer.ffn_W_w2.weight), _p(layer.ffn_W_w2.bias)
        p.tau = layer.ffn_divisor
      p.split_h = [w.shape[0] for w in hs]
      p.h_w, p.h_b = _p(_pad_rows(torch.cat(hs, dim=0))), _p(_pad_rows(torch.cat(hb, dim=0)))
      p.down_w = _p(dw)
      p.ln1_w, p.ln2_w = _p(layer.ln_1.weight.float()), _p(layer.ln_2.weight.float())

      ox = p.off_x
      p.xb = _p(torch.cat((p.x_b[: ox["shift"][1]], p.x_b[ox["gates"][0] : ox["gates"][1]])))
      if a.write_M:
        o3 = 2 * p.inter + 2 * p.mem_inter
        p.wb = _p(torch.cat((p.x_b[ox["write_gate"][0] : ox["write_gate"][1]], p.h_b[o3 : o3 + p.split_h[4]])))
      self.layers.append(p)

  def new_state(self, batch: int, max_len: int, layout: str = "rows") -> PaddedState:
    return PaddedState(
        len(self.layers), batch, self.n, max_len, self.kd + self.s, self.kd, (self.kd, self.vd),
        dtype=self.embed.dtype, device=self.embed.device, layout=layout,
    )

  def _heads(self, x):
    return x.unflatten(-1, (self.n, -1))

  def _embedding_write(self, e):
    n, kd = self.n, self.kd
    z = F.linear(e, self.e_w, self.e_b)
    data = (z[..., : n * kd].float() / self.e_tau).to(e.dtype)
    hid = F.gelu(z[..., n * kd : n * kd + self.e_bottleneck], approximate="tanh")
    gate = torch.sigmoid(z[..., n * kd + self.e_bottleneck : n * kd + self.e_bottleneck + n])
    address = _rms(self._heads(F.linear(hid, self.e_up_w, self.e_up_b)), self.write_eps).to(e.dtype)
    return torch.matmul((gate[..., None] * self._heads(data)).transpose(-1, -2), address)

  def _layer(self, i, p, x, M, M_prev, valid, cos, sin, state, lengths, decode: bool):
    n, kd, vd, dt = self.n, self.kd, self.vd, x.dtype
    z = F.linear(_rmsw(x, p.ln1_w, self.eps), p.x_w, p.x_b)
    parts = list(torch.split(z[..., : sum(p.split_x)], p.split_x, dim=-1))
    shift, q_std, k_std, read_down = parts[:4]
    rest = parts[4:]
    write_down = rest.pop(0) if p.has_write else None
    gates, static_gates = torch.stack(rest[:4], dim=-2), torch.stack(rest[4:8], dim=-2)
    write_gate = rest[8] if p.has_write else None
    up = F.linear(F.gelu(read_down, approximate="tanh"), p.up_w, p.up_b)[..., : 4 * n * vd].unflatten(-1, (4, n, vd))
    keys = (self.key_scale * torch.sigmoid(gates.float()))[..., None] * _rms(up, self.key_eps)
    keys = keys + (self.static_scale * torch.sigmoid(static_gates.float()))[..., None] * p.static_key
    s = torch.sigmoid(shift.float()).to(M.dtype)[..., None, :]
    mixed = torch.where(valid[..., None, None], (1.0 - s) * M + s * M_prev, M)
    f = mixed.float()
    inv = torch.rsqrt(f.pow(2).mean((-2, -1)) + self.eps)
    reads = torch.matmul(mixed, keys.to(M.dtype).flatten(2, 3).transpose(-1, -2))
    reads = (reads.float() * inv[..., None, None]).transpose(-1, -2).unflatten(-2, (4, n))
    q_loc, k_loc, value, local_o = reads.unbind(2)
    q = torch.cat(((q_loc * p.qk_scale).to(dt), self._heads(q_std)), dim=-1)
    k = torch.cat(((k_loc * p.qk_scale).to(dt), self._heads(k_std)), dim=-1)
    q = _rope_tail(q, kd, cos, sin) / torch.tensor(math.sqrt(self.hd), dtype=dt)
    k = _rope_tail(k, kd, cos, sin)
    value, local_o = (value * p.vo_scale).to(dt).contiguous(), (local_o * p.vo_scale).to(dt)
    if decode:
      y = attend_and_append(q[:, 0], state.k(i), state.v(i), k[:, 0], value[:, 0], state.pos, lengths, True, state.layout)[:, None]
    else:
      if state is not None:
        state.store_prefill(i, k.transpose(1, 2), value.transpose(1, 2))
      y = prefill_sdpa(q, k, value, strict=True)
    o_head = y + local_o
    x = x + F.linear(o_head.flatten(-2), p.out_w)
    hz = F.linear(_rmsw(x, p.ln2_w, self.eps), p.h_w, p.h_b)
    sh = p.split_h
    mlp = F.linear(_swiglu(hz, p.inter, self.beta1, self.beta2), p.down_w)
    if p.has_write:
      content = torch.sigmoid(write_gate)[..., None] * _rms(o_head, self.write_eps).to(dt)
      address = _rms(self._heads(F.linear(F.gelu(write_down, approximate="tanh"), p.wup_w, p.wup_b)), self.write_eps).to(dt)
      if p.has_mlp_write:
        o = 2 * p.inter
        mem = F.linear(_swiglu(hz[..., o : o + 2 * p.mem_inter], p.mem_inter), p.mem_down_w)
        memory = self._heads(mem if p.tau == 1.0 else (mem.float() / p.tau).to(dt))
        o += 2 * p.mem_inter
        mgate, adown = hz[..., o : o + sh[4]], hz[..., o + sh[4] : o + sh[4] + sh[5]]
        mlp_content = torch.sigmoid(mgate)[..., None] * memory
        mlp_address = _rms(self._heads(F.linear(F.gelu(adown, approximate="tanh"), p.addr_up_w, p.addr_up_b)), self.write_eps).to(dt)
        content, address = torch.cat((content, mlp_content), dim=-2), torch.cat((address, mlp_address), dim=-2)
      M = M + torch.matmul(content.transpose(-1, -2), address)
    return x + mlp, M

  def _forward(self, ids, state, decode: bool, last_only: bool = True):
    b, t = ids.shape
    x = F.embedding(ids, self.embed)
    pos = state.pos if state is not None else torch.zeros((), dtype=torch.long, device=ids.device)
    lengths = pos.to(torch.int32).expand(b).contiguous() if decode else None
    positions = pos + torch.arange(t, device=ids.device)
    cos, sin = self.rope_cos[positions][:, None], self.rope_sin[positions][:, None]
    if decode:
      valid = (pos > 0).expand(b, t)
    else:
      valid = torch.ones(b, t, dtype=torch.bool, device=ids.device)
      valid[:, 0] = False
    M = self._embedding_write(x)
    for i, p in enumerate(self.layers):
      M_in = M
      M_prev = state.m_prev(i, pos)[:, None] if decode else torch.cat((M[:, :1], M[:, :-1]), dim=1)
      x, M = self._layer(i, p, x, M, M_prev, valid, cos, sin, state, lengths, decode)
      if state is not None:
        state.m_store(i, pos + t - 1, M_in[:, -1])
    if state is not None:
      state.pos.add_(t)
    return self._logits(x[:, -1:] if last_only else x, b)

  def _forward_kernels(self, ids, state, decode: bool, last_only: bool = True):
    b, t = ids.shape
    rows = b * t
    n, kd, vd, s, d = self.n, self.kd, self.vd, self.s, self.d
    x = F.embedding(ids, self.embed).view(rows, d)
    pos = state.pos if state is not None else torch.zeros((), dtype=torch.long, device=ids.device)
    prev_valid = (pos > 0).to(torch.int32)
    lengths = pos.to(torch.int32).expand(b).contiguous() if decode else None
    M = self._embedding_write(x).reshape(rows, kd * vd)
    q_mult = 1.0 / math.sqrt(self.hd)
    if not decode:
      sub, groups, warps = 1, 2, 2
    elif rows == 1:
      sub, groups, warps = 4 if n > 16 else 2, 1, 4
    else:
      sub, groups, warps = (2, 1, 4) if rows <= 16 else (1, 2, 4)
    pending = None
    for i, p in enumerate(self.layers):
      z = F.linear(_rmsw(x, p.ln1_w, self.eps), p.x_w)
      o = p.off_x
      rd = z[:, o["read_down"][0] : o["read_down"][1]]
      wup = None
      if decode and rows == 1:
        if p.has_write:
          up, wup = kernels_llama.gemv2(
              z, o["read_down"][0], 0, p.up_w, p.up_b, 1, z, o["write_down"][0], 0, p.wup_w, p.wup_b, 1, 8, 256, 4
          )
        else:
          (up,) = gemv_cols([z], [o["read_down"][0]], [p.up_w], [p.up_b], [1], 16, 256, 4)
      elif decode and p.has_write:
        g = F.gelu(z[:, o["read_down"][0] : o["write_down"][1]], approximate="tanh")
        kr = p.up_w.shape[1]
        up, wup = kernels_pythia.grouped_gemm(
            [g[:, :kr], g[:, kr:]], [p.up_w, p.wup_w], [p.up_b, p.wup_b], [0, 0], None, 128, 64, 4 if rows <= 16 else 8, 3
        )
      else:
        up = F.linear(F.gelu(rd, approximate="tanh"), p.up_w, p.up_b)
      write, wb, (g1_off, g2_off), c1_vo = (
          (pending[:6], pending[6], pending[7:9], pending[9]) if pending is not None else ((M,) * 6, p.xb, (0, 0), None)
      )
      m_new, qk, vo = kernels_llama.fused_read(
          M, *write, state.m(i) if decode else M, prev_valid, pos, z, z, up, z, p.static_key_rows, self.rope_cos,
          self.rope_sin, p.xb, wb, b, n, kd, vd, s, pending is not None, decode, self.write_eps, self.eps, self.key_eps,
          self.key_scale, self.static_scale, p.qk_scale, p.vo_scale, q_mult, sub, groups, warps,
          state.k(i) if decode else None, state.v(i) if decode else None, g1_off, g2_off, o["shift"][0],
          o["std"][0], o["gates"][0], c1_vo,
      )
      if pending is not None:
        M = m_new
      if state is not None and not (decode and pending is not None):
        state.m_store(i, pos + t - 1, M.view(b, t, kd, vd)[:, -1])
      qw, vw = qk.shape[-1], vo.shape[-1]
      q, k = (qk[:, j].view(b, t, n, qw) for j in (0, 1))
      value, local_o = (vo[:, j].view(b, t, n, vw)[..., :kd] for j in (0, 1))
      if decode:
        y = decode_attention(q[:, 0, :, None, : kd + s], state.k(i), state.v(i), lengths, state.layout)[:, None, :, 0]
      else:
        if state is not None:
          state.store_prefill(i, k[..., : kd + s].transpose(1, 2), value.transpose(1, 2))
        y = strict_attention_cudnn(q, k, vo[:, 0].view(b, t, n, vw))[..., :kd]
      o_head = (y + local_o).reshape(rows, n * kd)
      x = x + F.linear(o_head, p.out_w)
      c1, c1_vo = (y.reshape(rows, n * kd), vo) if decode else (o_head, None)
      hz = F.linear(_rmsw(x, p.ln2_w, self.eps), p.h_w)
      mlp = F.linear(_swiglu(hz, p.inter, self.beta1, self.beta2), p.down_w)
      if p.has_write:
        assert p.has_mlp_write and p.tau == 1.0
        sh, o2 = p.split_h, 2 * p.inter
        o3 = o2 + 2 * p.mem_inter
        mem_in, mgate, adown = (
            _swiglu(hz[:, o2:o3], p.mem_inter),
            hz[:, o3 : o3 + sh[4]],
            hz[:, o3 + sh[4] : o3 + sh[4] + sh[5]],
        )
        if decode and rows == 1:
          mem, addr = kernels_llama.gemv2(
              hz, o2, p.mem_inter, p.mem_down_w, None, 3, hz, o3 + sh[4], 0, p.addr_up_w, p.addr_up_b, 1, 8, 1024, 8
          )
        elif decode:
          mem, addr = kernels_pythia.grouped_gemm(
              [mem_in, F.gelu(adown, approximate="tanh")], [p.mem_down_w, p.addr_up_w], [None, p.addr_up_b], [0, 0], None,
              32 if rows <= 16 else 64, 128, 4, 3,
          )
        else:
          mem = F.linear(mem_in, p.mem_down_w)
          addr = F.linear(F.gelu(adown, approximate="tanh"), p.addr_up_w, p.addr_up_b)
        if wup is None:
          wd = z[:, o["write_down"][0] : o["write_down"][1]]
          wup = F.linear(F.gelu(wd, approximate="tanh"), p.wup_w, p.wup_b)
        pending = (c1, z, wup, mem, hz, addr, p.wb, o["write_gate"][0], o3, c1_vo)
      else:
        pending = None
      x = x + mlp
    if state is not None:
      state.pos.add_(t)
    x = x.view(b, t, d)
    return self._logits(x[:, -1:] if last_only else x, b)

  def prefill(self, ids, state=None, last_only: bool = True):
    if self.kernels:
      return self._forward_kernels(ids, state, decode=False, last_only=last_only)
    return self._forward(ids, state, decode=False, last_only=last_only)

  def decode(self, ids, state):
    if self.kernels:
      return self._forward_kernels(ids, state, decode=True)
    return self._forward(ids, state, decode=True)


def build(hf):
  """Wrap a loaded HuggingFace model (`LoomPythiaForCausalLM`, `LoomFormerForCausalLM`, or `GPTNeoXForCausalLM`)."""
  mt = hf.config.model_type
  if mt == "loompythia":
    return FastLoomPythia(hf).eval()
  if mt == "gpt_neox":
    return FastPythia(hf).eval()
  if mt == "loomformer":
    return (FastLoomFormer if hf.config.loom_enabled else FastMHA)(hf).eval()
  raise ValueError(f"Unsupported model_type for fast_inference.build: {mt!r}")
