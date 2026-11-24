"""LoomFormer (Llama RMSNorm / SwiGLU backbone) in JAX/Flax on official `maxtext-v0.2.0`.

Every module follows a PyTorch-familiar `setup()` (define parameters & submodules) + `__call__()`
(forward pass) structure, aligned with Paper Section 3.2 and Appendix C (`loomformer_pseudo.py`).

Subscript notation used in `einsum` / shape comments:
  b = batch size, t = seq len, D = emb_dim (n * d), m = port bottleneck width d_r
  n = num_heads, d = head_dim, k = matrix slots d_k, r = RoPE dim (floor(d / 4))
  h = full Q/K head width (d + r). Per-token matrix memory M has shape [b, t, k, d] (btkd).
"""
import functools
import math
from typing import Any, Optional

from flax import linen as nn
import jax
from jax import lax
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp

from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.layers import decoders, embeddings, linears, nnx_wrappers
from maxtext.layers.initializers import variable_to_logically_partitioned
from maxtext.models import models

from . import loom_pallas
from .config import LoomConfig

Array = jax.Array
NEG_INF = -1e9


# ==============================================================================
# 1. Parameter-Free Norms, Activations & Helper Builders
# ==============================================================================

def rms_norm_unit(x: Array, eps: float) -> Array:
  """Parameter-free RMSNorm over the last axis with FP32 statistics (Paper Eqs. 5, 6, 8)."""
  f = x.astype(jnp.float32)
  return (f * lax.rsqrt(jnp.mean(jnp.square(f), axis=-1, keepdims=True) + eps)).astype(x.dtype)


def m_norm(M: Array, eps: float) -> Array:
  """Whole-matrix RMSNorm over `(-2, -1) = (k, d)` of `M` `[b, t, k, d]` (Paper Eq. 7)."""
  f = M.astype(jnp.float32)
  return (f * lax.rsqrt(jnp.mean(jnp.square(f), axis=(-2, -1), keepdims=True) + eps)).astype(M.dtype)


def situ_gate(x: Array, beta: float) -> Array:
  """Bounded SiTU gate `beta * tanh(x / beta) * sigmoid(x)` (Paper Appendix D.1)."""
  xf = x.astype(jnp.float32)
  return (beta * jnp.tanh(xf / beta) * jax.nn.sigmoid(xf)).astype(x.dtype)


def situ_up(x: Array, beta: float) -> Array:
  """Bounded SiTU up-branch `beta * tanh(x / beta)` (Paper Appendix D.1)."""
  xf = x.astype(jnp.float32)
  return (beta * jnp.tanh(xf / beta)).astype(x.dtype)


@jax.custom_vjp
def bf16_dot_fp32_output(hidden: Array, kernel: Array) -> Array:
  """Keep native TPU MXU FP32 accumulator output while keeping BF16 operands in forward & backward."""
  if hidden.dtype != jnp.bfloat16 or kernel.dtype != jnp.bfloat16:
    raise TypeError("bf16_dot_fp32_output requires BF16 hidden states and kernel")
  return jnp.matmul(hidden, kernel, preferred_element_type=jnp.float32)


def _bf16_dot_fwd(hidden: Array, kernel: Array):
  return bf16_dot_fp32_output(hidden, kernel), (hidden, kernel)


def _bf16_dot_bwd(residual, gradient: Array):
  hidden, kernel = residual
  g = gradient.astype(jnp.bfloat16)
  dh = jnp.tensordot(g, kernel, axes=((-1,), (-1,)))
  tok_axes = tuple(range(hidden.ndim - 1))
  dw = jnp.tensordot(g, hidden, axes=(tok_axes, tok_axes)).T
  return dh, dw


bf16_dot_fp32_output.defvjp(_bf16_dot_fwd, _bf16_dot_bwd)


def m_shift_valid_mask(shape, segment_ids=None, positions=None) -> Array:
  """Valid previous-token mask `[b, t]` for `MShift` (excludes `t = 0` and packed segment boundaries)."""
  valid = jnp.broadcast_to(jnp.arange(shape[1])[None, :] > 0, shape[:2])
  if segment_ids is not None:
    prev_seg = jnp.concatenate((segment_ids[:, :1], segment_ids[:, :-1]), axis=1)
    valid &= (segment_ids == prev_seg) & (segment_ids != 0)
  if positions is not None:
    prev_pos = jnp.concatenate((positions[:, :1], positions[:, :-1]), axis=1)
    valid &= (positions == prev_pos + 1)
  return valid


def logit(p: float) -> float:
  return math.log(p / (1.0 - p))


def replicate_rank_up(lc: LoomConfig, num_heads: int, num_layers: int, mesh: Any, rank: int) -> bool:
  mode = str(lc.loom_replicate_rank_up).lower()
  if mode != "auto":
    return mode == "true"
  fsdp = mesh.shape.get("fsdp", 1) * mesh.shape.get("fsdp_transpose", 1)
  if fsdp <= 1 or rank >= 64 * fsdp:
    return False
  return (num_layers * 6 * rank * num_heads * lc.loom_d_k * 16 / (2 ** 30)) <= 2.0


def dense_proj(features, init, axes: tuple[Optional[str], ...], cfg: Any, *,
               in_features=None, axis=-1, use_bias=False, name: str):
  """Configure official `maxtext.layers.linears.dense_general`."""
  return linears.dense_general(
      in_features_shape=cfg.emb_dim if in_features is None else in_features,
      out_features_shape=features,
      axis=axis,
      weight_dtype=cfg.weight_dtype,
      dtype=cfg.dtype,
      kernel_init=lambda k, s, dt, *_: init(k, s, dt),
      kernel_axes=axes,
      use_bias=use_bias,
      shard_mode=cfg.shard_mode,
      matmul_precision=cfg.matmul_precision,
      name=name,
  )


def rms_norm_layer(cfg: Any, name: str):
  """Standard `flax.linen.RMSNorm` with FP32 statistics."""
  return nn.RMSNorm(
      epsilon=cfg.normalization_layer_epsilon,
      dtype=cfg.dtype,
      param_dtype=cfg.weight_dtype,
      scale_init=nn.with_logical_partitioning(nn.initializers.ones, ("norm",)),
      name=name,
  )


def strict_causal_attention(query: Array, key: Array, value: Array, segment_ids,
                            cfg: Any, lc: LoomConfig, mesh: Any,
                            window: int, seq_minor_default: bool) -> Array:
  """Strictly causal (`s < t`, row `t = 0` returns zero) horizontal attention (Paper Eq. 8)."""
  t = query.shape[1]
  if lc.loom_splash and t % 128 == 0 and jax.default_backend() == "tpu":
    seq_minor = seq_minor_default if lc.loom_splash_seq_minor is None else bool(lc.loom_splash_seq_minor)
    return loom_pallas.strict_causal_splash(query, key, value, segment_ids, cfg, lc, mesh, window=window, seq_minor=seq_minor)

  cs = min(256, t)
  float32_logits = getattr(cfg, "float32_logits", False)
  outs = []
  for q0 in range(0, t, cs):
    q1 = min(q0 + cs, t)
    s0 = max(0, q0 - window) if window < t else 0
    tgt, src = jnp.arange(q0, q1)[:, None], jnp.arange(s0, q1)[None, :]
    valid = (src < tgt)[None]
    if window < q1:
      valid &= (src > tgt - window)
    if segment_ids is not None:
      valid &= segment_ids[:, q0:q1, None] == segment_ids[:, None, s0:q1]
    logits = jnp.einsum("bqnh,bsnh->bnqs", query[:, q0:q1], key[:, s0:q1])  # bqnh,bsnh->bnqs
    if float32_logits:
      logits = logits.astype(jnp.float32)
    logits = jnp.where(valid[:, None], logits, jnp.asarray(NEG_INF, logits.dtype))
    alpha = jax.nn.softmax(logits, axis=-1)
    alpha = jnp.where(jnp.any(valid[:, None], axis=-1, keepdims=True), alpha, jnp.asarray(0, alpha.dtype))
    outs.append(jnp.einsum("bnqs,bsnd->bqnd", alpha.astype(value.dtype), value[:, s0:q1]))  # bnqs,bsnd->bqnd
  return jnp.concatenate(outs, axis=1)


# ==============================================================================
# 2. Matrix-Memory Primitives (`MShift`, `MRead`, `MWrite`, `LoomEmbedWrite`)
# ==============================================================================

class MShift(nn.Module):
  """Token-shift mixing `z * M[t-1] + (1 - z) * M[t]` per slot `k` (Paper Eq. 7)."""

  config: Any
  loom_cfg: LoomConfig

  def setup(self):
    cfg, lc = self.config, self.loom_cfg
    d_k = lc.loom_d_k
    self.W_sg = dense_proj(d_k, nn.initializers.zeros, ("embed", "kv"), cfg, name="W_sg")
    self.b_sg = self.param(
        "b_sg",
        nn.with_logical_partitioning(nn.initializers.constant(logit(lc.loom_m_shift_init)), ("kv",)),
        (d_k,), cfg.weight_dtype,
    )

  def logits(self, x: Array) -> Array:
    return self.W_sg(x).astype(jnp.float32) + self.b_sg.astype(jnp.float32)  # btD,Dk->btk

  def __call__(self, M: Array, x: Array, segment_ids=None, positions=None) -> Array:
    M_prev = jnp.concatenate((M[:, :1], M[:, :-1]), axis=1)  # shift t dim of btkd
    z = jax.nn.sigmoid(self.logits(x)).astype(M.dtype)[..., None]  # btk->btk1
    mixed = z * M_prev + (jnp.asarray(1, M.dtype) - z) * M  # btk1,btkd->btkd
    valid = m_shift_valid_mask(M.shape, segment_ids, positions)
    return jnp.where(valid[..., None, None], mixed, M)


class MRead(nn.Module):
  """Vertical linear-attention read over per-token matrix memory `M` (Paper Eq. 5)."""

  config: Any
  loom_cfg: LoomConfig
  up_lead_axis: Optional[str] = "embed"
  is_value_arm: bool = False
  up_init_override: Any = None

  def setup(self):
    cfg, lc = self.config, self.loom_cfg
    n, d_k, d_r = cfg.num_query_heads, lc.loom_d_k, lc.loom_d_r
    loom_init = nn.initializers.normal(lc.loom_init_std)
    zeros_init = nn.initializers.zeros
    up_init = self.up_init_override if self.up_init_override is not None else loom_init

    read_open = float(lc.loom_read_gate_init)
    stat_bias = math.log(lc.loom_static_read_gate_init / (1.1 - lc.loom_static_read_gate_init))
    qbar_init = nn.initializers.normal(d_k ** -0.5) if self.is_value_arm else zeros_init

    self.W_r2 = dense_proj((n, d_k), up_init, (self.up_lead_axis, "q_heads", "kv"), cfg,
                           in_features=d_r, use_bias=True, name="W_r2")
    self.W_rg = dense_proj(n, zeros_init, ("embed", "q_heads"), cfg, name="W_rg")
    self.b_rg = self.param(
        "b_rg", nn.with_logical_partitioning(nn.initializers.constant(logit(read_open)), ("q_heads",)),
        (n,), cfg.weight_dtype,
    )
    self.q_bar = self.param(
        "q_bar", nn.with_logical_partitioning(qbar_init, ("q_heads", "kv")),
        (n, d_k), cfg.weight_dtype,
    )
    self.W_rg_bar = dense_proj(n, zeros_init, ("embed", "q_heads"), cfg, name="W_rg_bar")
    self.b_rg_bar = self.param(
        "b_rg_bar", nn.with_logical_partitioning(nn.initializers.constant(stat_bias), ("q_heads",)),
        (n,), cfg.weight_dtype,
    )

  def raw_ports(self, x: Array, xr: Array) -> tuple[Array, Array, Array]:
    """Raw `(qm_raw, dyn_gate_logits, stat_gate)` for fused TPU Pallas `read`."""
    cfg, lc = self.config, self.loom_cfg
    stat_init = float(lc.loom_static_read_gate_init)
    qm_raw = self.W_r2(xr)                                                    # btm,m(nk)->btnk
    dyn_logits = self.W_rg(x) + jnp.asarray(self.b_rg, cfg.dtype)             # btD,Dn->btn
    stat_logits = self.W_rg_bar(x) + jnp.asarray(self.b_rg_bar, cfg.dtype)    # btD,Dn->btn
    stat_gate = (1.1 / stat_init) * jax.nn.sigmoid(stat_logits)
    return qm_raw, dyn_logits, stat_gate

  def __call__(self, M: Array, x: Array, xr: Array, scale: Array) -> Array:
    cfg, lc = self.config, self.loom_cfg
    read_eps = float(lc.loom_read_key_eps)
    key_scale = float(lc.loom_read_key_scale)

    qm_raw, dyn_logits, g_bar = self.raw_ports(x, xr)
    g = jax.nn.sigmoid(dyn_logits)[..., None]                                 # btn->btn1
    qm_dyn = key_scale * g * rms_norm_unit(qm_raw, read_eps)                  # btn1,btnk->btnk
    qm_stat = jnp.asarray(self.q_bar, M.dtype) * g_bar[..., None]             # nk,btn1->btnk
    # u = einsum('btnk,btkd->btnd', qm, M)  (dynamic + static ports contracted with M)
    u = jnp.einsum("btnk,btkd->btnd", qm_dyn, M) + jnp.einsum("btnk,btkd->btnd", qm_stat, M)
    return (u.astype(jnp.float32) * scale.astype(jnp.float32)).astype(cfg.dtype)


class MWrite(nn.Module):
  """Vertical outer-product write into per-token matrix memory `M` (Paper Eq. 6)."""

  config: Any
  loom_cfg: LoomConfig
  up_lead_axis: Optional[str] = "embed"

  def setup(self):
    cfg, lc = self.config, self.loom_cfg
    n, d_k, d_r = cfg.num_query_heads, lc.loom_d_k, lc.loom_d_r
    loom_init = nn.initializers.normal(lc.loom_init_std)

    self.W_w2 = dense_proj((n, d_k), loom_init, (self.up_lead_axis, "q_heads", "kv"), cfg,
                           in_features=d_r, use_bias=True, name="W_w2")
    self.W_wg = dense_proj(n, loom_init, ("embed", "q_heads"), cfg, name="W_wg")
    self.b_wg = self.param(
        "b_wg", nn.with_logical_partitioning(nn.initializers.constant(logit(lc.loom_write_gate_init)), ("q_heads",)),
        (n,), cfg.weight_dtype,
    )

  def factors(self, o: Array, x: Array, xw: Array) -> tuple[Array, Array, Array]:
    """Raw `(o, wg_logits, km_raw)` write factors (used directly by fused TPU Pallas `write`)."""
    km_raw = self.W_w2(xw)                                                    # btm,m(nk)->btnk
    wg_logits = self.W_wg(x) + jnp.asarray(self.b_wg, self.config.dtype)      # btD,Dn->btn
    return o, wg_logits, km_raw

  def __call__(self, o: Array, x: Array, xw: Array) -> Array:
    cfg, lc = self.config, self.loom_cfg
    write_eps = float(lc.loom_write_eps if lc.loom_write_eps is not None else cfg.normalization_layer_epsilon)
    o, wg_logits, km_raw = self.factors(o, x, xw)
    km = rms_norm_unit(km_raw, write_eps) * jax.nn.sigmoid(wg_logits)[..., None]  # btnk,btn1->btnk
    dM = jnp.einsum("btnk,btnd->btkd", km, o)                                 # einsum('btnk,btnd->btkd', km, o)
    return dM.astype(cfg.dtype)


class LoomEmbedWrite(nn.Module):
  """Initializes `M` `[b, t, k, d]` from token embeddings `x` via `MWrite` (Paper Eq. 10)."""

  config: Any
  loom_cfg: LoomConfig

  def setup(self):
    cfg, lc = self.config, self.loom_cfg
    n, d, d_r = cfg.num_query_heads, cfg.head_dim, lc.loom_d_r
    loom_init = nn.initializers.normal(lc.loom_init_std)

    self.W_e = dense_proj((n, d), loom_init, ("embed", "q_heads", "kv"), cfg, name="W_e")
    self.W_w1 = dense_proj(d_r, loom_init, ("embed", None), cfg, name="W_w1")
    self.m_write = MWrite(config=cfg, loom_cfg=lc, up_lead_axis="embed", name="m_write")

  def __call__(self, x: Array) -> Array:
    cfg, lc = self.config, self.loom_cfg
    write_eps = float(lc.loom_write_eps if lc.loom_write_eps is not None else cfg.normalization_layer_epsilon)

    o = self.W_e(x)                                                           # btD,D(nd)->btnd
    o = (
        (o.astype(jnp.float32) / float(lc.loom_embed_write_divisor)).astype(cfg.dtype)
        if lc.loom_embed_write_divisor else rms_norm_unit(o, write_eps)
    )
    xw = nn.gelu(self.W_w1(x))                                                # btD,Dm->btm
    return self.m_write(o, x, xw)                                             # btnk,btnd->btkd


# ==============================================================================
# 3. LoomFormer Blocks (`LoomMHA`, `LoomDecoderLayer`, `LoomDecoder`, `LoomFormer`)
# ==============================================================================

class LoomMHA(nn.Module):
  """Loom Multi-Head Attention sub-layer on Llama backbone (Paper Section 3.2 Eqs. 5-8)."""

  config: Any
  loom_cfg: LoomConfig
  mesh: Any
  write_M: bool = True

  def setup(self):
    cfg, lc = self.config, self.loom_cfg
    n, d = cfg.num_query_heads, cfg.head_dim
    d_r, r_dim = lc.loom_d_r, lc.loom_rope_dim
    loom_init = nn.initializers.normal(lc.loom_init_std)
    up_lead = None if replicate_rank_up(lc, n, cfg.num_decoder_layers, self.mesh, d_r) else "embed"

    self.m_shift = MShift(config=cfg, loom_cfg=lc, name="m_shift")
    self.W_r1 = dense_proj(d_r, loom_init, ("embed", None), cfg, name="W_r1")

    parent = self
    def vo_clone_init(_key, shape, dtype):
      src = nn.meta.unbox(parent.variables["params"]["read_o"]["W_r2"]["kernel"])
      return jnp.reshape(src, shape).astype(dtype)

    v_up_init = vo_clone_init if lc.loom_clone_vo_init else None
    self.read_q = MRead(config=cfg, loom_cfg=lc, up_lead_axis=up_lead, name="read_q")
    self.read_k = MRead(config=cfg, loom_cfg=lc, up_lead_axis=up_lead, name="read_k")
    self.read_o = MRead(config=cfg, loom_cfg=lc, up_lead_axis=up_lead, name="read_o")
    self.read_v = MRead(config=cfg, loom_cfg=lc, up_lead_axis=up_lead, is_value_arm=True,
                        up_init_override=v_up_init, name="read_v")
    self.c_qk = self.param("c_qk", nn.with_logical_partitioning(nn.initializers.ones, (None,)), (1,), cfg.weight_dtype)
    self.c_vo = self.param("c_vo", nn.with_logical_partitioning(nn.initializers.ones, (None,)), (1,), cfg.weight_dtype)

    self.W_q = dense_proj((n, r_dim), loom_init, ("embed", "q_heads", "kv"), cfg, name="W_q")
    self.W_k = dense_proj((n, r_dim), loom_init, ("embed", "kv_heads", "kv_head_dim"), cfg, name="W_k")
    self.rotary_q = nnx_wrappers.to_linen(
        embeddings.RotaryEmbedding, min_timescale=cfg.rope_min_timescale, max_timescale=cfg.rope_max_timescale,
        mesh=self.mesh, embedding_dims=r_dim, fprop_dtype=cfg.dtype,
        metadata_fn=variable_to_logically_partitioned, name="rotary_q",
    )
    self.rotary_k = nnx_wrappers.to_linen(
        embeddings.RotaryEmbedding, min_timescale=cfg.rope_min_timescale, max_timescale=cfg.rope_max_timescale,
        mesh=self.mesh, embedding_dims=r_dim, fprop_dtype=cfg.dtype,
        metadata_fn=variable_to_logically_partitioned, name="rotary_k",
    )
    self.W_o = dense_proj(cfg.emb_dim, loom_init, ("heads", "kv", "embed"), cfg,
                          in_features=(n, d), axis=(-2, -1), name="W_o")

    if self.write_M:
      self.W_w1 = dense_proj(d_r, loom_init, ("embed", None), cfg, name="W_w1")
      self.m_write = MWrite(config=cfg, loom_cfg=lc, up_lead_axis=up_lead, name="m_write")

  def attend(self, q: Array, k: Array, v: Array, segment_ids, window: int, seq_minor_default: bool) -> Array:
    return strict_causal_attention(q, k, v, segment_ids, self.config, self.loom_cfg, self.mesh, window, seq_minor_default)

  def __call__(self, x: Array, positions: Array, segment_ids, M_in: Array, window: int):
    cfg, lc = self.config, self.loom_cfg
    x = nn.with_logical_constraint(x, ("activation_batch", "activation_length", "activation_embed"))
    if lc.loom_pallas and jax.default_backend() == "tpu":
      valid = m_shift_valid_mask(x.shape, segment_ids, positions)
      return loom_pallas.mha_forward(
          self, x, positions, segment_ids, M_in, window,
          use_qk_norm=False, logit_scale_dim=cfg.head_dim, valid_shift_mask=valid,
      )

    # 1. Vertical reads from shifted & normalized matrix memory M (Paper Eqs. 5, 7)
    M_tilde = m_norm(self.m_shift(M_in, x, segment_ids, positions), float(cfg.normalization_layer_epsilon))
    xr = nn.gelu(self.W_r1(x))                                                # btD,Dm->btm
    q_tilde = self.read_q(M_tilde, x, xr, self.c_qk[0])                       # btnk,btkd->btnd
    k_tilde = self.read_k(M_tilde, x, xr, self.c_qk[0])                       # btnk,btkd->btnd
    o_m = self.read_o(M_tilde, x, xr, self.c_vo[0])                           # btnk,btkd->btnd (run before read_v for vo_clone_init)
    v = self.read_v(M_tilde, x, xr, self.c_vo[0])                             # btnk,btkd->btnd

    # 2. Vector-stream RoPE coordinates concatenated onto M-read Q/K (Paper Eq. 7)
    qr = self.rotary_q(self.W_q(x), positions)                                # btD,D(nr)->btnr
    kr = self.rotary_k(self.W_k(x), positions)                                # btD,D(nr)->btnr
    q = jnp.concatenate((q_tilde, qr), axis=-1)                               # btnd,btnr->btnh (h = d + r)
    k = jnp.concatenate((k_tilde, kr), axis=-1)                               # btnd,btnr->btnh

    q = nn.with_logical_constraint(q, ("activation_kv_batch", "activation_length", "activation_kv_heads", "activation_kv_head_dim"))
    k = nn.with_logical_constraint(k, ("activation_kv_batch", "activation_length", "activation_kv_heads", "activation_kv_head_dim"))
    v = nn.with_logical_constraint(v, ("activation_kv_batch", "activation_length", "activation_kv_heads", "activation_kv_head_dim"))

    # 3. Strictly causal horizontal attention + local bypass o_m, then write back to M (Paper Eq. 8)
    q_scaled = q / jnp.sqrt(jnp.asarray(cfg.head_dim, cfg.dtype))
    o = o_m + self.attend(q_scaled, k, v, segment_ids, window, seq_minor_default=True)  # btnd
    o = nn.with_logical_constraint(o, ("activation_batch", "activation_length", "activation_heads", "activation_kv"))
    dx = self.W_o(o)                                                          # btnd,(nd)D->btD

    if not self.write_M:
      return dx, M_in, None
    xw = nn.gelu(self.W_w1(x))                                                # btD,Dm->btm
    return dx, M_in, self.m_write.factors(o, x, xw)


class LoomDecoderLayer(nn.Module):
  """Single LoomFormer block `(x, M) <- (x, M) + LoomMHA + FFN` (Paper Eqs. 9, 11)."""

  config: Any
  mesh: Any
  model_mode: str = MODEL_MODE_TRAIN
  quant: Any = None
  loom_cfg: LoomConfig = LoomConfig()
  write_M: bool = True
  is_last_layer: bool = False

  def setup(self):
    cfg, lc = self.config, self.loom_cfg
    loom_init = nn.initializers.normal(lc.loom_init_std)
    write_here = self.write_M and not self.is_last_layer

    self.ln_1 = rms_norm_layer(cfg, "ln_1")
    self.loom_mha = LoomMHA(config=cfg, loom_cfg=lc, mesh=self.mesh, write_M=write_here, name="loom_mha")
    self.ln_2 = rms_norm_layer(cfg, "ln_2")

    ffn_x_dim = lc.loom_last_ffn_x_dim if self.is_last_layer else cfg.mlp_dim
    x_acts = (
        (functools.partial(situ_gate, beta=lc.loom_situ_beta1), functools.partial(situ_up, beta=lc.loom_situ_beta2))
        if (lc.loom_situ_beta1 > 0.0 and lc.loom_situ_beta2 > 0.0)
        else ("silu", "linear")
    )
    self.ffn_x = linears.mlp_block(
        config=cfg, mesh=self.mesh, in_features=cfg.emb_dim, intermediate_dim=ffn_x_dim,
        activations=x_acts, kernel_init=lambda k, s, dt, *_: loom_init(k, s, dt),
        intermediate_dropout_rate=cfg.dropout_rate, dtype=cfg.dtype, weight_dtype=cfg.weight_dtype,
        use_bias=False, quant=self.quant, model_mode=self.model_mode, name="ffn_x",
    )

    if write_here:
      n, d_r = cfg.num_query_heads, lc.loom_d_r
      up_lead = None if replicate_rank_up(lc, n, cfg.num_decoder_layers, self.mesh, d_r) else "embed"
      self.ffn_m = linears.mlp_block(
          config=cfg, mesh=self.mesh, in_features=cfg.emb_dim, intermediate_dim=lc.loom_ffn_m_dim,
          activations=("silu", "linear"), kernel_init=lambda k, s, dt, *_: loom_init(k, s, dt),
          intermediate_dropout_rate=cfg.dropout_rate, dtype=cfg.dtype, weight_dtype=cfg.weight_dtype,
          use_bias=False, quant=self.quant, model_mode=self.model_mode, name="ffn_m",
      )
      self.ffn_W_w1 = dense_proj(d_r, loom_init, ("embed", None), cfg, name="ffn_W_w1")
      self.ffn_m_write = MWrite(config=cfg, loom_cfg=lc, up_lead_axis=up_lead, name="ffn_m_write")

    self.dropout = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,))

  def __call__(self, carry, decoder_segment_ids, decoder_positions, deterministic: bool = False,
               model_mode: str = MODEL_MODE_TRAIN, **_unused):
    del model_mode
    cfg, lc = self.config, self.loom_cfg
    write_here = self.write_M and not self.is_last_layer
    x, M = carry

    x = nn.with_logical_constraint(x, ("activation_batch", "activation_norm_length", "activation_embed"))
    x = checkpoint_name(x, "decoder_layer_input")

    # 1. LoomMHA sub-layer
    h1 = nn.with_logical_constraint(self.ln_1(x), ("activation_batch", "activation_norm_length", "activation_embed"))
    dx, M, attn_write = self.loom_mha(h1, decoder_positions, decoder_segment_ids, M, cfg.max_target_length)
    x_mid = x + nn.with_logical_constraint(dx, ("activation_batch", "activation_norm_length", "activation_embed"))

    # 2. Dual-branch FFN (`ffn_x` -> vector stream `x`; `ffn_m` -> matrix stream `M`, Paper Eq. 9)
    h2 = nn.with_logical_constraint(self.ln_2(x_mid), ("activation_batch", "activation_norm_length", "activation_embed"))
    df_x = self.ffn_x(h2, deterministic=deterministic)                        # btD->btD
    df_x = nn.with_logical_constraint(df_x, ("activation_batch", "activation_norm_length", "activation_embed"))

    if write_here:
      n, d = cfg.num_query_heads, cfg.head_dim
      write_eps = float(lc.loom_write_eps if lc.loom_write_eps is not None else cfg.normalization_layer_epsilon)
      o_ffn = self.ffn_m(h2, deterministic=deterministic).reshape(h2.shape[:2] + (n, d))  # bt(nd)->btnd
      xw_f = nn.gelu(self.ffn_W_w1(h2))                                       # btD,Dm->btm

      if lc.loom_pallas and jax.default_backend() == "tpu":
        M = loom_pallas.write(
            M, [attn_write, self.ffn_m_write.factors(o_ffn, h2, xw_f)],
            epsilon=write_eps, content_divisors=(None, 1.0),
            forward_tile=128, reverse_tile=128, vmem_mib=lc.loom_pallas_vmem_mib,
        )
      else:
        o_attn, wg_a_logits, km_a_raw = attn_write
        km_a = rms_norm_unit(km_a_raw, write_eps) * jax.nn.sigmoid(wg_a_logits)[..., None]  # btnk,btn1->btnk
        dM_attn = jnp.einsum("btnk,btnd->btkd", km_a, rms_norm_unit(o_attn, write_eps))     # btnk,btnd->btkd
        dM_ffn = self.ffn_m_write(o_ffn, h2, xw_f)                                          # btnk,btnd->btkd
        M = (M + dM_attn + dM_ffn).astype(cfg.dtype)

    x_out = self.dropout(x_mid + df_x, deterministic=deterministic)
    x_out = nn.with_logical_constraint(x_out, ("activation_batch", "activation_norm_length", "activation_embed"))
    return (x_out, M), None


class LoomDecoder(decoders.Decoder):
  """Dual-stream `(x, M)` decoder stack (`setup` + `__call__`) on official `maxtext.layers.decoders.Decoder`."""

  loom_cfg: LoomConfig = LoomConfig()

  def get_decoder_layers(self):
    return [LoomDecoderLayer]

  def setup(self):
    cfg, lc = self.config, self.loom_cfg
    loom_init = nn.initializers.normal(lc.loom_init_std)

    self.decoder_layer = self.get_decoder_layers()
    self.emb_dropout = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,))
    self.embed_write = LoomEmbedWrite(config=cfg, loom_cfg=lc, name="embed_write")

    RematLayer = self.set_remat_policy(self.decoder_layer, self.get_remat_policy())[0]
    if cfg.scan_layers and cfg.num_decoder_layers > 1:
      self.layers = self.scan_decoder_layers(
          cfg, RematLayer, cfg.num_decoder_layers - 1, "layers", self.mesh,
          in_axes_tuple=(nn.broadcast,) * 4, model_mode=self.model_mode,
          loom_cfg=lc, write_M=True, is_last_layer=False,
      )
      self.last_layer = RematLayer(
          config=cfg, mesh=self.mesh, name="last_layer", quant=self.quant, model_mode=self.model_mode,
          loom_cfg=lc, write_M=False, is_last_layer=True,
      )
    else:
      self.unscanned_layers = [
          RematLayer(
              config=cfg, mesh=self.mesh, name=f"layers_{lyr}", quant=self.quant, model_mode=self.model_mode,
              loom_cfg=lc, write_M=(lyr < cfg.num_decoder_layers - 1),
              is_last_layer=(lyr == cfg.num_decoder_layers - 1),
          )
          for lyr in range(cfg.num_decoder_layers)
      ]

    self.decoder_norm = rms_norm_layer(cfg, "decoder_norm")
    self.out_dropout = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,))
    self.logits_dense = dense_proj(cfg.vocab_size, loom_init, ("embed", "vocab"), cfg, name="logits_dense")

  def apply_output_head(self, shared_embedding, y, deterministic, model_mode):
    del shared_embedding, model_mode
    cfg, lc = self.config, self.loom_cfg
    y = self.decoder_norm(y)
    if y.ndim >= 3:
      y = self.out_dropout(y, deterministic=deterministic)
    if lc.loom_logits_bf16_dot_fp32_out and y.dtype == jnp.bfloat16:
      if self.is_initializing():
        _ = self.logits_dense(y[..., :1, :])
      w = nn.meta.unbox(self.variables["params"]["logits_dense"]["kernel"]).astype(cfg.dtype)
      w = nn.with_logical_constraint(w, (None, "vocab"))
      logits = bf16_dot_fp32_output(y, w)                                     # btD,DV->btV (FP32 accum)
    else:
      logits = self.logits_dense(y)                                           # btD,DV->btV
      if getattr(cfg, "cast_logits_to_fp32", True):
        logits = logits.astype(jnp.float32)
    spec = (
        ("activation_embed_and_logits_batch", "activation_length_no_exp", "activation_vocab")
        if logits.ndim == 3
        else ("activation_embed_and_logits_batch_sequence", "activation_vocab")
    )
    return nn.with_logical_constraint(logits, spec)

  def __call__(self, shared_embedding, decoder_input_tokens, decoder_positions,
               decoder_segment_ids=None, deterministic=False, model_mode=MODEL_MODE_TRAIN, **_unused):
    cfg = self.config
    x = shared_embedding(decoder_input_tokens.astype("int32"), model_mode=model_mode)
    x = self.emb_dropout(x, deterministic=deterministic).astype(cfg.dtype)
    M = self.embed_write(x)                                                   # btD->btkd
    carry = (x, M)

    bcast = (decoder_segment_ids, decoder_positions, deterministic, model_mode)
    if cfg.scan_layers and cfg.num_decoder_layers > 1:
      carry, _ = self.layers(carry, *bcast)
      carry, _ = self.last_layer(carry, *bcast)
    else:
      for layer in self.unscanned_layers:
        carry, _ = layer(carry, *bcast)

    hidden_state = carry[0]
    if self.is_initializing() and cfg.num_vocab_tiling > 1:
      _ = self.apply_output_head(shared_embedding, hidden_state[:, :1], deterministic, model_mode)
    if cfg.num_vocab_tiling > 1 and self.model_mode == MODEL_MODE_TRAIN:
      self.sow("intermediates", "hidden_states", hidden_state)
      return None, hidden_state, None
    logits = self.apply_output_head(shared_embedding, hidden_state, deterministic, model_mode)
    return logits, hidden_state, None


class LoomFormer(models.TransformerLinenPure):
  """Top-level LoomFormer language model plugging `LoomDecoder` into MaxText's `TransformerLinenPure`."""

  quant: Any = None
  loom_cfg: LoomConfig = LoomConfig()

  def setup(self):
    cfg, lc = self.config, self.loom_cfg
    lc.validate(cfg.num_query_heads, cfg.head_dim, cfg.emb_dim)
    loom_init = nn.initializers.normal(lc.loom_init_std)
    self.shared_embedding = embeddings.embed_as_linen(
        num_embeddings=cfg.vocab_size,
        num_features=cfg.emb_dim,
        dtype=cfg.dtype,
        attend_dtype=jnp.float32 if cfg.logits_dot_in_fp32 else cfg.dtype,
        embedding_init=loom_init,
        name="token_embedder",
        config=cfg,
        mesh=self.mesh,
    )
    self.vision_encoder = None
    self.audio_encoder = None
    self.decoder = LoomDecoder(
        config=cfg, mesh=self.mesh, quant=self.quant, model_mode=self.model_mode, loom_cfg=lc, name="decoder"
    )
