"""Minimal-delta JAX/Flax implementation of LoomFormer & LoomPythia on official `maxtext-v0.2.0`.

Reuses standard `flax.linen` (`nn.DenseGeneral`, `nn.RMSNorm`, `nn.LayerNorm`) and `maxtext`
(`mlp_block`, `RotaryEmbedding`, `embed_as_linen`, `Decoder`, `TransformerLinenPure`) for all
standard Transformer components, implementing ONLY LoomFormer's own incremental operations:
  - `bf16_dot_fp32_output`: lossless MXU FP32 logit accumulator with BF16 backward (`@jax.custom_vjp`)
  - `situ_gate` / `situ_up`: bounded SiTU-GLU activation callables passed into `maxtext.layers.linears.mlp_block`
  - `LoomEmbedWrite`, `LoomMHA`, `LoomDecoderLayer`: matrix-memory stream (`M` in `[B, T, d_k, d]`)
  - `LoomDecoder(decoders.Decoder)` & `LoomTransformer(models.TransformerLinenPure)`: dual-stream `(x, M)` scan hook
"""
import functools
import math
from typing import Any, Optional, Sequence

from flax import linen as nn
import jax
from jax import lax
from jax.ad_checkpoint import checkpoint_name
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask
from jax.experimental.shard_map import shard_map
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
# 1. LoomFormer-Specific Numerics & Activation Callables
# ==============================================================================

@jax.custom_vjp
def bf16_dot_fp32_output(hidden: Array, kernel: Array) -> Array:
  """Keep native MXU FP32 accumulator output while keeping BF16 operands in forward & backward."""
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


def situ_gate(x: Array, beta: float) -> Array:
  xf = x.astype(jnp.float32)
  return (beta * jnp.tanh(xf / beta) * jax.nn.sigmoid(xf)).astype(x.dtype)


def situ_up(x: Array, beta: float) -> Array:
  xf = x.astype(jnp.float32)
  return (beta * jnp.tanh(xf / beta)).astype(x.dtype)


def rms_norm_unit(x: Array, eps: float) -> Array:
  """Parameter-free RMSNorm over the last axis with FP32 statistics (Paper Eqs. 5, 6, 8)."""
  f = x.astype(jnp.float32)
  return (f * lax.rsqrt(jnp.mean(jnp.square(f), axis=-1, keepdims=True) + eps)).astype(x.dtype)


def m_norm(M: Array, eps: float) -> Array:
  """Whole-matrix RMSNorm over `(-2, -1) = (d_k, d)` with FP32 statistics (Paper Eq. 7)."""
  f = M.astype(jnp.float32)
  return (f * lax.rsqrt(jnp.mean(jnp.square(f), axis=(-2, -1), keepdims=True) + eps)).astype(M.dtype)


def m_shift_valid_mask(shape, segment_ids=None, positions=None) -> Array:
  """Valid previous-token mask for `MShift` (excludes `t = 0` and packed segment/position breaks)."""
  valid = jnp.broadcast_to(jnp.arange(shape[1])[None, :] > 0, shape[:2])
  if segment_ids is not None:
    prev_seg = jnp.concatenate((segment_ids[:, :1], segment_ids[:, :-1]), axis=1)
    valid &= (segment_ids == prev_seg) & (segment_ids != 0)
  if positions is not None:
    prev_pos = jnp.concatenate((positions[:, :1], positions[:, :-1]), axis=1)
    valid &= (positions == prev_pos + 1)
  return valid


def m_shift(M: Array, z: Array, segment_ids=None, positions=None) -> Array:
  """Mix `M[t]` with `M[t-1]` via per-slot gate `z[t]` in `[B, T, d_k]` (Paper Eq. 7)."""
  M_prev = jnp.concatenate((M[:, :1], M[:, :-1]), axis=1)
  valid = m_shift_valid_mask(M.shape, segment_ids, positions)
  z_cast = z.astype(M.dtype)[..., None]
  mixed = z_cast * M_prev + (jnp.asarray(1, M.dtype) - z_cast) * M
  return jnp.where(valid[..., None, None], mixed, M)


def _logit(p: float) -> float:
  return math.log(p / (1.0 - p))


def _dense(features, init, axes: tuple[Optional[str], ...], cfg: Any, *,
           in_features=None, axis=-1, use_bias=False, name: str):
  """Thin helper configuring official `maxtext.layers.linears.dense_general`."""
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


def _decoder_norm(cfg: Any, is_pythia: bool, name: str):
  """Standard `flax.linen.LayerNorm` (Pythia) or `flax.linen.RMSNorm` (Llama) with FP32 statistics."""
  if is_pythia:
    return nn.LayerNorm(
        epsilon=cfg.normalization_layer_epsilon,
        dtype=cfg.dtype,
        param_dtype=cfg.weight_dtype,
        scale_init=nn.with_logical_partitioning(nn.initializers.ones, ("norm",)),
        bias_init=nn.with_logical_partitioning(nn.initializers.zeros, ("norm",)),
        name=name,
    )
  return nn.RMSNorm(
      epsilon=cfg.normalization_layer_epsilon,
      dtype=cfg.dtype,
      param_dtype=cfg.weight_dtype,
      scale_init=nn.with_logical_partitioning(nn.initializers.ones, ("norm",)),
      name=name,
  )


def _model_inits(cfg: Any, lc: LoomConfig):
  loom_init = nn.initializers.normal(lc.loom_init_std)
  if lc.loom_backbone == "pythia":
    base_std = math.sqrt(2.0 / (5.0 * cfg.emb_dim))
    out_std = 2.0 / (cfg.num_decoder_layers * math.sqrt(cfg.emb_dim))
    return loom_init, nn.initializers.normal(base_std), nn.initializers.normal(out_std)
  return loom_init, loom_init, loom_init


def _replicate_rank_up(lc: LoomConfig, num_heads: int, num_layers: int, mesh: Any, rank: int) -> bool:
  mode = str(lc.loom_replicate_rank_up).lower()
  if mode != "auto":
    return mode == "true"
  fsdp = mesh.shape.get("fsdp", 1) * mesh.shape.get("fsdp_transpose", 1)
  if fsdp <= 1 or rank >= 64 * fsdp:
    return False
  return (num_layers * 6 * rank * num_heads * lc.loom_d_k * 16 / (2 ** 30)) <= 2.0


# ==============================================================================
# 2. Strictly Causal (`s < t`) Horizontal Attention (Splash + Reference Einsum)
# ==============================================================================

def _strict_causal_splash(query, key, value, segment_ids, cfg, lc: LoomConfig, mesh, window=None, seq_minor=False):
  q, k, v = (jnp.transpose(x, (0, 2, 1, 3)) for x in (query, key, value))
  t = q.shape[2]
  axis_names = nn.logical_to_mesh_axes(("activation_batch", "activation_heads", "activation_length", "activation_kv"))
  seg_names = nn.logical_to_mesh_axes(("activation_batch", "activation_length_no_exp"))
  segments = (
      None if segment_ids is None or lc.loom_splash_ignore_segments
      else splash_attention_kernel.SegmentIds(segment_ids, segment_ids)
  )

  def body(q, k, v, segments=None):
    fused = cfg.sa_use_fused_bwd_kernel
    block_sizes = splash_attention_kernel.BlockSizes(
        block_q=min(cfg.sa_block_q, t),
        block_kv=min(cfg.sa_block_kv, t),
        block_kv_compute=min(cfg.sa_block_kv_compute, t),
        block_q_dkv=min(cfg.sa_block_q_dkv, t),
        block_kv_dkv=min(cfg.sa_block_kv_dkv, t),
        block_kv_dkv_compute=min(cfg.sa_block_kv_dkv_compute, t),
        block_q_dq=None if fused else min(cfg.sa_block_q_dq, t),
        block_kv_dq=None if fused else min(cfg.sa_block_kv_dq, t),
        use_fused_bwd_kernel=fused,
        q_layout=splash_attention_kernel.QKVLayout["SEQ_MINOR" if seq_minor else cfg.sa_q_layout],
        k_layout=splash_attention_kernel.QKVLayout["SEQ_MINOR" if seq_minor else cfg.sa_k_layout],
        v_layout=splash_attention_kernel.QKVLayout["SEQ_MINOR" if seq_minor else cfg.sa_v_layout],
    )
    mask = splash_attention_mask.CausalMask(shape=(t, t), offset=-1)
    if window is not None and window < t:
      mask &= splash_attention_mask.LocalMask(shape=(t, t), window_size=(window - 1, 0), offset=0)
    mask = splash_attention_mask.MultiHeadMask(masks=(mask,) * q.shape[1])
    res_name = lc.loom_splash_residual_name or None
    soft_cap = cfg.attn_logits_soft_cap if getattr(cfg, "attn_logits_soft_cap", 0.0) else None
    kernel = splash_attention_kernel.make_splash_mha(
        mask=mask,
        head_shards=1,
        q_seq_shards=1,
        block_sizes=block_sizes,
        attn_logits_soft_cap=soft_cap,
        **({"residual_checkpoint_name": res_name} if res_name else {}),
    )
    kernel = loom_pallas.make_strict_causal_splash(kernel)
    return jax.vmap(kernel)(q, k, v, segment_ids=segments)

  if segments is None:
    out = shard_map(body, mesh=mesh, in_specs=(axis_names,) * 3, out_specs=axis_names, check_rep=False)(q, k, v)
  else:
    out = shard_map(body, mesh=mesh, in_specs=(axis_names, axis_names, axis_names, seg_names),
                    out_specs=axis_names, check_rep=False)(q, k, v, segments)
  return jnp.transpose(out, (0, 2, 1, 3))


def _strict_causal_einsum_chunks(query, key, value, segment_ids, window_size, chunk_size=256, float32_logits=False):
  t = query.shape[1]
  cs = min(int(chunk_size or t), t)
  outs = []
  for q0 in range(0, t, cs):
    q1 = min(q0 + cs, t)
    s0 = max(0, q0 - window_size) if window_size < t else 0
    tgt, src = jnp.arange(q0, q1)[:, None], jnp.arange(s0, q1)[None, :]
    valid = (src < tgt)[None]
    if window_size < q1:
      valid &= (src > tgt - window_size)
    if segment_ids is not None:
      valid &= segment_ids[:, q0:q1, None] == segment_ids[:, None, s0:q1]
    logits = jnp.einsum("bqnd,bsnd->bnqs", query[:, q0:q1], key[:, s0:q1])
    if float32_logits:
      logits = logits.astype(jnp.float32)
    logits = jnp.where(valid[:, None], logits, jnp.asarray(NEG_INF, logits.dtype))
    alpha = jax.nn.softmax(logits, axis=-1)
    alpha = jnp.where(jnp.any(valid[:, None], axis=-1, keepdims=True), alpha, jnp.asarray(0, alpha.dtype))
    outs.append(jnp.einsum("bnqs,bsnd->bqnd", alpha.astype(value.dtype), value[:, s0:q1]))
  return jnp.concatenate(outs, axis=1)


# ==============================================================================
# 3. LoomFormer Modules (`LoomEmbedWrite`, `LoomMHA`, `LoomDecoderLayer`)
# ==============================================================================

class LoomEmbedWrite(nn.Module):
  """Initializes `M_0` `[B, T, d_k, d]` from token embeddings `x_0` via `MWrite` (Paper Eq. 10)."""

  config: Any
  loom_cfg: LoomConfig

  @nn.compact
  def __call__(self, x_0: Array) -> Array:
    cfg, lc = self.config, self.loom_cfg
    n, d, d_k, d_r = cfg.num_query_heads, cfg.head_dim, lc.loom_d_k, lc.loom_d_r
    loom_init, _, _ = _model_inits(cfg, lc)
    write_eps = float(lc.loom_write_eps if lc.loom_write_eps is not None else cfg.normalization_layer_epsilon)

    o = _dense((n, d), loom_init, ("embed", "q_heads", "kv"), cfg, name="W_e")(x_0)
    xw_0 = nn.gelu(_dense(d_r, loom_init, ("embed", None), cfg, name="W_w1")(x_0))
    km_raw = _dense((n, d_k), loom_init, ("embed", "q_heads", "kv"), cfg, in_features=d_r, use_bias=True, name="W_w2")(xw_0)
    wg_logits = _dense(n, loom_init, ("embed", "q_heads"), cfg, name="W_wg")(x_0)
    b_wg = self.param("b_wg", nn.with_logical_partitioning(nn.initializers.constant(_logit(lc.loom_write_gate_init)), ("q_heads",)),
                      (n,), cfg.weight_dtype)
    gate = jax.nn.sigmoid(wg_logits + jnp.asarray(b_wg, cfg.dtype))

    o_scaled = (
        (o.astype(jnp.float32) / float(lc.loom_embed_write_divisor)).astype(cfg.dtype)
        if lc.loom_embed_write_divisor else rms_norm_unit(o, write_eps)
    )
    km = rms_norm_unit(km_raw, write_eps) * gate[..., None]
    return jnp.einsum("btnk,btnd->btkd", km, o_scaled).astype(cfg.dtype)


class LoomMHA(nn.Module):
  """Loom Multi-Head Attention sub-layer (Paper Section 3.2 Eqs. 5-8)."""

  config: Any
  loom_cfg: LoomConfig
  mesh: Any
  write_M: bool = True

  def setup(self):
    cfg, lc = self.config, self.loom_cfg
    n, d = cfg.num_query_heads, cfg.head_dim
    d_k, d_r, r_dim = lc.loom_d_k, lc.loom_d_r, lc.loom_rope_dim
    is_pythia = lc.loom_backbone == "pythia"
    loom_init, base_init, out_init = _model_inits(cfg, lc)
    zeros_init = nn.initializers.zeros
    up_lead = None if _replicate_rank_up(lc, n, cfg.num_decoder_layers, self.mesh, d_r) else "embed"

    self.W_q = _dense((n, r_dim), base_init, ("embed", "q_heads", "kv"), cfg, use_bias=is_pythia, name="W_q")
    self.W_k = _dense((n, r_dim), base_init, ("embed", "kv_heads", "kv_head_dim"), cfg, use_bias=is_pythia, name="W_k")
    self.W_o = _dense(cfg.emb_dim, out_init, ("heads", "kv", "embed"), cfg, in_features=(n, d), axis=(-2, -1), use_bias=is_pythia, name="W_o")

    # Reuse official MaxText `RotaryEmbedding` via its NNX->Linen bridge
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
    if getattr(cfg, "use_qk_norm", False):
      self.q_norm_scale = self.param("q_norm_scale", nn.with_logical_partitioning(nn.initializers.ones, ("embed",)),
                                     (d + r_dim,), cfg.weight_dtype)
      self.k_norm_scale = self.param("k_norm_scale", nn.with_logical_partitioning(nn.initializers.ones, ("embed",)),
                                     (d + r_dim,), cfg.weight_dtype)

    self.W_sg = _dense(d_k, zeros_init, ("embed", "kv"), cfg, name="W_sg")
    self.b_sg = self.param("b_sg", nn.with_logical_partitioning(nn.initializers.constant(_logit(lc.loom_m_shift_init)), ("kv",)),
                           (d_k,), cfg.weight_dtype)
    self.W_r1 = _dense(d_r, loom_init, ("embed", None), cfg, name="W_r1")

    read_open = float(lc.loom_read_gate_init)
    stat_bias = math.log(lc.loom_static_read_gate_init / (1.1 - lc.loom_static_read_gate_init))

    for arm in ("q", "k", "o", "v"):
      if arm == "v" and lc.loom_clone_vo_init:
        def vo_clone_init(_key, shape, dtype):
          src = nn.meta.unbox(self.variables["params"]["W_r2_o"]["kernel"])
          return jnp.reshape(src, shape).astype(dtype)
        up_init = vo_clone_init
      else:
        up_init = loom_init

      setattr(self, f"W_r2_{arm}", _dense((n, d_k), up_init, (up_lead, "q_heads", "kv"), cfg, in_features=d_r, use_bias=True, name=f"W_r2_{arm}"))
      setattr(self, f"W_rg_{arm}", _dense(n, zeros_init, ("embed", "q_heads"), cfg, name=f"W_rg_{arm}"))
      setattr(self, f"b_rg_{arm}", self.param(
          f"b_rg_{arm}", nn.with_logical_partitioning(nn.initializers.constant(_logit(read_open)), ("q_heads",)),
          (n,), cfg.weight_dtype,
      ))
      qbar_init = nn.initializers.normal(d_k ** -0.5) if arm == "v" else nn.initializers.zeros
      setattr(self, f"q_bar_{arm}", self.param(
          f"q_bar_{arm}", nn.with_logical_partitioning(qbar_init, ("q_heads", "kv")), (n, d_k), cfg.weight_dtype,
      ))
      setattr(self, f"W_rg_bar_{arm}", _dense(n, zeros_init, ("embed", "q_heads"), cfg, name=f"W_rg_bar_{arm}"))
      setattr(self, f"b_rg_bar_{arm}", self.param(
          f"b_rg_bar_{arm}", nn.with_logical_partitioning(nn.initializers.constant(stat_bias), ("q_heads",)),
          (n,), cfg.weight_dtype,
      ))

    self.c_qk = self.param("c_qk", nn.with_logical_partitioning(nn.initializers.ones, (None,)), (1,), cfg.weight_dtype)
    self.c_vo = self.param("c_vo", nn.with_logical_partitioning(nn.initializers.ones, (None,)), (1,), cfg.weight_dtype)

    if self.write_M:
      self.W_w1 = _dense(d_r, loom_init, ("embed", None), cfg, name="W_w1")
      self.W_w2 = _dense((n, d_k), loom_init, (up_lead, "q_heads", "kv"), cfg, in_features=d_r, use_bias=True, name="W_w2")
      self.W_wg = _dense(n, loom_init, ("embed", "q_heads"), cfg, name="W_wg")
      self.b_wg = self.param("b_wg", nn.with_logical_partitioning(nn.initializers.constant(_logit(lc.loom_write_gate_init)), ("q_heads",)),
                             (n,), cfg.weight_dtype)

  def _fused_projections(self, x: Array, names: Sequence[str]):
    cfg = self.config
    params = self.variables["params"]
    kernels, shapes, biases = [], [], []
    for name in names:
      p = params[name]
      k = jnp.asarray(nn.meta.unbox(p["kernel"]), cfg.dtype)
      shapes.append(k.shape[1:])
      kernels.append(k.reshape(k.shape[0], -1))
      b = p.get("bias") if hasattr(p, "get") else None
      biases.append(None if b is None else jnp.asarray(nn.meta.unbox(b), cfg.dtype))
    y = lax.dot_general(
        jnp.asarray(x, cfg.dtype),
        jnp.concatenate(kernels, axis=1),
        (((x.ndim - 1,), (0,)), ((), ())),
        precision=lax.Precision(cfg.matmul_precision),
    )
    widths = [k.shape[1] for k in kernels]
    cuts = [sum(widths[: i + 1]) for i in range(len(widths) - 1)]
    parts = jnp.split(y, cuts, axis=-1) if cuts else [y]
    return {
        name: (part.reshape(x.shape[:-1] + shape) if bias is None else part.reshape(x.shape[:-1] + shape) + bias)
        for name, shape, part, bias in zip(names, shapes, parts, biases)
    }

  def _logit_scale_dim(self, full_qk_width: int) -> int:
    return full_qk_width if self.loom_cfg.loom_qk_scale_dim == "qk_dim" else self.config.head_dim

  def _attend(self, q: Array, k: Array, v: Array, segment_ids, window: int, seq_minor_default: bool) -> Array:
    cfg, lc = self.config, self.loom_cfg
    t = q.shape[1]
    if lc.loom_splash and t % 128 == 0 and jax.default_backend() == "tpu":
      seq_minor = seq_minor_default if lc.loom_splash_seq_minor is None else bool(lc.loom_splash_seq_minor)
      return _strict_causal_splash(q, k, v, segment_ids, cfg, lc, self.mesh, window=window, seq_minor=seq_minor)
    return _strict_causal_einsum_chunks(q, k, v, segment_ids, window, 256, getattr(cfg, "float32_logits", False))

  def _pallas_forward(self, x: Array, positions: Array, segment_ids, M_in: Array, window: int):
    cfg, lc = self.config, self.loom_cfg
    b, t = x.shape[:2]
    n, d, d_k, r_dim = cfg.num_query_heads, cfg.head_dim, lc.loom_d_k, lc.loom_rope_dim
    full_qk = d + r_dim
    read_eps, key_scale, stat_init = float(lc.loom_read_key_eps), float(lc.loom_read_key_scale), float(lc.loom_static_read_gate_init)

    fused = not self.is_initializing()
    if fused:
      proj_names = (
          "W_q", "W_k", "W_r1",
          "W_rg_q", "W_rg_k", "W_rg_v", "W_rg_o",
          "W_rg_bar_q", "W_rg_bar_k", "W_rg_bar_v", "W_rg_bar_o",
          "W_sg",
      ) + (("W_w1", "W_wg") if self.write_M else ())
      raw = self._fused_projections(x, proj_names)
      qr, kr = raw["W_q"], raw["W_k"]
      xr = nn.gelu(raw["W_r1"])
    else:
      qr, kr = self.W_q(x), self.W_k(x)
      xr = nn.gelu(self.W_r1(x))

    if not getattr(cfg, "use_qk_norm", False):
      qr = self.rotary_q(qr, positions)
      kr = self.rotary_k(kr, positions)
      qk_norm_arg = None
      q_scale = 1.0 / math.sqrt(self._logit_scale_dim(full_qk))
    else:
      q_mult = jnp.asarray(self.q_norm_scale, jnp.float32)
      k_mult = jnp.asarray(self.k_norm_scale, jnp.float32)
      ss_q = jnp.sum(jnp.square(qr.astype(jnp.float32)), axis=-1)
      ss_k = jnp.sum(jnp.square(kr.astype(jnp.float32)), axis=-1)
      qr = self.rotary_q(qr * q_mult[d:].astype(qr.dtype), positions)
      kr = self.rotary_k(kr * k_mult[d:].astype(kr.dtype), positions)
      ones = jnp.ones((r_dim,), jnp.float32)
      m_q = jnp.concatenate((q_mult[:d], ones)) / math.sqrt(self._logit_scale_dim(full_qk))
      m_k = jnp.concatenate((k_mult[:d], ones))
      qk_norm_arg = (ss_q, ss_k, m_q, m_k, float(cfg.normalization_layer_epsilon))
      q_scale = 1.0

    def dyn_logits(arm):
      return (raw[f"W_rg_{arm}"] if fused else getattr(self, f"W_rg_{arm}")(x)) + jnp.asarray(getattr(self, f"b_rg_{arm}"), cfg.dtype)

    def stat_gate(arm):
      g = (raw[f"W_rg_bar_{arm}"] if fused else getattr(self, f"W_rg_bar_{arm}")(x)) + jnp.asarray(getattr(self, f"b_rg_bar_{arm}"), cfg.dtype)
      return (1.1 * jax.nn.sigmoid(g)) / stat_init

    if fused:
      params = self.variables["params"]
      kernels = [jnp.asarray(nn.meta.unbox(params[f"W_r2_{arm}"]["kernel"]), cfg.dtype) for arm in ("q", "k", "v", "o")]
      biases = [jnp.asarray(nn.meta.unbox(params[f"W_r2_{arm}"]["bias"]), cfg.dtype) for arm in ("q", "k", "v", "o")]
      ones_tok = jnp.ones((b, t, 1), xr.dtype)
      weight_up = jnp.concatenate((jnp.concatenate(kernels, axis=1), jnp.concatenate(biases, axis=0)[None]), axis=0)
      packed_qkvo = jnp.einsum(
          "btr,rkc->bckt", jnp.concatenate((xr, ones_tok), axis=-1), weight_up,
          precision=lax.Precision(cfg.matmul_precision),
      )
      q_key = jnp.zeros((b, t, n, d_k), cfg.dtype)
      k_key = q_key
      vo_key = jnp.zeros((b, t, 2 * n, d_k), cfg.dtype)
    else:
      q_key = self.W_r2_q(xr)
      k_key = self.W_r2_k(xr)
      o_key = self.W_r2_o(xr)
      v_key = self.W_r2_v(xr)
      vo_key = jnp.concatenate((v_key, o_key), axis=-2)
      packed_qkvo = None

    c_qk, c_vo = self.c_qk[0], self.c_vo[0]
    sw = loom_pallas.static_weight(
        (self.q_bar_q * c_qk.astype(self.q_bar_q.dtype)).T,
        (self.q_bar_k * c_qk.astype(self.q_bar_k.dtype)).T,
        (self.q_bar_v * c_vo.astype(self.q_bar_v.dtype)).T,
        (self.q_bar_o * c_vo.astype(self.q_bar_o.dtype)).T,
        jnp.eye(d_k, dtype=self.q_bar_q.dtype),
    )

    shift_logits = (raw["W_sg"] if fused else self.W_sg(x)).astype(jnp.float32) + self.b_sg.astype(jnp.float32)
    valid = m_shift_valid_mask(x.shape, segment_ids, positions)
    shift = jnp.where(valid[..., None], jax.nn.sigmoid(shift_logits).astype(M_in.dtype), jnp.zeros((b, t, d_k), M_in.dtype))

    ones_n = jnp.ones((n,), jnp.float32)
    outs = loom_pallas.read(
        M_in, sw,
        q_key, dyn_logits("q"),
        k_key, dyn_logits("k"),
        vo_key, dyn_logits("o"), dyn_logits("v"),
        qr, kr,
        qk_cols=d,
        read_epsilon=read_eps,
        key_scale=key_scale,
        forward_tile=128,
        reverse_tile=128,
        vmem_mib=lc.loom_pallas_vmem_mib,
        q_scale=q_scale,
        static_gates=tuple(stat_gate(a) for a in ("q", "k", "v", "o")),
        shift=shift,
        m_norm=float(cfg.normalization_layer_epsilon),
        q_dynamic_scale=c_qk.astype(jnp.float32) * ones_n,
        k_dynamic_scale=c_qk.astype(jnp.float32) * ones_n,
        v_dynamic_scale=c_vo.astype(jnp.float32) * ones_n,
        o_dynamic_scale=c_vo.astype(jnp.float32) * ones_n,
        qk_norm=qk_norm_arg,
        keys_cnt=packed_qkvo,
        shift_carry="auto",
        stride_skew="auto",
    )
    q, k, v, o_m, M_pass = outs[:5]
    q = nn.with_logical_constraint(q, ("activation_kv_batch", "activation_length", "activation_kv_heads", "activation_kv_head_dim"))
    k = nn.with_logical_constraint(k, ("activation_kv_batch", "activation_length", "activation_kv_heads", "activation_kv_head_dim"))
    v = nn.with_logical_constraint(v, ("activation_kv_batch", "activation_length", "activation_kv_heads", "activation_kv_head_dim"))

    o = o_m + self._attend(q, k, v, segment_ids, window, seq_minor_default=False)
    o = nn.with_logical_constraint(o, ("activation_batch", "activation_length", "activation_heads", "activation_kv"))
    dx = self.W_o(o)

    if not self.write_M:
      return dx, M_pass, None
    xw = nn.gelu(raw["W_w1"] if fused else self.W_w1(x))
    km_raw = self.W_w2(xw)
    wg_logits = (raw["W_wg"] if fused else self.W_wg(x)) + jnp.asarray(self.b_wg, cfg.dtype)
    return dx, M_pass, (o, wg_logits, km_raw)

  def __call__(self, x: Array, positions: Array, segment_ids, M_in: Array, window: int):
    cfg, lc = self.config, self.loom_cfg
    x = nn.with_logical_constraint(x, ("activation_batch", "activation_length", "activation_embed"))
    if lc.loom_pallas and jax.default_backend() == "tpu":
      return self._pallas_forward(x, positions, segment_ids, M_in, window)

    d = cfg.head_dim
    read_eps, key_scale, stat_init = float(lc.loom_read_key_eps), float(lc.loom_read_key_scale), float(lc.loom_static_read_gate_init)

    z = jax.nn.sigmoid(self.W_sg(x).astype(jnp.float32) + self.b_sg.astype(jnp.float32))
    M_tilde = m_norm(m_shift(M_in, z, segment_ids, positions), float(cfg.normalization_layer_epsilon))
    xr = nn.gelu(self.W_r1(x))

    def m_read(arm: str, scale: Array) -> Array:
      r = getattr(self, f"W_r2_{arm}")(xr)
      g = jax.nn.sigmoid(getattr(self, f"W_rg_{arm}")(x) + jnp.asarray(getattr(self, f"b_rg_{arm}"), cfg.dtype))
      g_bar = (1.1 / stat_init) * jax.nn.sigmoid(
          getattr(self, f"W_rg_bar_{arm}")(x) + jnp.asarray(getattr(self, f"b_rg_bar_{arm}"), cfg.dtype)
      )
      q_dyn = key_scale * g[..., None] * rms_norm_unit(r, read_eps)
      q_stat = jnp.asarray(getattr(self, f"q_bar_{arm}"), M_tilde.dtype) * g_bar[..., None]
      u = jnp.einsum("btnk,btkd->btnd", q_dyn, M_tilde) + jnp.einsum("btnk,btkd->btnd", q_stat, M_tilde)
      return (u.astype(jnp.float32) * scale.astype(jnp.float32)).astype(cfg.dtype)

    q_tilde = m_read("q", self.c_qk[0])
    k_tilde = m_read("k", self.c_qk[0])
    o_m = m_read("o", self.c_vo[0])
    v = m_read("v", self.c_vo[0])

    qr, kr = self.W_q(x), self.W_k(x)
    if not getattr(cfg, "use_qk_norm", False):
      qr = self.rotary_q(qr, positions)
      kr = self.rotary_k(kr, positions)
      q = jnp.concatenate((q_tilde, qr), axis=-1)
      k = jnp.concatenate((k_tilde, kr), axis=-1)
    else:
      q = jnp.concatenate((q_tilde, qr), axis=-1)
      k = jnp.concatenate((k_tilde, kr), axis=-1)
      q = rms_norm_unit(q, float(cfg.normalization_layer_epsilon)) * jnp.asarray(self.q_norm_scale, cfg.dtype)
      k = rms_norm_unit(k, float(cfg.normalization_layer_epsilon)) * jnp.asarray(self.k_norm_scale, cfg.dtype)
      q = jnp.concatenate((q[..., :d], self.rotary_q(q[..., d:], positions)), axis=-1)
      k = jnp.concatenate((k[..., :d], self.rotary_k(k[..., d:], positions)), axis=-1)

    q = nn.with_logical_constraint(q, ("activation_kv_batch", "activation_length", "activation_kv_heads", "activation_kv_head_dim"))
    k = nn.with_logical_constraint(k, ("activation_kv_batch", "activation_length", "activation_kv_heads", "activation_kv_head_dim"))
    v = nn.with_logical_constraint(v, ("activation_kv_batch", "activation_length", "activation_kv_heads", "activation_kv_head_dim"))

    q_scaled = q / jnp.sqrt(jnp.asarray(self._logit_scale_dim(q.shape[-1]), cfg.dtype))
    o = o_m + self._attend(q_scaled, k, v, segment_ids, window, seq_minor_default=True)
    o = nn.with_logical_constraint(o, ("activation_batch", "activation_length", "activation_heads", "activation_kv"))
    dx = self.W_o(o)

    if not self.write_M:
      return dx, M_in, None
    xw = nn.gelu(self.W_w1(x))
    km_raw = self.W_w2(xw)
    wg_logits = self.W_wg(x) + jnp.asarray(self.b_wg, cfg.dtype)
    return dx, M_in, (o, wg_logits, km_raw)


class LoomDecoderLayer(nn.Module):
  """Single LoomFormer / LoomPythia block reusing `maxtext.layers.linears.mlp_block` & `flax.linen` norms."""

  config: Any
  mesh: Any
  model_mode: str = MODEL_MODE_TRAIN
  quant: Any = None
  loom_cfg: LoomConfig = LoomConfig()
  write_M: bool = True
  is_last_layer: bool = False

  @nn.compact
  def __call__(self, carry, decoder_segment_ids, decoder_positions, deterministic: bool = False,
               model_mode: str = MODEL_MODE_TRAIN, **_unused):
    cfg, lc = self.config, self.loom_cfg
    is_pythia = lc.loom_backbone == "pythia"
    loom_init, base_init, out_init = _model_inits(cfg, lc)
    x, M = carry

    x = nn.with_logical_constraint(x, ("activation_batch", "activation_norm_length", "activation_embed"))
    x = checkpoint_name(x, "decoder_layer_input")
    ln1_x = _decoder_norm(cfg, is_pythia, "ln_1")(x)
    ln1_x = nn.with_logical_constraint(ln1_x, ("activation_batch", "activation_norm_length", "activation_embed"))

    write_here = self.write_M and not self.is_last_layer
    dx, M, attn_write_factors = LoomMHA(
        config=cfg, loom_cfg=lc, mesh=self.mesh, write_M=write_here, name="loom_mha"
    )(ln1_x, decoder_positions, decoder_segment_ids, M, cfg.max_target_length)

    dx = nn.with_logical_constraint(dx, ("activation_batch", "activation_norm_length", "activation_embed"))
    x_mid = x + dx

    ln2_x = _decoder_norm(cfg, is_pythia, "ln_2")(x_mid)
    ln2_x = nn.with_logical_constraint(ln2_x, ("activation_batch", "activation_norm_length", "activation_embed"))

    # Reuse official `maxtext.layers.linears.mlp_block` for both `ffn_x` and `ffn_m`
    ffn_x_dim = lc.loom_last_ffn_x_dim if self.is_last_layer else cfg.mlp_dim
    if is_pythia:
      x_acts = (functools.partial(jax.nn.gelu, approximate=False),)
      m_acts = x_acts
    elif lc.loom_situ_beta1 > 0.0 and lc.loom_situ_beta2 > 0.0:
      x_acts = (
          functools.partial(situ_gate, beta=lc.loom_situ_beta1),
          functools.partial(situ_up, beta=lc.loom_situ_beta2),
      )
      m_acts = ("silu", "linear")
    else:
      x_acts = ("silu", "linear")
      m_acts = x_acts

    def ffn_x_init(key, shape, dtype, _in_ax, _out_ax):
      return (out_init if shape[-1] == cfg.emb_dim else base_init)(key, shape, dtype)

    df_x = linears.mlp_block(
        config=cfg, mesh=self.mesh, in_features=cfg.emb_dim, intermediate_dim=ffn_x_dim,
        activations=x_acts, kernel_init=ffn_x_init, intermediate_dropout_rate=cfg.dropout_rate,
        dtype=cfg.dtype, weight_dtype=cfg.weight_dtype, use_bias=is_pythia,
        quant=self.quant, model_mode=model_mode, name="ffn_x",
    )(ln2_x, deterministic=deterministic)
    df_x = nn.with_logical_constraint(df_x, ("activation_batch", "activation_norm_length", "activation_embed"))

    if write_here:
      n, d, d_k, d_r = cfg.num_query_heads, cfg.head_dim, lc.loom_d_k, lc.loom_d_r
      up_lead = None if _replicate_rank_up(lc, n, cfg.num_decoder_layers, self.mesh, d_r) else "embed"

      ffn_m_out = linears.mlp_block(
          config=cfg, mesh=self.mesh, in_features=cfg.emb_dim, intermediate_dim=lc.loom_ffn_m_dim,
          activations=m_acts, kernel_init=lambda k, s, dt, *_: loom_init(k, s, dt),
          intermediate_dropout_rate=cfg.dropout_rate, dtype=cfg.dtype, weight_dtype=cfg.weight_dtype,
          use_bias=False, quant=self.quant, model_mode=model_mode, name="ffn_m",
      )(ln2_x, deterministic=deterministic).reshape(ln2_x.shape[:2] + (n, d))

      xw_f = nn.gelu(_dense(d_r, loom_init, ("embed", None), cfg, name="ffn_W_w1")(ln2_x))
      km_f_raw = _dense((n, d_k), loom_init, (up_lead, "q_heads", "kv"), cfg, in_features=d_r, use_bias=True, name="ffn_W_w2")(xw_f)
      wg_f_logits = _dense(n, loom_init, ("embed", "q_heads"), cfg, name="ffn_W_wg")(ln2_x)
      b_wg_f = self.param("ffn_b_wg", nn.with_logical_partitioning(nn.initializers.constant(_logit(lc.loom_write_gate_init)), ("q_heads",)),
                          (n,), cfg.weight_dtype)
      wg_f_logits = wg_f_logits + jnp.asarray(b_wg_f, cfg.dtype)

      write_eps = float(lc.loom_write_eps if lc.loom_write_eps is not None else cfg.normalization_layer_epsilon)
      if lc.loom_pallas and jax.default_backend() == "tpu":
        M = loom_pallas.write(
            M, [attn_write_factors, (ffn_m_out, wg_f_logits, km_f_raw)],
            epsilon=write_eps, content_divisors=(None, 1.0),
            forward_tile=128, reverse_tile=128, vmem_mib=lc.loom_pallas_vmem_mib,
        )
      else:
        o_attn, wg_a_logits, km_a_raw = attn_write_factors
        km_a = rms_norm_unit(km_a_raw, write_eps) * jax.nn.sigmoid(wg_a_logits)[..., None]
        dM_attn = jnp.einsum("btnk,btnd->btkd", km_a, rms_norm_unit(o_attn, write_eps))
        km_f = rms_norm_unit(km_f_raw, write_eps) * jax.nn.sigmoid(wg_f_logits)[..., None]
        dM_ffn = jnp.einsum("btnk,btnd->btkd", km_f, ffn_m_out)
        M = (M + dM_attn + dM_ffn).astype(cfg.dtype)

    x_out = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,))(x_mid + df_x, deterministic=deterministic)
    x_out = nn.with_logical_constraint(x_out, ("activation_batch", "activation_norm_length", "activation_embed"))
    return (x_out, M), None


# ==============================================================================
# 4. Thin Subclasses of Official MaxText `Decoder` & `TransformerLinenPure`
# ==============================================================================

class LoomDecoder(decoders.Decoder):
  """Subclasses official `maxtext.layers.decoders.Decoder` to carry dual-stream `(x, M)`."""

  loom_cfg: LoomConfig = LoomConfig()

  def get_decoder_layers(self):
    return [LoomDecoderLayer]

  @nn.compact
  def apply_output_head(self, shared_embedding, y, deterministic, model_mode):
    del shared_embedding, model_mode
    cfg, lc = self.config, self.loom_cfg
    y = _decoder_norm(cfg, lc.loom_backbone == "pythia", "decoder_norm")(y)
    if y.ndim >= 3:
      y = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,))(y, deterministic=deterministic)
    _, base_init, _ = _model_inits(cfg, lc)
    dense = _dense(cfg.vocab_size, base_init, ("embed", "vocab"), cfg, name="logits_dense")
    if lc.loom_logits_bf16_dot_fp32_out and y.dtype == jnp.bfloat16:
      if self.is_initializing():
        _ = dense(y[..., :1, :])
      w = nn.meta.unbox(self.variables["params"]["logits_dense"]["kernel"]).astype(cfg.dtype)
      w = nn.with_logical_constraint(w, (None, "vocab"))
      logits = bf16_dot_fp32_output(y, w)
    else:
      logits = dense(y)
      if getattr(cfg, "cast_logits_to_fp32", True):
        logits = logits.astype(jnp.float32)
    spec = (
        ("activation_embed_and_logits_batch", "activation_length_no_exp", "activation_vocab")
        if logits.ndim == 3
        else ("activation_embed_and_logits_batch_sequence", "activation_vocab")
    )
    return nn.with_logical_constraint(logits, spec)

  @nn.compact
  def __call__(self, shared_embedding, decoder_input_tokens, decoder_positions,
               decoder_segment_ids=None, deterministic=False, model_mode=MODEL_MODE_TRAIN, **kwargs):
    cfg, lc = self.config, self.loom_cfg
    x = self._apply_embedding(
        shared_embedding, decoder_input_tokens, decoder_positions, deterministic, model_mode,
        kwargs.get("image_embeddings"), kwargs.get("bidirectional_mask"),
        kwargs.get("image_masks"), kwargs.get("audio_embeddings"), kwargs.get("audio_masks"),
    )
    M = LoomEmbedWrite(config=cfg, loom_cfg=lc, name="embed_write")(x)
    carry = (x, M)

    policy = self.get_remat_policy()
    RematLayer = self.set_remat_policy(self.decoder_layer, policy)[0]
    bcast = (decoder_segment_ids, decoder_positions, deterministic, model_mode)

    if cfg.scan_layers and cfg.num_decoder_layers > 1:
      carry, _ = self.scan_decoder_layers(
          cfg, RematLayer, cfg.num_decoder_layers - 1, "layers", self.mesh,
          in_axes_tuple=(nn.broadcast,) * len(bcast), model_mode=model_mode,
          loom_cfg=lc, write_M=True, is_last_layer=False,
      )(carry, *bcast)
      carry, _ = RematLayer(
          config=cfg, mesh=self.mesh, name="last_layer", quant=self.quant, model_mode=model_mode,
          loom_cfg=lc, write_M=False, is_last_layer=True,
      )(carry, *bcast)
    else:
      for lyr in range(cfg.num_decoder_layers):
        is_last = lyr == cfg.num_decoder_layers - 1
        carry, _ = RematLayer(
            config=cfg, mesh=self.mesh, name=f"layers_{lyr}", quant=self.quant, model_mode=model_mode,
            loom_cfg=lc, write_M=not is_last, is_last_layer=is_last,
        )(carry, *bcast)

    hidden_state = carry[0]
    if self.is_initializing() and cfg.num_vocab_tiling > 1:
      _ = self.apply_output_head(shared_embedding, hidden_state[:, :1], deterministic, model_mode)
    if cfg.num_vocab_tiling > 1 and self.model_mode == MODEL_MODE_TRAIN:
      self.sow("intermediates", "hidden_states", hidden_state)
      return None, hidden_state, None
    logits = self.apply_output_head(shared_embedding, hidden_state, deterministic, model_mode)
    return logits, hidden_state, None


class LoomTransformer(models.TransformerLinenPure):
  """Subclasses official `maxtext.models.models.TransformerLinenPure` to plug in `LoomDecoder`."""

  quant: Any = None
  loom_cfg: LoomConfig = LoomConfig()

  def setup(self):
    cfg, lc = self.config, self.loom_cfg
    lc.validate(cfg.num_query_heads, cfg.head_dim, cfg.emb_dim)
    loom_init, _, _ = _model_inits(cfg, lc)
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
