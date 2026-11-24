"""LoomPythia (GPT-NeoX / Pythia backbone with Loom Attention, Section 5.2) in JAX/Flax.

Reuses the matrix-memory primitives (`MShift`, `MRead`, `MWrite`, `LoomEmbedWrite`) directly from
`loomformer.py`, pairing them cleanly with Pythia's NeoX backbone conventions:
  - `flax.linen.LayerNorm` + linear biases + exact `GELU` FFN + Wang/SmallInit scaling
  - Full-head `QKNorm` applied before RoPE on the `r`-dim tail, with `1 / sqrt(d + r)` logit scaling
  - `bf16_dot_fp32_output` lossless MXU FP32 logit accumulator (`@jax.custom_vjp`)

Every module follows the same PyTorch-familiar `setup()` + `__call__()` structure as `loomformer.py`.
"""
import functools
import math
from typing import Any

from flax import linen as nn
import jax
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp

from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.layers import decoders, embeddings, linears, nnx_wrappers
from maxtext.layers.initializers import variable_to_logically_partitioned
from maxtext.models import models

from . import loom_pallas
from .config import LoomConfig
from .loomformer import (
    LoomEmbedWrite,
    MRead,
    MShift,
    MWrite,
    bf16_dot_fp32_output,
    dense_proj,
    m_norm,
    m_shift_valid_mask,
    replicate_rank_up,
    rms_norm_unit,
    strict_causal_attention,
)

Array = jax.Array


# ==============================================================================
# 1. Pythia Backbone Helpers
# ==============================================================================


def layer_norm(cfg: Any, name: str):
  """Standard `flax.linen.LayerNorm` (with bias) and FP32 statistics."""
  return nn.LayerNorm(
      epsilon=cfg.normalization_layer_epsilon,
      dtype=cfg.dtype,
      param_dtype=cfg.weight_dtype,
      scale_init=nn.with_logical_partitioning(nn.initializers.ones, ("norm",)),
      bias_init=nn.with_logical_partitioning(nn.initializers.zeros, ("norm",)),
      name=name,
  )


def pythia_inits(cfg: Any, lc: LoomConfig):
  """GPT-NeoX / Pythia SmallInit & Wang output-layer initialization."""
  loom_init = nn.initializers.normal(lc.loom_init_std)
  base_init = nn.initializers.normal(math.sqrt(2.0 / (5.0 * cfg.emb_dim)))
  out_init = nn.initializers.normal(2.0 / (cfg.num_decoder_layers * math.sqrt(cfg.emb_dim)))
  return loom_init, base_init, out_init


# ==============================================================================
# 2. LoomPythia Blocks (`LoomPythiaMHA`, `LoomPythiaDecoderLayer`, `LoomPythiaDecoder`, `LoomPythia`)
# ==============================================================================

class LoomPythiaMHA(nn.Module):
  """Loom Multi-Head Attention on Pythia backbone (linear biases + full-head QKNorm + `1/sqrt(d+r)`)."""

  config: Any
  loom_cfg: LoomConfig
  mesh: Any
  write_M: bool = True

  def setup(self):
    cfg, lc = self.config, self.loom_cfg
    n, d = cfg.num_query_heads, cfg.head_dim
    d_r, r_dim = lc.loom_d_r, lc.loom_rope_dim
    loom_init, base_init, out_init = pythia_inits(cfg, lc)
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

    self.W_q = dense_proj((n, r_dim), base_init, ("embed", "q_heads", "kv"), cfg, use_bias=True, name="W_q")
    self.W_k = dense_proj((n, r_dim), base_init, ("embed", "kv_heads", "kv_head_dim"), cfg, use_bias=True, name="W_k")
    self.q_norm_scale = self.param(
        "q_norm_scale", nn.with_logical_partitioning(nn.initializers.ones, ("embed",)),
        (d + r_dim,), cfg.weight_dtype,
    )
    self.k_norm_scale = self.param(
        "k_norm_scale", nn.with_logical_partitioning(nn.initializers.ones, ("embed",)),
        (d + r_dim,), cfg.weight_dtype,
    )
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
    self.W_o = dense_proj(cfg.emb_dim, out_init, ("heads", "kv", "embed"), cfg,
                          in_features=(n, d), axis=(-2, -1), use_bias=True, name="W_o")

    if self.write_M:
      self.W_w1 = dense_proj(d_r, loom_init, ("embed", None), cfg, name="W_w1")
      self.m_write = MWrite(config=cfg, loom_cfg=lc, up_lead_axis=up_lead, name="m_write")

  def _logit_scale_dim(self) -> int:
    full_qk = self.config.head_dim + self.loom_cfg.loom_rope_dim
    return full_qk if self.loom_cfg.loom_qk_scale_dim == "qk_dim" else self.config.head_dim

  def attend(self, q: Array, k: Array, v: Array, segment_ids, window: int, seq_minor_default: bool) -> Array:
    return strict_causal_attention(q, k, v, segment_ids, self.config, self.loom_cfg, self.mesh, window, seq_minor_default)

  def __call__(self, x: Array, positions: Array, segment_ids, M_in: Array, window: int):
    cfg, lc = self.config, self.loom_cfg
    d = cfg.head_dim
    use_qknorm = bool(getattr(cfg, "use_qk_norm", True))
    scale_dim = self._logit_scale_dim()
    x = nn.with_logical_constraint(x, ("activation_batch", "activation_length", "activation_embed"))
    if lc.loom_pallas and jax.default_backend() == "tpu":
      valid = m_shift_valid_mask(x.shape, segment_ids, positions)
      return loom_pallas.mha_forward(
          self, x, positions, segment_ids, M_in, window,
          use_qk_norm=use_qknorm, logit_scale_dim=scale_dim, valid_shift_mask=valid,
      )

    # 1. Vertical reads from shifted & normalized matrix memory M
    M_tilde = m_norm(self.m_shift(M_in, x, segment_ids, positions), float(cfg.normalization_layer_epsilon))
    xr = nn.gelu(self.W_r1(x))                                                # btD,Dm->btm
    q_tilde = self.read_q(M_tilde, x, xr, self.c_qk[0])                       # btnk,btkd->btnd
    k_tilde = self.read_k(M_tilde, x, xr, self.c_qk[0])                       # btnk,btkd->btnd
    o_m = self.read_o(M_tilde, x, xr, self.c_vo[0])                           # btnk,btkd->btnd (run before read_v for vo_clone_init)
    v = self.read_v(M_tilde, x, xr, self.c_vo[0])                             # btnk,btkd->btnd

    # 2. Concatenate RoPE tail projections, apply full-head QKNorm, then RoPE on tail [d : d+r]
    qr, kr = self.W_q(x), self.W_k(x)                                         # btD,D(nr)->btnr
    if use_qknorm:
      q = jnp.concatenate((q_tilde, qr), axis=-1)                             # btnd,btnr->btnh (h = d + r)
      k = jnp.concatenate((k_tilde, kr), axis=-1)                             # btnd,btnr->btnh
      q = rms_norm_unit(q, float(cfg.normalization_layer_epsilon)) * jnp.asarray(self.q_norm_scale, cfg.dtype)
      k = rms_norm_unit(k, float(cfg.normalization_layer_epsilon)) * jnp.asarray(self.k_norm_scale, cfg.dtype)
      q = jnp.concatenate((q[..., :d], self.rotary_q(q[..., d:], positions)), axis=-1)
      k = jnp.concatenate((k[..., :d], self.rotary_k(k[..., d:], positions)), axis=-1)
    else:
      q = jnp.concatenate((q_tilde, self.rotary_q(qr, positions)), axis=-1)
      k = jnp.concatenate((k_tilde, self.rotary_k(kr, positions)), axis=-1)

    q = nn.with_logical_constraint(q, ("activation_kv_batch", "activation_length", "activation_kv_heads", "activation_kv_head_dim"))
    k = nn.with_logical_constraint(k, ("activation_kv_batch", "activation_length", "activation_kv_heads", "activation_kv_head_dim"))
    v = nn.with_logical_constraint(v, ("activation_kv_batch", "activation_length", "activation_kv_heads", "activation_kv_head_dim"))

    # 3. Strictly causal horizontal attention + local bypass o_m, then write back to M
    q_scaled = q / jnp.sqrt(jnp.asarray(scale_dim, cfg.dtype))
    o = o_m + self.attend(q_scaled, k, v, segment_ids, window, seq_minor_default=True)  # btnd
    o = nn.with_logical_constraint(o, ("activation_batch", "activation_length", "activation_heads", "activation_kv"))
    dx = self.W_o(o)                                                          # btnd,(nd)D->btD

    if not self.write_M:
      return dx, M_in, None
    xw = nn.gelu(self.W_w1(x))                                                # btD,Dm->btm
    return dx, M_in, self.m_write.factors(o, x, xw)


class LoomPythiaDecoderLayer(nn.Module):
  """Single LoomPythia serial pre-LayerNorm block (`setup` + `__call__`)."""

  config: Any
  mesh: Any
  model_mode: str = MODEL_MODE_TRAIN
  quant: Any = None
  loom_cfg: LoomConfig = LoomConfig()
  write_M: bool = True
  is_last_layer: bool = False

  def setup(self):
    cfg, lc = self.config, self.loom_cfg
    loom_init, base_init, out_init = pythia_inits(cfg, lc)
    write_here = self.write_M and not self.is_last_layer

    self.ln_1 = layer_norm(cfg, "ln_1")
    self.loom_mha = LoomPythiaMHA(config=cfg, loom_cfg=lc, mesh=self.mesh, write_M=write_here, name="loom_mha")
    self.ln_2 = layer_norm(cfg, "ln_2")

    ffn_x_dim = lc.loom_last_ffn_x_dim if self.is_last_layer else cfg.mlp_dim
    exact_gelu = (functools.partial(jax.nn.gelu, approximate=False),)

    def ffn_x_init(key, shape, dtype, _in_ax, _out_ax):
      return (out_init if shape[-1] == cfg.emb_dim else base_init)(key, shape, dtype)

    self.ffn_x = linears.mlp_block(
        config=cfg, mesh=self.mesh, in_features=cfg.emb_dim, intermediate_dim=ffn_x_dim,
        activations=exact_gelu, kernel_init=ffn_x_init,
        intermediate_dropout_rate=cfg.dropout_rate, dtype=cfg.dtype, weight_dtype=cfg.weight_dtype,
        use_bias=True, quant=self.quant, model_mode=self.model_mode, name="ffn_x",
    )

    if write_here:
      n, d_r = cfg.num_query_heads, lc.loom_d_r
      up_lead = None if replicate_rank_up(lc, n, cfg.num_decoder_layers, self.mesh, d_r) else "embed"
      self.ffn_m = linears.mlp_block(
          config=cfg, mesh=self.mesh, in_features=cfg.emb_dim, intermediate_dim=lc.loom_ffn_m_dim,
          activations=exact_gelu, kernel_init=lambda k, s, dt, *_: loom_init(k, s, dt),
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

    h1 = nn.with_logical_constraint(self.ln_1(x), ("activation_batch", "activation_norm_length", "activation_embed"))
    dx, M, attn_write = self.loom_mha(h1, decoder_positions, decoder_segment_ids, M, cfg.max_target_length)
    x_mid = x + nn.with_logical_constraint(dx, ("activation_batch", "activation_norm_length", "activation_embed"))

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


class LoomPythiaDecoder(decoders.Decoder):
  """Dual-stream `(x, M)` LoomPythia decoder stack (`setup` + `__call__`)."""

  loom_cfg: LoomConfig = LoomConfig()

  def get_decoder_layers(self):
    return [LoomPythiaDecoderLayer]

  def setup(self):
    cfg, lc = self.config, self.loom_cfg
    _, base_init, _ = pythia_inits(cfg, lc)

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

    self.decoder_norm = layer_norm(cfg, "decoder_norm")
    self.out_dropout = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,))
    self.logits_dense = dense_proj(cfg.vocab_size, base_init, ("embed", "vocab"), cfg, name="logits_dense")

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


class LoomPythia(models.TransformerLinenPure):
  """Top-level LoomPythia language model plugging `LoomPythiaDecoder` into MaxText's `TransformerLinenPure`."""

  quant: Any = None
  loom_cfg: LoomConfig = LoomConfig(loom_backbone="pythia")

  def setup(self):
    cfg, lc = self.config, self.loom_cfg
    lc.validate(cfg.num_query_heads, cfg.head_dim, cfg.emb_dim)
    loom_init, _, _ = pythia_inits(cfg, lc)
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
    self.decoder = LoomPythiaDecoder(
        config=cfg, mesh=self.mesh, quant=self.quant, model_mode=self.model_mode, loom_cfg=lc, name="decoder"
    )
