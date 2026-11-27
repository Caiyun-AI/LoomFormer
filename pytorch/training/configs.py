"""Parameter-matched model presets for LoomFormer (`loom_enabled=True`) and Llama (`loom_enabled=False`).

Available presets (`vocab_size` untied embeddings, matching JAX LoomFormer configs):
  - `160m` (vocab_size=50304, D=768, L=12, n=12, d=64):
      - Llama-162M (`loom_enabled=False`):    162,220,800 params (84.95M non-embedding)
      - LoomLlama-162M (`loom_enabled=True`): 162,223,020 params (84.96M non-embedding, +0.0014% vs Llama)
  - `medium` (vocab_size=50432, D=1200, L=18, n=16, d=75; Paper Section 5.1 & Appendix D):
      - Llama-Medium (`loom_enabled=False`):     432,121,200 params (311.08M non-embedding)
      - LoomFormer-Medium (`loom_enabled=True`): 432,112,228 params (311.07M non-embedding, -0.002% vs Llama)
  - `large` (vocab_size=50432, D=1536, L=22, n=16, d=96; Paper Section 5.1 & Appendix D):
      - Llama-Large (`loom_enabled=False`):      777,850,368 params (622.92M non-embedding)
      - LoomFormer-Large (`loom_enabled=True`):  777,849,500 params (622.92M non-embedding, -0.0001% vs Llama)
  - `xl` (vocab_size=50432, D=1920, L=28, n=20, d=96; Paper Section 5.1 & Appendix D):
      - Llama-XL (`loom_enabled=False`):         1,432,398,720 params (1.239B non-embedding)
      - LoomFormer-XL (`loom_enabled=True`):     1,432,396,996 params (1.239B non-embedding, -0.0001% vs Llama)
  - `2.8b` (vocab_size=50304, D=2560, L=32, n=32, d=80; Paper 2.8B head/slot geometry on Llama SwiGLU backbone):
      - Llama-2.8B (`loom_enabled=False`):       2,773,649,920 params (2.516B non-embedding)
      - LoomFormer-2.8B (`loom_enabled=True`):   2,773,824,032 params (2.516B non-embedding, +0.0063% vs Llama)
"""
import math
from typing import Dict

from loomformer import LoomFormerConfig


def default_embed_write_divisor(hidden_size: int, initializer_range: float = 0.02) -> float:
  """Calibrate `embed_write_divisor` so `W_e(embed_tokens(ids)) / divisor` has unit RMS at initialization."""
  return math.sqrt(hidden_size) * (initializer_range ** 2)


def _build_base_config(
    loom_enabled: bool,
    vocab_size: int,
    hidden_size: int,
    num_hidden_layers: int,
    num_attention_heads: int,
    head_dim: int,
    max_position_embeddings: int,
    situ_beta1: float,
    situ_beta2: float,
    initializer_range: float,
    **overrides,
) -> LoomFormerConfig:
  defaults = dict(
      vocab_size=vocab_size,
      hidden_size=hidden_size,
      num_hidden_layers=num_hidden_layers,
      num_attention_heads=num_attention_heads,
      head_dim=head_dim,
      rms_norm_eps=1e-6,
      situ_beta1=situ_beta1,
      situ_beta2=situ_beta2,
      max_position_embeddings=max_position_embeddings,
      initializer_range=initializer_range,
      tie_word_embeddings=False,
      loom_enabled=loom_enabled,
  )
  if loom_enabled:
    defaults.update(
        read_key_eps=1e-4,
        read_key_scale=0.1,
        write_eps=1e-6,
        read_gate_init=0.05,
        static_read_gate_init=0.99,
        write_gate_init=0.1,
        m_shift_init=0.5,
        clone_vo_init=True,
        embed_write_divisor=default_embed_write_divisor(hidden_size, initializer_range),
        ffn_write_divisor=1.0,
    )
  else:
    defaults.update(mha_nope_dim=0)
  defaults.update(overrides)
  return LoomFormerConfig(**defaults)


def build_loomllama_160m_config(
    loom_enabled: bool = True,
    vocab_size: int = 50304,
    max_position_embeddings: int = 2048,
    situ_beta1: float = 0.0,
    situ_beta2: float = 0.0,
    initializer_range: float = 0.02,
    **overrides,
) -> LoomFormerConfig:
  """Return a ~162.22M parameter `LoomFormerConfig` (D=768, L=12, n=12, d=64) for LoomLlama or matched Llama."""
  hidden_size = overrides.pop("hidden_size", 768)
  num_hidden_layers = overrides.pop("num_hidden_layers", 12)
  num_attention_heads = overrides.pop("num_attention_heads", 12)
  head_dim = overrides.pop("head_dim", hidden_size // num_attention_heads)
  preset = (
      dict(intermediate_size=1648, ffn_m_dim=736, last_ffn_x_dim=2560, d_k=24, d_r=128, rope_dim=16)
      if loom_enabled
      else dict(intermediate_size=2048, last_ffn_x_dim=2048)
  )
  preset.update(overrides)
  return _build_base_config(
      loom_enabled=loom_enabled,
      vocab_size=vocab_size,
      hidden_size=hidden_size,
      num_hidden_layers=num_hidden_layers,
      num_attention_heads=num_attention_heads,
      head_dim=head_dim,
      max_position_embeddings=max_position_embeddings,
      situ_beta1=situ_beta1,
      situ_beta2=situ_beta2,
      initializer_range=initializer_range,
      **preset,
  )


def build_loomllama_medium_config(
    loom_enabled: bool = True,
    vocab_size: int = 50432,
    max_position_embeddings: int = 4096,
    situ_beta1: float = 0.0,
    situ_beta2: float = 0.0,
    initializer_range: float = 0.006,
    **overrides,
) -> LoomFormerConfig:
  """Return the paper's ~432M Medium `LoomFormerConfig` (D=1200, L=18, n=16, d=75) or matched Llama baseline."""
  hidden_size = overrides.pop("hidden_size", 1200)
  num_hidden_layers = overrides.pop("num_hidden_layers", 18)
  num_attention_heads = overrides.pop("num_attention_heads", 16)
  head_dim = overrides.pop("head_dim", hidden_size // num_attention_heads)
  preset = (
      dict(
          intermediate_size=2510,
          ffn_m_dim=1128,
          last_ffn_x_dim=3904,
          d_k=32,
          d_r=256,
          rope_dim=18,
          embed_write_divisor=0.0016052833,
      )
      if loom_enabled
      else dict(intermediate_size=3200, last_ffn_x_dim=3200, mha_nope_dim=1)
  )
  preset.update(overrides)
  return _build_base_config(
      loom_enabled=loom_enabled,
      vocab_size=vocab_size,
      hidden_size=hidden_size,
      num_hidden_layers=num_hidden_layers,
      num_attention_heads=num_attention_heads,
      head_dim=head_dim,
      max_position_embeddings=max_position_embeddings,
      situ_beta1=situ_beta1,
      situ_beta2=situ_beta2,
      initializer_range=initializer_range,
      **preset,
  )


def build_loomllama_large_config(
    loom_enabled: bool = True,
    vocab_size: int = 50432,
    max_position_embeddings: int = 4096,
    situ_beta1: float = 0.0,
    situ_beta2: float = 0.0,
    initializer_range: float = 0.006,
    **overrides,
) -> LoomFormerConfig:
  """Return ~778M Paper-Large `LoomFormerConfig` (D=1536, L=22, n=16, d=96) or parameter-matched Llama baseline."""
  hidden_size = overrides.pop("hidden_size", 1536)
  num_hidden_layers = overrides.pop("num_hidden_layers", 22)
  num_attention_heads = overrides.pop("num_attention_heads", 16)
  head_dim = overrides.pop("head_dim", hidden_size // num_attention_heads)
  preset = (
      dict(
          intermediate_size=3317,
          ffn_m_dim=1539,
          last_ffn_x_dim=5083,
          d_k=32,
          d_r=256,
          rope_dim=24,
          embed_write_divisor=0.0018161707314488303,
      )
      if loom_enabled
      else dict(intermediate_size=4096, last_ffn_x_dim=4096, mha_nope_dim=0)
  )
  preset.update(overrides)
  return _build_base_config(
      loom_enabled=loom_enabled,
      vocab_size=vocab_size,
      hidden_size=hidden_size,
      num_hidden_layers=num_hidden_layers,
      num_attention_heads=num_attention_heads,
      head_dim=head_dim,
      max_position_embeddings=max_position_embeddings,
      situ_beta1=situ_beta1,
      situ_beta2=situ_beta2,
      initializer_range=initializer_range,
      **preset,
  )


def build_loomllama_xl_config(
    loom_enabled: bool = True,
    vocab_size: int = 50432,
    max_position_embeddings: int = 4096,
    situ_beta1: float = 0.0,
    situ_beta2: float = 0.0,
    initializer_range: float = 0.006,
    **overrides,
) -> LoomFormerConfig:
  """Return ~1.43B Paper-XL `LoomFormerConfig` (D=1920, L=28, n=20, d=96) or parameter-matched Llama baseline."""
  hidden_size = overrides.pop("hidden_size", 1920)
  num_hidden_layers = overrides.pop("num_hidden_layers", 28)
  num_attention_heads = overrides.pop("num_attention_heads", 20)
  head_dim = overrides.pop("head_dim", hidden_size // num_attention_heads)
  preset = (
      dict(
          intermediate_size=4048,
          ffn_m_dim=1827,
          last_ffn_x_dim=6277,
          d_k=40,
          d_r=400,
          rope_dim=24,
          embed_write_divisor=0.0020305404338184567,
      )
      if loom_enabled
      else dict(intermediate_size=5120, last_ffn_x_dim=5120, mha_nope_dim=0)
  )
  preset.update(overrides)
  return _build_base_config(
      loom_enabled=loom_enabled,
      vocab_size=vocab_size,
      hidden_size=hidden_size,
      num_hidden_layers=num_hidden_layers,
      num_attention_heads=num_attention_heads,
      head_dim=head_dim,
      max_position_embeddings=max_position_embeddings,
      situ_beta1=situ_beta1,
      situ_beta2=situ_beta2,
      initializer_range=initializer_range,
      **preset,
  )


def build_loomllama_2p8b_config(
    loom_enabled: bool = True,
    vocab_size: int = 50304,
    max_position_embeddings: int = 4096,
    situ_beta1: float = 0.0,
    situ_beta2: float = 0.0,
    initializer_range: float = 0.006,
    **overrides,
) -> LoomFormerConfig:
  """Return ~2.77B `LoomFormerConfig` matching the paper's 2.8B head/slot geometry (D=2560, L=32, n=32, d=80) or Llama baseline."""
  hidden_size = overrides.pop("hidden_size", 2560)
  num_hidden_layers = overrides.pop("num_hidden_layers", 32)
  num_attention_heads = overrides.pop("num_attention_heads", 32)
  head_dim = overrides.pop("head_dim", hidden_size // num_attention_heads)
  preset = (
      dict(
          intermediate_size=5203,
          ffn_m_dim=1962,
          last_ffn_x_dim=6653,
          d_k=64,
          d_r=640,
          rope_dim=20,
          embed_write_divisor=0.0020825863,
      )
      if loom_enabled
      else dict(intermediate_size=6824, last_ffn_x_dim=6824, mha_nope_dim=0)
  )
  preset.update(overrides)
  return _build_base_config(
      loom_enabled=loom_enabled,
      vocab_size=vocab_size,
      hidden_size=hidden_size,
      num_hidden_layers=num_hidden_layers,
      num_attention_heads=num_attention_heads,
      head_dim=head_dim,
      max_position_embeddings=max_position_embeddings,
      situ_beta1=situ_beta1,
      situ_beta2=situ_beta2,
      initializer_range=initializer_range,
      **preset,
  )


MODEL_PRESETS = {
    "160m": build_loomllama_160m_config,
    "medium": build_loomllama_medium_config,
    "large": build_loomllama_large_config,
    "xl": build_loomllama_xl_config,
    "2.8b": build_loomllama_2p8b_config,
}


def build_preset_config(preset: str, loom_enabled: bool = True, **overrides) -> LoomFormerConfig:
  """Build a named model preset (`'160m'`, `'medium'`, `'large'`, `'xl'`, or `'2.8b'`) with optional field overrides."""
  if preset not in MODEL_PRESETS:
    raise ValueError(f"Unknown model preset {preset!r}; expected one of {sorted(MODEL_PRESETS)}")
  return MODEL_PRESETS[preset](loom_enabled=loom_enabled, **overrides)


def count_parameters(model) -> Dict[str, int]:
  """Return total, trainable, and non-embedding parameter counts."""
  total = sum(p.numel() for p in model.parameters())
  trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
  emb = 0
  if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
    emb += model.model.embed_tokens.weight.numel()
  if hasattr(model, "lm_head"):
    emb += model.lm_head.weight.numel()
  return {"total": total, "trainable": trainable, "embedding_and_head": emb, "non_embedding": total - emb}
