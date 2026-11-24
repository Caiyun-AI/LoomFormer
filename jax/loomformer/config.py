"""LoomFormer configuration dataclass and YAML/CLI separator for official MaxText (`maxtext-v0.2.0`)."""
import dataclasses
import math
import os
import tempfile
from typing import Optional

import yaml


@dataclasses.dataclass(frozen=True)
class LoomConfig:
  """Architecture hyperparameters specific to LoomFormer / LoomPythia (Section 3.2 & Appendix D)."""

  loom_backbone: str = "llama"  # 'llama' (RMSNorm + SwiGLU/SiTU) | 'pythia' (NeoX LayerNorm + GELU + bias)
  loom_d_k: int = 32
  loom_d_r: int = 256
  loom_rope_dim: int = 18
  loom_ffn_m_dim: int = 1128
  loom_last_ffn_x_dim: int = 3904
  loom_pallas: bool = True
  loom_splash: bool = True
  loom_splash_seq_minor: Optional[bool] = None
  loom_splash_ignore_segments: bool = False
  loom_splash_residual_name: str = ""
  loom_pallas_vmem_mib: int = 48
  loom_replicate_rank_up: str = "auto"  # 'auto' | 'true' | 'false'
  loom_read_gate_init: float = 0.05
  loom_static_read_gate_init: float = 0.99
  loom_write_gate_init: float = 0.1
  loom_m_shift_init: float = 0.5
  loom_read_key_scale: float = 0.1
  loom_read_key_eps: float = 1e-4
  loom_write_eps: Optional[float] = None
  loom_embed_write_divisor: Optional[float] = None
  loom_qk_scale_dim: str = "head_dim"  # 'head_dim' | 'qk_dim'
  loom_clone_vo_init: bool = True
  loom_init_std: float = 0.006
  loom_situ_beta1: float = 0.0
  loom_situ_beta2: float = 0.0
  loom_logits_bf16_dot_fp32_out: bool = False

  def validate(self, num_heads: int, head_dim: int, emb_dim: int) -> None:
    if self.loom_backbone not in ("llama", "pythia"):
      raise ValueError(f"loom_backbone must be 'llama' or 'pythia', got {self.loom_backbone!r}")
    if num_heads * head_dim != emb_dim:
      raise ValueError(f"LoomFormer requires emb_dim == num_query_heads * head_dim ({emb_dim} != {num_heads} * {head_dim})")
    for name in ("loom_d_k", "loom_d_r", "loom_rope_dim", "loom_ffn_m_dim", "loom_last_ffn_x_dim"):
      val = getattr(self, name)
      if not isinstance(val, int) or val <= 0:
        raise ValueError(f"{name} must be a positive integer, got {val!r}")
    if self.loom_rope_dim % 2 != 0:
      raise ValueError(f"loom_rope_dim must be even, got {self.loom_rope_dim}")
    if not 0.0 < self.loom_static_read_gate_init < 1.1:
      raise ValueError(f"loom_static_read_gate_init must be in (0, 1.1), got {self.loom_static_read_gate_init}")
    if (self.loom_situ_beta1 != 0.0 or self.loom_situ_beta2 != 0.0) and not (
        math.isfinite(self.loom_situ_beta1) and self.loom_situ_beta1 > 0
        and math.isfinite(self.loom_situ_beta2) and self.loom_situ_beta2 > 0
    ):
      raise ValueError("SiTU requires both loom_situ_beta1 and loom_situ_beta2 to be finite positive numbers")


LOOM_FIELD_TYPES = {f.name: f.type for f in dataclasses.fields(LoomConfig)}


def _parse_scalar(raw: str):
  low = raw.strip().lower()
  if low in ("none", "null"):
    return None
  if low == "true":
    return True
  if low == "false":
    return False
  try:
    return int(raw)
  except ValueError:
    pass
  try:
    return float(raw)
  except ValueError:
    return raw


def extract_loom_and_maxtext_argv(argv: list[str]) -> tuple[LoomConfig, list[str], str]:
  """Split a LoomFormer YAML + CLI `argv` into `(LoomConfig, clean_maxtext_argv, tmp_yaml_path)`.

  Official `maxtext-v0.2.0` uses Pydantic `ConfigDict(extra='forbid')`. Separating `loom_*` keys here
  lets us pass a 100% standard `MaxText` config without patching `maxtext.configs` at all.
  """
  if len(argv) < 2:
    raise ValueError("Usage: python train_loom.py <path/to/loom_config.yml> [key=value ...]")

  yml_path = argv[1]
  with open(yml_path, "r") as f:
    raw_yml = yaml.safe_load(f) or {}

  loom_kwargs = {}
  mt_yml = {}
  for k, v in raw_yml.items():
    if k in LOOM_FIELD_TYPES:
      loom_kwargs[k] = v
    else:
      mt_yml[k] = v

  mt_yml.setdefault("base_config", "base.yml")

  clean_cli = [argv[0]]
  for arg in argv[2:]:
    if "=" in arg:
      k, v = arg.split("=", 1)
      if k in LOOM_FIELD_TYPES:
        loom_kwargs[k] = _parse_scalar(v)
        continue
    clean_cli.append(arg)

  fd, tmp_yml = tempfile.mkstemp(prefix="loom_maxtext_", suffix=".yml")
  with os.fdopen(fd, "w") as f:
    yaml.safe_dump(mt_yml, f, sort_keys=False)
  clean_cli.insert(1, tmp_yml)

  return LoomConfig(**loom_kwargs), clean_cli, tmp_yml
