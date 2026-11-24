from .config import LoomConfig, extract_loom_and_maxtext_argv
from .loomformer import (
    LoomDecoder,
    LoomDecoderLayer,
    LoomEmbedWrite,
    LoomFormer,
    LoomMHA,
    MRead,
    MShift,
    MWrite,
)
from .loompythia import (
    LoomPythia,
    LoomPythiaDecoder,
    LoomPythiaDecoderLayer,
    LoomPythiaMHA,
    bf16_dot_fp32_output,
)


def build_loom_model(config, mesh, quant=None, loom_cfg: LoomConfig = LoomConfig(), model_mode: str = "train"):
  """Instantiate `LoomFormer` (`loom_backbone='llama'`) or `LoomPythia` (`loom_backbone='pythia'`)."""
  cls = LoomPythia if loom_cfg.loom_backbone == "pythia" else LoomFormer
  return cls(config=config, mesh=mesh, quant=quant, loom_cfg=loom_cfg, model_mode=model_mode)


__all__ = [
    "LoomConfig",
    "extract_loom_and_maxtext_argv",
    "build_loom_model",
    "MShift",
    "MRead",
    "MWrite",
    "LoomEmbedWrite",
    "LoomMHA",
    "LoomDecoderLayer",
    "LoomDecoder",
    "LoomFormer",
    "LoomPythiaMHA",
    "LoomPythiaDecoderLayer",
    "LoomPythiaDecoder",
    "LoomPythia",
    "bf16_dot_fp32_output",
]
