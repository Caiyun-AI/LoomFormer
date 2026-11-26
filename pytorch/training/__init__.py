"""PyTorch training utilities for LoomFormer (LoomLlama) and matched Llama baselines."""
from .configs import (
    MODEL_PRESETS,
    build_loomllama_160m_config,
    build_loomllama_medium_config,
    build_preset_config,
    count_parameters,
)
from .data import FineWebShardLoader, download_fineweb10b_shards, load_fineweb_shard
from .engine import FastLoomLayer, FastMHALayer, FastTrainModel, build_train_model

__all__ = [
    "MODEL_PRESETS",
    "build_loomllama_160m_config",
    "build_loomllama_medium_config",
    "build_preset_config",
    "count_parameters",
    "FineWebShardLoader",
    "download_fineweb10b_shards",
    "load_fineweb_shard",
    "FastLoomLayer",
    "FastMHALayer",
    "FastTrainModel",
    "build_train_model",
]
