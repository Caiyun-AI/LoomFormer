"""FineWeb10B (GPT-2 tokenized) shard downloader and DDP-aware memmap dataloader.

Uses the exact `.bin` shard format from `https://github.com/KellerJordan/modded-nanogpt`
(`kjj0/fineweb10B-gpt2` on HuggingFace):
  - 256 x int32 header: magic == 20240520, version == 1, header[2] == num_tokens
  - Followed by `num_tokens` x uint16 GPT-2 token IDs (vocab size 50257, EOT/BOS = 50256)
  - Validation shard: `fineweb_val_000000.bin` (~100M tokens; modded-nanogpt evaluates 10,485,760 tokens)
  - Training shards: `fineweb_train_000001.bin`, `fineweb_train_000002.bin`, ... (100M tokens / ~200 MB each)
"""
import glob
import os
from typing import List, Tuple

import numpy as np
import torch

FINEWEB_MAGIC = 20240520
FINEWEB_VERSION = 1
HEADER_INT32 = 256
HEADER_BYTES = HEADER_INT32 * 4
DEFAULT_REPO_ID = "kjj0/fineweb10B-gpt2"


def download_fineweb10b_shards(
    data_dir: str,
    num_train_shards: int = 2,
    repo_id: str = DEFAULT_REPO_ID,
) -> Tuple[List[str], List[str]]:
  """Download `fineweb_val_000000.bin` and the first `num_train_shards` training shards if not already present."""
  os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
  from huggingface_hub import hf_hub_download

  os.makedirs(data_dir, exist_ok=True)
  val_names = ["fineweb_val_000000.bin"]
  train_names = [f"fineweb_train_{i:06d}.bin" for i in range(1, num_train_shards + 1)]

  def _fetch(fname: str) -> str:
    dst = os.path.join(data_dir, fname)
    if not os.path.exists(dst):
      print(f"[data] Downloading {repo_id}/{fname} -> {dst}", flush=True)
      hf_hub_download(repo_id=repo_id, filename=fname, repo_type="dataset", local_dir=data_dir)
    return dst

  val_files = [_fetch(f) for f in val_names]
  train_files = [_fetch(f) for f in train_names]
  return train_files, val_files


def load_fineweb_shard(path: str) -> np.memmap:
  """Memory-map a FineWeb `.bin` shard and validate its `modded-nanogpt` header."""
  header = np.fromfile(path, dtype=np.int32, count=HEADER_INT32)
  if header.size < HEADER_INT32:
    raise ValueError(f"Corrupt shard header (too short): {path}")
  if int(header[0]) != FINEWEB_MAGIC:
    raise ValueError(f"Magic mismatch in {path}: got {int(header[0])}, expected {FINEWEB_MAGIC}")
  if int(header[1]) != FINEWEB_VERSION:
    raise ValueError(f"Unsupported version in {path}: got {int(header[1])}, expected {FINEWEB_VERSION}")
  num_tokens = int(header[2])
  expected_bytes = HEADER_BYTES + 2 * num_tokens
  actual_bytes = os.path.getsize(path)
  if actual_bytes != expected_bytes:
    raise ValueError(f"Size mismatch in {path}: {actual_bytes} != {expected_bytes} bytes")
  return np.memmap(path, dtype=np.uint16, mode="r", offset=HEADER_BYTES, shape=(num_tokens,))


class FineWebShardLoader:
  """Deterministic, zero-copy memmap shard dataloader supporting multi-GPU DDP and epoch wrap-around."""

  def __init__(
      self,
      pattern_or_files,
      micro_batch_size: int,
      seq_len: int,
      rank: int = 0,
      world_size: int = 1,
      device: str = "cuda",
  ):
    if isinstance(pattern_or_files, str):
      self.files = sorted(glob.glob(pattern_or_files))
    else:
      self.files = sorted(list(pattern_or_files))
    if not self.files:
      raise FileNotFoundError(f"No FineWeb shards matched: {pattern_or_files}")

    self.B = micro_batch_size
    self.T = seq_len
    self.rank = rank
    self.world_size = world_size
    self.device = device
    self.tokens_per_rank = self.B * self.T
    self.tokens_per_step = self.tokens_per_rank * self.world_size

    self.shard_idx = 0
    self.tokens = load_fineweb_shard(self.files[0])
    self.pos = self.rank * self.tokens_per_rank
    self.total_tokens = sum(len(load_fineweb_shard(f)) for f in self.files)

  def reset(self) -> None:
    self.shard_idx = 0
    self.tokens = load_fineweb_shard(self.files[0])
    self.pos = self.rank * self.tokens_per_rank

  def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return `(x, y)` of shape `[B, T]` (`int64`) on `self.device`."""
    # Advance to next shard if current shard does not have enough tokens for all ranks + 1 target token.
    if self.pos + self.tokens_per_rank + 1 > len(self.tokens):
      self.shard_idx = (self.shard_idx + 1) % len(self.files)
      self.tokens = load_fineweb_shard(self.files[self.shard_idx])
      self.pos = self.rank * self.tokens_per_rank

    start = self.pos
    buf = np.asarray(self.tokens[start : start + self.tokens_per_rank + 1], dtype=np.int64)
    x = torch.from_numpy(buf[:-1]).view(self.B, self.T)
    y = torch.from_numpy(buf[1:]).view(self.B, self.T)

    # Advance pointer across all ranks cleanly
    self.pos += self.tokens_per_step
    if self.pos - self.rank * self.tokens_per_rank + self.tokens_per_step + 1 > len(self.tokens):
      # Wrap shard on next call if the next global step would overflow
      self.pos = len(self.tokens)

    if self.device != "cpu":
      x = x.pin_memory().to(self.device, non_blocking=True)
      y = y.pin_memory().to(self.device, non_blocking=True)
    return x, y
