"""Static KV and matrix-memory (`M`) caches for CUDA-graph-compatible autoregressive decoding."""
import torch
from torch import nn

from .decode_attention import kt_cache_shape, kt_from_rows


class State(nn.Module):
  """Preallocated per-layer KV + `M` buffers with device-side scalar `pos` (CUDA-graph safe)."""

  def __init__(
      self,
      layers: int,
      batch: int,
      heads: int,
      max_len: int,
      dqk: int,
      dv: int,
      m_shape=None,
      dtype: torch.dtype = torch.bfloat16,
      device="cuda",
      layout: str = "rows",
  ):
    super().__init__()
    self.max_len, self.layers, self.batch, self.layout = max_len, layers, batch, layout
    kshape = kt_cache_shape(batch, heads, max_len, dqk) if layout == "kt" else (batch, heads, max_len, dqk)
    for i in range(layers):
      self.register_buffer(f"k{i}", torch.zeros(kshape, dtype=dtype, device=device))
      if layout == "kt":
        self.register_buffer(f"kt{i}", torch.zeros(batch, heads, 2, kshape[-1], dqk, dtype=dtype, device=device))
      self.register_buffer(f"v{i}", torch.zeros(batch, heads, max_len, dv, dtype=dtype, device=device))
      if m_shape is not None:
        self.register_buffer(f"m{i}", torch.zeros(batch, *m_shape, dtype=dtype, device=device))
    self.register_buffer("pos", torch.zeros((), dtype=torch.long, device=device))

  def k(self, i: int) -> torch.Tensor:
    return getattr(self, f"k{i}")

  def v(self, i: int) -> torch.Tensor:
    return getattr(self, f"v{i}")

  def m(self, i: int) -> torch.Tensor:
    return getattr(self, f"m{i}")

  def tail(self, i: int):
    return getattr(self, f"kt{i}", None)

  def reset(self, pos: int = 0):
    self.pos.fill_(pos)

  def store_prefill(self, i: int, k: torch.Tensor, v: torch.Tensor):
    """Store prefill keys `k` `[B, N, T, D_qk]` and values `v` `[B, N, T, D_v]` into layer `i`."""
    t = k.shape[2]
    if self.layout == "kt":
      kb = kt_from_rows(k)
      self.k(i)[:, :, :kb.shape[2]] = kb
      blk = self.tail(i).shape[3]
      last = (t - 1) // blk
      for j in range(max(last - 1, 0), last + 1):
        piece = k[:, :, j * blk : (j + 1) * blk]
        self.tail(i)[:, :, j % 2, :piece.shape[2]] = piece
    else:
      self.k(i)[:, :, :t] = k
    self.v(i)[:, :, :t] = v

  def load_rows(self, src: "State", row: int):
    for name, buf in self.named_buffers():
      if name != "pos":
        buf[row : row + src.batch].copy_(getattr(src, name))
    self.pos.copy_(src.pos)


class PaddedState(State):
  """Static cache with row strides padded to multiples of 16 elements (32-byte GPU sectors) and 2-slot M ping-pong."""

  def __init__(self, layers: int, batch: int, heads: int, max_len: int, dqk: int, dv: int, m_shape=None, **kw):
    pad16 = lambda w: -(-w // 16) * 16
    super().__init__(layers, batch, heads, max_len, pad16(dqk), pad16(dv), m_shape and (2, *m_shape), **kw)
    self.dqk, self.dv = dqk, dv

  def m_prev(self, i: int, pos: torch.Tensor) -> torch.Tensor:
    return self.m(i).index_select(1, ((pos + 1) % 2).view(1))[:, 0]

  def m_store(self, i: int, q: torch.Tensor, value: torch.Tensor):
    self.m(i).index_copy_(1, (q % 2).view(1), value[:, None])

  def k(self, i: int) -> torch.Tensor:
    return super().k(i)[..., :self.dqk]

  def v(self, i: int) -> torch.Tensor:
    return super().v(i)[..., :self.dv]
