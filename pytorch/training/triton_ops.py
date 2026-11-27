"""Fused Triton forward & backward autograd kernels for LoomFormer training (Llama backbone).

Reuses the forward Triton primitives from `fast_inference.kernels_llama` (`_write_kernel`,
`_load_m`, `_store_m`, `_vsplit`, `_ksplit`, `_pow2`, `qk_width`) and pairs them
with atomic-free, zero-spill Triton backward kernels and FlashAttention-2-aligned `[B, T, N, QW]`
BSHD layout (`Is_even_K=True`, `Is_even_MN=True`).

Physical memory layout matches `fast_inference/kernels_llama.py`:
  - `M`: `[rows, K * V]` (`rows = B * T`, `K = head_dim >= 16`, `V = d_k`, k-major)
  - `Q_sub, K_sub, V_sub`: `[B, T, N, QW]` (1-step shifted with a zero-padded trailing slot `t=T-1`
    and `QW` rounded to native FlashAttention-2 `HEADDIM` buckets `{64, 96, 128, ...}`)
  - `O_M`: `[rows, N, K]` (per-token local vertical bypass added to attention output)
"""
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import triton
import triton.language as tl

from fast_inference import kernels_llama as kl

_FA2_HEADDIMS = (64, 96, 128, 160, 192, 224, 256)


@dataclass(frozen=True)
class _LoomKernelConfig:
  """Shape-adaptive Triton launch configuration derived from `(N, K, V)` tile footprint.

  Thresholds are calibrated across 160M–7B geometries (`K=64..128, V=24..64, N=12..32`)
  using padded memory-tile elements `m_elems = k_pad * (v0 + v1)` so every model scale
  hits zero-spill, peak-occupancy execution without per-run autotuning overhead.
  """
  v0: int
  v1: int
  k1p: int
  m_elems: int
  use_fb_read: bool
  read_fwd_warps: int
  read_fwd_fb_warps: int
  m_prep_warps: int
  read_bwd_fb_warps: int
  mshift_bwd_warps: int
  write_fwd_warps: int
  write_bwd_warps: int


def _select_kernel_config(n: int, k: int, v: int) -> _LoomKernelConfig:
  """Select optimal Triton kernel path and warp counts for a given `(N, K, V)` geometry."""
  v0, v1 = kl._vsplit(v)
  k1p = kl._ksplit(k)
  k_pad = 64 + (k1p if k > 64 else 0)
  m_elems = k_pad * (v0 + v1)

  # Precomputed-FB read pipeline wins whenever the inline backward cannot use compact SUB=1 (2*N <= 32, V1 == 0)
  use_fb_read = (2 * n > 32 or v1 > 0) and 2 * n <= 64
  # Read forward: huge M (>=6144) needs 4 warps; split-V mid M (wideV, XL) maximizes blocks/SM at 1 warp; else 2 warps
  read_fwd_warps = 4 if m_elems >= 6144 else (1 if v1 > 0 else 2)
  # Read forward from precomputed FB has lower register pressure (no p/m/shift live), so 1 warp wins for 3072..5120 (XL, 2.8B)
  read_fwd_fb_warps = 4 if m_elems >= 6144 else (1 if m_elems >= 3072 else 2)
  # M-prep (R=8 ascending carry): 1 warp zero-spill up to 4608 elems, 2 warps for 2.8B+
  m_prep_warps = 2 if m_elems > 4608 else 1
  # Read backward from FB: zero-spill native registers (no maxnreg cap); single V-tile (V1==0) stays optimal at 2 warps up to 5120 (2.8B)
  read_bwd_fb_warps = 2 if (m_elems <= 3072 or (v1 == 0 and m_elems <= 5120)) else 4
  # MShift+MNorm backward (R=16 descending carry): 4 warps above 4608 elems prevents severe register spilling (2.8B+)
  mshift_bwd_warps = 4 if m_elems > 4608 else 2
  # Write forward & backward: scale warps with M tile footprint and head count
  write_fwd_warps = 4 if m_elems >= 8192 else (2 if m_elems > 3072 or n > 16 else 1)
  write_bwd_warps = 1 if m_elems <= 3072 else 2

  return _LoomKernelConfig(
      v0=v0, v1=v1, k1p=k1p, m_elems=m_elems,
      use_fb_read=use_fb_read,
      read_fwd_warps=read_fwd_warps,
      read_fwd_fb_warps=read_fwd_fb_warps,
      m_prep_warps=m_prep_warps,
      read_bwd_fb_warps=read_bwd_fb_warps,
      mshift_bwd_warps=mshift_bwd_warps,
      write_fwd_warps=write_fwd_warps,
      write_bwd_warps=write_bwd_warps,
  )


def _fa_head_dim(k: int, s: int) -> int:
  """Round `k + s` up to a native FlashAttention-2 `HEADDIM` bucket so `Is_even_K=True`."""
  raw = k + s
  for d in _FA2_HEADDIMS:
    if raw <= d:
      return d
  return kl.qk_width(k, s)


def _unit_inner(t: torch.Tensor) -> torch.Tensor:
  """Ensure tensor has unit inner stride (`stride(-1) == 1`) without copying column slices."""
  return t if t.stride(-1) == 1 else t.contiguous()


def can_use_triton_loom(tensor: torch.Tensor, head_dim: int) -> bool:
  """Check whether the fused Triton Loom kernels can run on this tensor & head geometry."""
  return tensor.is_cuda and tensor.dtype == torch.bfloat16 and head_dim >= 16


# ==============================================================================
# 1. Strictly Causal & Causal SDPA (FlashAttention-2)
# ==============================================================================

def _sdpa_flash_or_cudnn(qt: torch.Tensor, kt: torch.Tensor, vt: torch.Tensor) -> torch.Tensor:
  """Causal SDPA preferring FlashAttention-2 (supports equal `D_qk == D_v == QW`)."""
  if qt.is_cuda and qt.dtype in (torch.bfloat16, torch.float16) and qt.shape[-1] == vt.shape[-1]:
    try:
      with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION]):
        return F.scaled_dot_product_attention(qt, kt, vt, is_causal=True, scale=1.0)
    except RuntimeError:
      pass
  return F.scaled_dot_product_attention(qt, kt, vt, is_causal=True, scale=1.0)


@triton.jit
def _shift_add_fwd_kernel(
    Y, sy_b, sy_t, sy_n,
    O_M, so_r,
    OUT,
    T, N: tl.constexpr, NP: tl.constexpr, K: tl.constexpr, KP: tl.constexpr,
):
  """Compute `OUT[b, t, :, :K] = O_M[b*T+t, :, :K] + (Y[b, t-1, :, :K] if t > 0 else 0)` (1 block per token)."""
  row = tl.program_id(0)
  seq = row // T
  t = row % T
  head = tl.arange(0, NP)
  kk = tl.arange(0, KP)
  mask = (head < N)[:, None] & (kk < K)[None, :]
  om = tl.load(O_M + row * so_r + head[:, None] * K + kk[None, :], mask=mask, other=0.0).to(tl.float32)
  if t > 0:
    y = tl.load(Y + seq * sy_b + (t - 1) * sy_t + head[:, None] * sy_n + kk[None, :], mask=mask, other=0.0).to(tl.float32)
    om = om + y
  tl.store(OUT + row * (N * K) + head[:, None] * K + kk[None, :], om.to(tl.bfloat16), mask=mask)


@triton.jit
def _shift_add_bwd_kernel(
    DOUT, sdo_r,
    DY,
    T, N: tl.constexpr, NP: tl.constexpr, K: tl.constexpr, QW: tl.constexpr, QWP: tl.constexpr,
):
  """Write `DY[b, t, :, :QW]` in contiguous BSHD layout: `DOUT[b*T+t+1, :, :K]` (zero for `t==T-1` or `d>=K`)."""
  row = tl.program_id(0)
  t = row % T
  head = tl.arange(0, NP)
  dd = tl.arange(0, QWP)
  hm = (head < N)[:, None]
  if t + 1 < T:
    val = tl.load(DOUT + (row + 1) * sdo_r + head[:, None] * K + dd[None, :], mask=hm & (dd < K)[None, :], other=0.0)
  else:
    val = tl.zeros((NP, QWP), tl.bfloat16)
  tl.store(DY + row * (N * QW) + head[:, None] * QW + dd[None, :], val, mask=hm & (dd < QW)[None, :])


@torch.library.custom_op("loom_train::strict_attn_shift_add_fwd", mutates_args=())
def _strict_attn_shift_add_fwd_op(
    y_sub: torch.Tensor, o_m: torch.Tensor, batch: int, t: int, n: int, k: int,
) -> torch.Tensor:
  rows = batch * t
  o_m = _unit_inner(o_m.reshape(rows, n * k))
  out = o_m.new_empty((rows, n * k))
  _shift_add_fwd_kernel[(rows,)](
      y_sub, y_sub.stride(0), y_sub.stride(2), y_sub.stride(1),
      o_m, o_m.stride(0), out,
      t, N=n, NP=kl._pow2(n), K=k, KP=kl._pow2(k), num_warps=4,
  )
  return out


@_strict_attn_shift_add_fwd_op.register_fake
def _(y_sub, o_m, batch: int, t: int, n: int, k: int):
  return o_m.new_empty((batch * t, n * k))


@torch.library.custom_op("loom_train::strict_attn_shift_add_bwd", mutates_args=())
def _strict_attn_shift_add_bwd_op(
    dout: torch.Tensor, batch: int, t: int, n: int, k: int, qw: int,
) -> torch.Tensor:
  rows = batch * t
  dout = _unit_inner(dout.reshape(rows, n * k))
  dy_bshd = dout.new_empty((batch, t, n, qw))
  _shift_add_bwd_kernel[(rows,)](
      dout, dout.stride(0), dy_bshd,
      t, N=n, NP=kl._pow2(n), K=k, QW=qw, QWP=kl._pow2(qw), num_warps=4,
  )
  return dy_bshd.transpose(1, 2)


@_strict_attn_shift_add_bwd_op.register_fake
def _(dout, batch: int, t: int, n: int, k: int, qw: int):
  return dout.new_empty((batch, t, n, qw)).transpose(1, 2)


def _strict_attn_shift_add_setup(ctx, inputs, output):
  y_sub, _o_m, batch, t, n, k = inputs
  ctx.meta = (batch, t, n, k, y_sub.shape[-1])


def _strict_attn_shift_add_backward(ctx, dout):
  batch, t, n, k, qw = ctx.meta
  dy_sub = _strict_attn_shift_add_bwd_op(dout, batch, t, n, k, qw)
  return dy_sub, dout.reshape(batch * t, n, k), None, None, None, None


_strict_attn_shift_add_fwd_op.register_autograd(
    _strict_attn_shift_add_backward, setup_context=_strict_attn_shift_add_setup
)


def strict_causal_attention_add_train(
    q_sub: torch.Tensor, k_sub: torch.Tensor, v_sub: torch.Tensor, o_m: torch.Tensor,
    batch: int, t: int, n: int, k: int,
) -> torch.Tensor:
  """Strictly causal (`s < t`, t=0 -> 0) FlashAttention-2 on `[B, T, N, QW]` (dummy tail slot) + `o_m`."""
  if t <= 1:
    return o_m.reshape(batch * t, n * k)
  y_sub = _sdpa_flash_or_cudnn(q_sub.transpose(1, 2), k_sub.transpose(1, 2), v_sub.transpose(1, 2))
  if y_sub.is_cuda and y_sub.dtype == torch.bfloat16 and y_sub.stride(-1) == 1:
    return _strict_attn_shift_add_fwd_op(y_sub, o_m, batch, t, n, k)
  y = F.pad(y_sub[:, :, :-1, :k].transpose(1, 2), (0, 0, 0, 0, 1, 0))
  return (y.reshape(batch * t, n, k) + o_m).reshape(batch * t, n * k)


def causal_attention_train(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  """Standard causal attention (`s <= t`) for Llama baseline (`loom_enabled=False`) via Flash/cuDNN SDPA."""
  d = q.shape[-1]
  pad = -d % 8
  if pad:
    q, k, v = F.pad(q, (0, pad)), F.pad(k, (0, pad)), F.pad(v, (0, pad))
  y = _sdpa_flash_or_cudnn(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
  return y[..., :d] if pad else y


# ==============================================================================
# 2. Loom Write (Forward = `kernels_llama._write_kernel`, Backward = Fused Triton)
# ==============================================================================

@triton.jit
def _write_pair_bwd(
    row, dm00, dm01, dm10, dm11,
    C, s_c, G, s_g, GB, A, s_a,
    DC, s_dc, DG, s_dg, DA, s_da,
    write_eps,
    RMS_C: tl.constexpr, N: tl.constexpr, NP: tl.constexpr,
    K: tl.constexpr, K1P: tl.constexpr, V: tl.constexpr, V0: tl.constexpr, V1: tl.constexpr,
):
  """Exact backward of one MWrite pair: dM_pair = sum_n sigmoid(g_n + gb_n) * c_hat_n (x) rms(a_n)."""
  n = tl.arange(0, NP)
  nm = n < N
  k0 = tl.arange(0, 64)
  km0 = k0 < K

  c0 = tl.load(C + row * s_c + n[:, None] * K + k0[None, :], mask=nm[:, None] & km0[None, :], other=0.0).to(tl.float32)
  if K > 64:
    k1 = 64 + tl.arange(0, K1P)
    km1 = k1 < K
    c1 = tl.load(C + row * s_c + n[:, None] * K + k1[None, :], mask=nm[:, None] & km1[None, :], other=0.0).to(tl.float32)
  sig = tl.sigmoid(
      tl.load(G + row * s_g + n, mask=nm, other=0.0).to(tl.float32)
      + tl.load(GB + n, mask=nm, other=0.0).to(tl.float32)
  )
  if RMS_C:
    ss_c = tl.sum(c0 * c0, 1)
    if K > 64:
      ss_c += tl.sum(c1 * c1, 1)
    inv_c = tl.rsqrt(ss_c / K + write_eps)
    ch0 = c0 * inv_c[:, None]
    if K > 64:
      ch1 = c1 * inv_c[:, None]
  else:
    inv_c = tl.full((NP,), 1.0, tl.float32)
    ch0 = c0
    if K > 64:
      ch1 = c1

  a = tl.arange(0, V0)
  am = a < V
  a0 = tl.load(A + row * s_a + n[:, None] * V + a[None, :], mask=nm[:, None] & am[None, :], other=0.0).to(tl.float32)
  ss_a = tl.sum(a0 * a0, 1)
  if V1 > 0:
    b = V0 + tl.arange(0, V1)
    bm = b < V
    a1 = tl.load(A + row * s_a + n[:, None] * V + b[None, :], mask=nm[:, None] & bm[None, :], other=0.0).to(tl.float32)
    ss_a += tl.sum(a1 * a1, 1)
  inv_a = tl.rsqrt(ss_a / V + write_eps)
  ah0 = a0 * inv_a[:, None]
  ah0_b = ah0.to(tl.bfloat16)

  # d(c_eff) = a_hat @ dM^T -> [NP, K]
  dce0 = tl.dot(ah0_b, tl.trans(dm00))
  if K > 64:
    dce1 = tl.dot(ah0_b, tl.trans(dm10))
  if V1 > 0:
    ah1 = a1 * inv_a[:, None]
    ah1_b = ah1.to(tl.bfloat16)
    dce0 += tl.dot(ah1_b, tl.trans(dm01))
    if K > 64:
      dce1 += tl.dot(ah1_b, tl.trans(dm11))

  # Gate gradient: d_sig = sum_k d(c_eff) * c_hat (compute & store before dah to free dce registers)
  dsig = tl.sum(dce0 * ch0, 1)
  if K > 64:
    dsig += tl.sum(dce1 * ch1, 1)
  dg = dsig * sig * (1.0 - sig)
  tl.store(DG + row * s_dg + n, dg.to(tl.bfloat16), mask=nm)

  # Content gradient: d(c_hat) = d(c_eff) * sig
  dch0 = dce0 * sig[:, None]
  if K > 64:
    dch1 = dce1 * sig[:, None]
  if RMS_C:
    dot_c = tl.sum(dch0 * ch0, 1)
    if K > 64:
      dot_c += tl.sum(dch1 * ch1, 1)
    dot_c = dot_c / K
    dc0 = (dch0 - ch0 * dot_c[:, None]) * inv_c[:, None]
    if K > 64:
      dc1 = (dch1 - ch1 * dot_c[:, None]) * inv_c[:, None]
  else:
    dc0 = dch0
    if K > 64:
      dc1 = dch1
  tl.store(DC + row * s_dc + n[:, None] * K + k0[None, :], dc0.to(tl.bfloat16), mask=nm[:, None] & km0[None, :])
  if K > 64:
    tl.store(DC + row * s_dc + n[:, None] * K + k1[None, :], dc1.to(tl.bfloat16), mask=nm[:, None] & km1[None, :])

  # d(a_hat) = c_eff @ dM -> [NP, V]
  ce0_b = (ch0 * sig[:, None]).to(tl.bfloat16)
  dah0 = tl.dot(ce0_b, dm00)
  if K > 64:
    ce1_b = (ch1 * sig[:, None]).to(tl.bfloat16)
    dah0 += tl.dot(ce1_b, dm10)
  if V1 > 0:
    dah1 = tl.dot(ce0_b, dm01)
    if K > 64:
      dah1 += tl.dot(ce1_b, dm11)

  # Address gradient (RMSNorm backward over V)
  dot_a = tl.sum(dah0 * ah0, 1)
  if V1 > 0:
    dot_a += tl.sum(dah1 * ah1, 1)
  dot_a = dot_a / V
  da0 = (dah0 - ah0 * dot_a[:, None]) * inv_a[:, None]
  tl.store(DA + row * s_da + n[:, None] * V + a[None, :], da0.to(tl.bfloat16), mask=nm[:, None] & am[None, :])
  if V1 > 0:
    da1 = (dah1 - ah1 * dot_a[:, None]) * inv_a[:, None]
    tl.store(DA + row * s_da + n[:, None] * V + b[None, :], da1.to(tl.bfloat16), mask=nm[:, None] & bm[None, :])


@triton.jit
def _write_bwd_kernel(
    DM, s_dm,
    C1, s_c1, G1, s_g1, A1, s_a1,
    C2, s_c2, G2, s_g2, A2, s_a2,
    WB,
    DC1, s_dc1, DG1, s_dg1, DA1, s_da1,
    DC2, s_dc2, DG2, s_dg2, DA2, s_da2,
    write_eps,
    N: tl.constexpr, NP: tl.constexpr, K: tl.constexpr, K1P: tl.constexpr,
    V: tl.constexpr, V0: tl.constexpr, V1: tl.constexpr,
):
  row = tl.program_id(0)
  dm00, dm01, dm10, dm11 = kl._load_m(DM + row * s_dm, K, K1P, V, V0, V1)
  _write_pair_bwd(
      row, dm00, dm01, dm10, dm11,
      C1, s_c1, G1, s_g1, WB, A1, s_a1,
      DC1, s_dc1, DG1, s_dg1, DA1, s_da1,
      write_eps, True, N, NP, K, K1P, V, V0, V1,
  )
  _write_pair_bwd(
      row, dm00, dm01, dm10, dm11,
      C2, s_c2, G2, s_g2, WB + N, A2, s_a2,
      DC2, s_dc2, DG2, s_dg2, DA2, s_da2,
      write_eps, False, N, NP, K, K1P, V, V0, V1,
  )


@torch.library.custom_op("loom_train::loom_write_fwd", mutates_args=())
def _loom_write_fwd_op(
    m_old: torch.Tensor,
    c1: torch.Tensor, g1: torch.Tensor, a1: torch.Tensor,
    c2: torch.Tensor, g2: torch.Tensor, a2: torch.Tensor,
    wb: torch.Tensor, n: int, k: int, v: int, write_eps: float,
) -> torch.Tensor:
  m_old, c1, g1, a1, c2, g2, a2 = (_unit_inner(x) for x in (m_old, c1, g1, a1, c2, g2, a2))
  wb = wb.contiguous()
  rows = m_old.shape[0]
  m_new = m_old.new_empty((rows, k * v))
  cfg = _select_kernel_config(n, k, v)
  kl._write_kernel[(rows,)](
      m_old, m_old.stride(0), m_new,
      c1, c1.stride(0), m_old, 0, 0, g1, g1.stride(0), a1, a1.stride(0),
      c2, c2.stride(0), g2, g2.stride(0), a2, a2.stride(0),
      wb, write_eps, False,
      N=n, NP=kl._pow2(n), K=k, K1P=cfg.k1p, V=v, V0=cfg.v0, V1=cfg.v1,
      num_warps=cfg.write_fwd_warps,
  )
  return m_new


@_loom_write_fwd_op.register_fake
def _(m_old, c1, g1, a1, c2, g2, a2, wb, n: int, k: int, v: int, write_eps: float):
  return m_old.new_empty((m_old.shape[0], k * v))


@torch.library.custom_op("loom_train::loom_write_bwd", mutates_args=())
def _loom_write_bwd_op(
    dm_new: torch.Tensor,
    c1: torch.Tensor, g1: torch.Tensor, a1: torch.Tensor,
    c2: torch.Tensor, g2: torch.Tensor, a2: torch.Tensor,
    wb: torch.Tensor, n: int, k: int, v: int, write_eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  dm_new, c1, g1, a1, c2, g2, a2 = (_unit_inner(x) for x in (dm_new, c1, g1, a1, c2, g2, a2))
  wb = wb.contiguous()
  rows = dm_new.shape[0]
  dc1, dg1, da1 = c1.new_empty(c1.shape), g1.new_empty(g1.shape), a1.new_empty(a1.shape)
  dc2, dg2, da2 = c2.new_empty(c2.shape), g2.new_empty(g2.shape), a2.new_empty(a2.shape)
  cfg = _select_kernel_config(n, k, v)
  _write_bwd_kernel[(rows,)](
      dm_new, dm_new.stride(0),
      c1, c1.stride(0), g1, g1.stride(0), a1, a1.stride(0),
      c2, c2.stride(0), g2, g2.stride(0), a2, a2.stride(0),
      wb,
      dc1, dc1.stride(0), dg1, dg1.stride(0), da1, da1.stride(0),
      dc2, dc2.stride(0), dg2, dg2.stride(0), da2, da2.stride(0),
      write_eps,
      N=n, NP=kl._pow2(n), K=k, K1P=cfg.k1p, V=v, V0=cfg.v0, V1=cfg.v1,
      num_warps=cfg.write_bwd_warps,
      num_stages=3 if cfg.v1 > 0 else 2,
  )
  dwb = torch.cat((dg1.sum(0, dtype=torch.float32), dg2.sum(0, dtype=torch.float32)), dim=0).to(wb.dtype)
  return dc1, dg1, da1, dc2, dg2, da2, dwb


@_loom_write_bwd_op.register_fake
def _(dm_new, c1, g1, a1, c2, g2, a2, wb, n: int, k: int, v: int, write_eps: float):
  return (
      c1.new_empty(c1.shape), g1.new_empty(g1.shape), a1.new_empty(a1.shape),
      c2.new_empty(c2.shape), g2.new_empty(g2.shape), a2.new_empty(a2.shape),
      wb.new_empty(wb.shape),
  )


def _loom_write_setup(ctx, inputs, output):
  _, c1, g1, a1, c2, g2, a2, wb, n, k, v, write_eps = inputs
  ctx.save_for_backward(c1, g1, a1, c2, g2, a2, wb)
  ctx.meta = (n, k, v, write_eps)


def _loom_write_backward(ctx, dm_new):
  c1, g1, a1, c2, g2, a2, wb = ctx.saved_tensors
  n, k, v, write_eps = ctx.meta
  dc1, dg1, da1, dc2, dg2, da2, dwb = _loom_write_bwd_op(
      dm_new, c1, g1, a1, c2, g2, a2, wb, n, k, v, write_eps
  )
  return dm_new, dc1, dg1, da1, dc2, dg2, da2, dwb, None, None, None, None


_loom_write_fwd_op.register_autograd(_loom_write_backward, setup_context=_loom_write_setup)


def loom_write_triton(
    m_old: torch.Tensor,
    c1: torch.Tensor, g1: torch.Tensor, a1: torch.Tensor,
    c2: torch.Tensor, g2: torch.Tensor, a2: torch.Tensor,
    wb: torch.Tensor,
    n: int, k: int, v: int, write_eps: float,
) -> torch.Tensor:
  """Autograd + `torch.compile`-enabled fused Triton MWrite (`M_{l+1} = M_l + dM_attn + dM_ffn`)."""
  return _loom_write_fwd_op(m_old, c1, g1, a1, c2, g2, a2, wb.to(m_old.dtype), n, k, v, write_eps)


# ==============================================================================
# 3. Loom Read (MShift + MNorm + 4-Arm MRead + RoPE, Forward & Zero-Spill Bwd)
# ==============================================================================

_MSHIFT_BWD_ROWS = 16  # consecutive token rows per MShift/MNorm backward program (carry kept in registers)


@triton.jit
def _zero_qw_row(base, rm, NP: tl.constexpr, QW: tl.constexpr):
  """Store zeros across `QW` columns of a dummy tail slot `t = T - 1`."""
  for q0 in tl.static_range(0, QW, 64):
    qi = q0 + tl.arange(0, 64)
    tl.store(base + qi[None, :], tl.zeros((NP, 64), tl.bfloat16), mask=rm & (qi[None, :] < QW))


@triton.jit
def _read_fwd_train_kernel(
    M, s_m, SHIFT, s_sh, STD, s_std, UP, s_up, GATES, s_gt, XB, SKEY, C_QK, C_VO, COS, SIN,
    Q_SUB, K_SUB, V_SUB, O_M,
    T, eps, key_eps, key_scale, static_scale, q_mult,
    N: tl.constexpr, NP: tl.constexpr, K: tl.constexpr, K1P: tl.constexpr,
    V: tl.constexpr, V0: tl.constexpr, V1: tl.constexpr,
    H: tl.constexpr, HP: tl.constexpr, QW: tl.constexpr,
    K_TAIL_P: tl.constexpr, Q_TAIL_P: tl.constexpr,
):
  """Fused MShift + MNorm + 4-arm MRead + RoPE writing FlashAttention-2-aligned `[B, T, N, QW]` & `[rows, N, K]`.

  One program per token row loads `M[t]` and `M[t-1]` once and loops over the 4 arms (`Q, K, V, O_M`).
  When `T > 1`, slot `t = T - 1` of `Q_SUB, K_SUB, V_SUB` is zero-filled so FlashAttention-2 receives
  length `T` (e.g. `4096`, enabling `Is_even_MN=True`) without affecting strictly causal outputs `0..T-2`.
  """
  row = tl.program_id(0)
  seq = row // T
  t = row % T

  pv = 1.0 if t > 0 else 0.0
  pp = M + (row - 1) * s_m if t > 0 else M + row * s_m
  p00, p01, p10, p11 = kl._load_m(pp, K, K1P, V, V0, V1)
  a = tl.arange(0, V0)
  am = a < V
  s0 = (pv * tl.sigmoid(
      tl.load(SHIFT + row * s_sh + a, mask=am, other=0.0).to(tl.float32)
      + tl.load(XB + a, mask=am, other=0.0).to(tl.float32)
  ))[None, :]
  if V1 > 0:
    b = V0 + tl.arange(0, V1)
    bm = b < V
    s1 = (pv * tl.sigmoid(
        tl.load(SHIFT + row * s_sh + b, mask=bm, other=0.0).to(tl.float32)
        + tl.load(XB + b, mask=bm, other=0.0).to(tl.float32)
    ))[None, :]

  m00, m01, m10, m11 = kl._load_m(M + row * s_m, K, K1P, V, V0, V1)
  f00 = m00.to(tl.float32)
  f00 = (f00 + s0 * (p00.to(tl.float32) - f00)).to(tl.bfloat16).to(tl.float32)
  ss = tl.sum(tl.sum(f00 * f00, 1), 0)
  fb00 = f00.to(tl.bfloat16)
  if K > 64:
    f10 = m10.to(tl.float32)
    f10 = (f10 + s0 * (p10.to(tl.float32) - f10)).to(tl.bfloat16).to(tl.float32)
    ss += tl.sum(tl.sum(f10 * f10, 1), 0)
    fb10 = f10.to(tl.bfloat16)
  if V1 > 0:
    f01 = m01.to(tl.float32)
    f01 = (f01 + s1 * (p01.to(tl.float32) - f01)).to(tl.bfloat16).to(tl.float32)
    ss += tl.sum(tl.sum(f01 * f01, 1), 0)
    fb01 = f01.to(tl.bfloat16)
    if K > 64:
      f11 = m11.to(tl.float32)
      f11 = (f11 + s1 * (p11.to(tl.float32) - f11)).to(tl.bfloat16).to(tl.float32)
      ss += tl.sum(tl.sum(f11 * f11, 1), 0)
      fb11 = f11.to(tl.bfloat16)
  inv = tl.rsqrt(ss / (K * V) + eps)

  qk_scale = tl.load(C_QK).to(tl.float32)
  vo_scale = tl.load(C_VO).to(tl.float32)
  GB = XB + V

  k0 = tl.arange(0, 64)
  km0 = k0 < K
  if K > 64:
    k1 = 64 + tl.arange(0, K1P)
    km1 = k1 < K
  if H > 0:
    j = tl.arange(0, HP)
    jm = j < H
    cs = tl.load(COS + t * H + j, mask=jm, other=0.0).to(tl.float32)
    sn = tl.load(SIN + t * H + j, mask=jm, other=0.0).to(tl.float32)

  head = tl.arange(0, NP)
  hm = head < N
  rm = hm[:, None]
  am2 = rm & am[None, :]
  if V1 > 0:
    bm2 = rm & bm[None, :]

  row_q = seq * T + (t - 1)
  row_k = seq * T + t
  row_pad = seq * T + (T - 1)

  for arm_idx in tl.static_range(4):
    cg = arm_idx * N + head
    gd = key_scale * tl.sigmoid(
        tl.load(GATES + row * s_gt + cg, mask=hm, other=0.0).to(tl.float32)
        + tl.load(GB + cg, mask=hm, other=0.0).to(tl.float32)
    )
    gs = static_scale * tl.sigmoid(
        tl.load(GATES + row * s_gt + 4 * N + cg, mask=hm, other=0.0).to(tl.float32)
        + tl.load(GB + 4 * N + cg, mask=hm, other=0.0).to(tl.float32)
    )
    up0 = tl.load(UP + row * s_up + cg[:, None] * V + a[None, :], mask=am2, other=0.0).to(tl.float32)
    ss_up = tl.sum(up0 * up0, 1)
    if V1 > 0:
      up1 = tl.load(UP + row * s_up + cg[:, None] * V + b[None, :], mask=bm2, other=0.0).to(tl.float32)
      ss_up += tl.sum(up1 * up1, 1)
    gk = (gd * tl.rsqrt(ss_up / V + key_eps))[:, None]
    sk0 = tl.load(SKEY + cg[:, None] * V + a[None, :], mask=am2, other=0.0).to(tl.float32)
    r0 = (up0 * gk + gs[:, None] * sk0).to(tl.bfloat16)
    o0 = tl.dot(r0, tl.trans(fb00))
    if K > 64:
      o1 = tl.dot(r0, tl.trans(fb10))
    if V1 > 0:
      sk1 = tl.load(SKEY + cg[:, None] * V + b[None, :], mask=bm2, other=0.0).to(tl.float32)
      r1 = (up1 * gk + gs[:, None] * sk1).to(tl.bfloat16)
      o0 += tl.dot(r1, tl.trans(fb01))
      if K > 64:
        o1 += tl.dot(r1, tl.trans(fb11))

    if arm_idx < 2:
      qm = q_mult if arm_idx == 0 else 1.0
      sq = inv * qk_scale * qm
      y0 = (o0 * sq).to(tl.bfloat16)
      if K > 64:
        y1 = (o1 * sq).to(tl.bfloat16)
      if H > 0:
        sa = tl.load(STD + row * s_std + cg[:, None] * (2 * H) + j[None, :], mask=rm & jm[None, :], other=0.0).to(tl.float32)
        sb = tl.load(STD + row * s_std + cg[:, None] * (2 * H) + H + j[None, :], mask=rm & jm[None, :], other=0.0).to(tl.float32)
        ra = ((sa * cs[None, :] - sb * sn[None, :]) * qm).to(tl.bfloat16)
        rb = ((sb * cs[None, :] + sa * sn[None, :]) * qm).to(tl.bfloat16)

      if arm_idx == 0:
        if t > 0:
          base = Q_SUB + row_q * (N * QW) + head[:, None] * QW
          tl.store(base + k0[None, :], y0, mask=rm & km0[None, :])
          if K > 64:
            tl.store(base + k1[None, :], y1, mask=rm & km1[None, :])
          if H > 0:
            tl.store(base + K + j[None, :], ra, mask=rm & jm[None, :])
            tl.store(base + K + H + j[None, :], rb, mask=rm & jm[None, :])
          if QW > K + 2 * H:
            zp = tl.arange(0, Q_TAIL_P)
            tl.store(base + K + 2 * H + zp[None, :], tl.zeros((NP, Q_TAIL_P), tl.bfloat16),
                     mask=rm & (zp[None, :] < QW - K - 2 * H))
        elif T > 1:
          _zero_qw_row(Q_SUB + row_pad * (N * QW) + head[:, None] * QW, rm, NP, QW)
      else:
        if t + 1 < T:
          base = K_SUB + row_k * (N * QW) + head[:, None] * QW
          tl.store(base + k0[None, :], y0, mask=rm & km0[None, :])
          if K > 64:
            tl.store(base + k1[None, :], y1, mask=rm & km1[None, :])
          if H > 0:
            tl.store(base + K + j[None, :], ra, mask=rm & jm[None, :])
            tl.store(base + K + H + j[None, :], rb, mask=rm & jm[None, :])
          if QW > K + 2 * H:
            zp = tl.arange(0, Q_TAIL_P)
            tl.store(base + K + 2 * H + zp[None, :], tl.zeros((NP, Q_TAIL_P), tl.bfloat16),
                     mask=rm & (zp[None, :] < QW - K - 2 * H))
        elif T > 1:
          _zero_qw_row(K_SUB + row_pad * (N * QW) + head[:, None] * QW, rm, NP, QW)
    else:
      sv = inv * vo_scale
      y0 = (o0 * sv).to(tl.bfloat16)
      if K > 64:
        y1 = (o1 * sv).to(tl.bfloat16)
      if arm_idx == 2:
        if t + 1 < T:
          base = V_SUB + row_k * (N * QW) + head[:, None] * QW
          tl.store(base + k0[None, :], y0, mask=rm & km0[None, :])
          if K > 64:
            tl.store(base + k1[None, :], y1, mask=rm & km1[None, :])
          if QW > K:
            zt = tl.arange(0, K_TAIL_P)
            tl.store(base + K + zt[None, :], tl.zeros((NP, K_TAIL_P), tl.bfloat16),
                     mask=rm & (zt[None, :] < QW - K))
        elif T > 1:
          _zero_qw_row(V_SUB + row_pad * (N * QW) + head[:, None] * QW, rm, NP, QW)
      else:
        base = O_M + row * (N * K) + head[:, None] * K
        tl.store(base + k0[None, :], y0, mask=rm & km0[None, :])
        if K > 64:
          tl.store(base + k1[None, :], y1, mask=rm & km1[None, :])


@triton.jit
def _read_bwd_group_kernel(
    M, s_m, SHIFT, s_sh, UP, s_up, GATES, s_gt, XB, SKEY, C_QK, C_VO, COS, SIN,
    DQ_IN, sdq_b, sdq_t, sdq_n,
    DK_IN, sdk_b, sdk_t, sdk_n,
    DV_IN, sdv_b, sdv_t, sdv_n,
    DO_M,
    DMT_G, s_dmt, S_BUF, s_sb,
    DSTD, s_dstd, DUP, s_dup, DGATES, s_dgt,
    DSK_ROWS, s_dskr, DC_ROWS,
    rows, T, eps, key_eps, key_scale, static_scale, q_mult,
    SUB: tl.constexpr, N: tl.constexpr, K: tl.constexpr, K1P: tl.constexpr,
    V: tl.constexpr, V0: tl.constexpr, V1: tl.constexpr,
    H: tl.constexpr, HP: tl.constexpr, CR: tl.constexpr, CRP: tl.constexpr,
):
  """Zero-spill split-group read backward: grid=(rows, 2) for (Q/K) and (V/O_M)."""
  row = tl.program_id(0)
  group = tl.program_id(1)
  seq = row // T
  t = row % T

  a = tl.arange(0, V0)
  am = a < V
  k0 = tl.arange(0, 64)
  km0 = k0 < K
  if K > 64:
    k1 = 64 + tl.arange(0, K1P)
    km1 = k1 < K
  if V1 > 0:
    b = V0 + tl.arange(0, V1)
    bm = b < V

  # Recompute M_tilde tiles (fb, inv) and cache shift gate in S_BUF once (group == 0)
  pv = 1.0 if t > 0 else 0.0
  pp = M + (row - 1) * s_m if t > 0 else M + row * s_m
  p00, p01, p10, p11 = kl._load_m(pp, K, K1P, V, V0, V1)
  m00, m01, m10, m11 = kl._load_m(M + row * s_m, K, K1P, V, V0, V1)

  s0_1d = pv * tl.sigmoid(
      tl.load(SHIFT + row * s_sh + a, mask=am, other=0.0).to(tl.float32)
      + tl.load(XB + a, mask=am, other=0.0).to(tl.float32)
  )
  if group == 0:
    tl.store(S_BUF + row * s_sb + a, s0_1d, mask=am)
  s0 = s0_1d[None, :]
  if V1 > 0:
    s1_1d = pv * tl.sigmoid(
        tl.load(SHIFT + row * s_sh + b, mask=bm, other=0.0).to(tl.float32)
        + tl.load(XB + b, mask=bm, other=0.0).to(tl.float32)
    )
    if group == 0:
      tl.store(S_BUF + row * s_sb + b, s1_1d, mask=bm)
    s1 = s1_1d[None, :]

  m00_f = m00.to(tl.float32)
  fb00 = (m00_f + s0 * (p00.to(tl.float32) - m00_f)).to(tl.bfloat16)
  f00_r = fb00.to(tl.float32)
  ss = tl.sum(tl.sum(f00_r * f00_r, 1), 0)
  if K > 64:
    m10_f = m10.to(tl.float32)
    fb10 = (m10_f + s0 * (p10.to(tl.float32) - m10_f)).to(tl.bfloat16)
    f10_r = fb10.to(tl.float32)
    ss += tl.sum(tl.sum(f10_r * f10_r, 1), 0)
  else:
    fb10 = tl.zeros((K1P, V0), tl.bfloat16)
  if V1 > 0:
    m01_f = m01.to(tl.float32)
    fb01 = (m01_f + s1 * (p01.to(tl.float32) - m01_f)).to(tl.bfloat16)
    f01_r = fb01.to(tl.float32)
    ss += tl.sum(tl.sum(f01_r * f01_r, 1), 0)
    if K > 64:
      m11_f = m11.to(tl.float32)
      fb11 = (m11_f + s1 * (p11.to(tl.float32) - m11_f)).to(tl.bfloat16)
      f11_r = fb11.to(tl.float32)
      ss += tl.sum(tl.sum(f11_r * f11_r, 1), 0)
    else:
      fb11 = tl.zeros((K1P, V1), tl.bfloat16)
  else:
    fb01 = tl.zeros((64, 16), tl.bfloat16)
    fb11 = tl.zeros((K1P, 16), tl.bfloat16)
  inv = tl.rsqrt(ss / (K * V) + eps)

  arm_scale = tl.load(C_QK).to(tl.float32) if group == 0 else tl.load(C_VO).to(tl.float32)
  GB = XB + V
  ci = tl.arange(0, CRP)

  dmt00 = tl.zeros((64, V0), tl.float32)
  dmt10 = tl.zeros((K1P, V0), tl.float32)
  if V1 > 0:
    dmt01 = tl.zeros((64, V1), tl.float32)
    dmt11 = tl.zeros((K1P, V1), tl.float32)
  else:
    dmt01 = tl.zeros((64, 16), tl.float32)
    dmt11 = tl.zeros((K1P, 16), tl.float32)
  dc_row = 0.0

  for sub_idx in range(0, SUB):
    c = sub_idx * CR + ci
    cm = (ci < CR) & (c < 2 * N)
    rm = cm[:, None]
    cg = group * 2 * N + c
    am2 = rm & am[None, :]
    arm = c // N
    head = c % N

    sig_d = tl.sigmoid(
        tl.load(GATES + row * s_gt + cg, mask=cm, other=0.0).to(tl.float32)
        + tl.load(GB + cg, mask=cm, other=0.0).to(tl.float32)
    )
    sig_s = tl.sigmoid(
        tl.load(GATES + row * s_gt + 4 * N + cg, mask=cm, other=0.0).to(tl.float32)
        + tl.load(GB + 4 * N + cg, mask=cm, other=0.0).to(tl.float32)
    )
    gd = key_scale * sig_d
    gs = static_scale * sig_s

    up0 = tl.load(UP + row * s_up + cg[:, None] * V + a[None, :], mask=am2, other=0.0).to(tl.float32)
    ss_up = tl.sum(up0 * up0, 1)
    if V1 > 0:
      bm2 = rm & bm[None, :]
      up1 = tl.load(UP + row * s_up + cg[:, None] * V + b[None, :], mask=bm2, other=0.0).to(tl.float32)
      ss_up += tl.sum(up1 * up1, 1)
    inv_up = tl.rsqrt(ss_up / V + key_eps)
    uh0 = up0 * inv_up[:, None]
    sk0 = tl.load(SKEY + cg[:, None] * V + a[None, :], mask=am2, other=0.0).to(tl.float32)
    r0 = (uh0 * gd[:, None] + sk0 * gs[:, None]).to(tl.bfloat16)
    if V1 > 0:
      uh1 = up1 * inv_up[:, None]
      sk1 = tl.load(SKEY + cg[:, None] * V + b[None, :], mask=bm2, other=0.0).to(tl.float32)
      r1 = (uh1 * gd[:, None] + sk1 * gs[:, None]).to(tl.bfloat16)

    if group == 0:
      is_q = arm == 0
      qm = tl.where(is_q, q_mult, 1.0)
      qb = DQ_IN + seq * sdq_b + (t - 1) * sdq_t + head[:, None] * sdq_n
      kb = DK_IN + seq * sdk_b + t * sdk_t + head[:, None] * sdk_n
      vm = (cm & ((is_q & (t > 0)) | ((~is_q) & (t + 1 < T))))[:, None]
      base = tl.where(is_q[:, None], qb, kb)

      do0_b = (tl.load(base + k0[None, :], mask=vm & km0[None, :], other=0.0).to(tl.float32) * qm[:, None]).to(tl.bfloat16)
      if K > 64:
        do1_b = (tl.load(base + k1[None, :], mask=vm & km1[None, :], other=0.0).to(tl.float32) * qm[:, None]).to(tl.bfloat16)

      # RoPE backward on the standard [Q_r, K_r] tail
      j = tl.arange(0, HP)
      jm = j < H
      dra = tl.load(base + K + j[None, :], mask=vm & jm[None, :], other=0.0).to(tl.float32) * qm[:, None]
      drb = tl.load(base + K + H + j[None, :], mask=vm & jm[None, :], other=0.0).to(tl.float32) * qm[:, None]
      cs = tl.load(COS + t * H + j, mask=jm, other=0.0).to(tl.float32)
      sn = tl.load(SIN + t * H + j, mask=jm, other=0.0).to(tl.float32)
      dsa = (dra * cs[None, :] + drb * sn[None, :]).to(tl.bfloat16)
      dsb = (drb * cs[None, :] - dra * sn[None, :]).to(tl.bfloat16)
      tl.store(DSTD + row * s_dstd + c[:, None] * (2 * H) + j[None, :], dsa, mask=rm & jm[None, :])
      tl.store(DSTD + row * s_dstd + c[:, None] * (2 * H) + H + j[None, :], dsb, mask=rm & jm[None, :])
    else:
      is_v = arm == 0
      vb = DV_IN + seq * sdv_b + t * sdv_t + head[:, None] * sdv_n
      ob = DO_M + row * (N * K) + head[:, None] * K
      vm = (cm & ((is_v & (t + 1 < T)) | (~is_v)))[:, None]
      base = tl.where(is_v[:, None], vb, ob)
      do0_b = tl.load(base + k0[None, :], mask=vm & km0[None, :], other=0.0)
      if K > 64:
        do1_b = tl.load(base + k1[None, :], mask=vm & km1[None, :], other=0.0)

    # Tensor-core contractions (bf16 operands -> fp32 accumulators)
    dr_un0 = tl.dot(do0_b, fb00)
    if K > 64:
      dr_un0 += tl.dot(do1_b, fb10)
    dr_un0 = dr_un0 * inv
    dc_row += tl.sum(tl.sum(dr_un0 * r0.to(tl.float32), 1), 0)
    dr0 = dr_un0 * arm_scale

    dmt00 += tl.dot(tl.trans(do0_b), r0) * arm_scale
    if K > 64:
      dmt10 += tl.dot(tl.trans(do1_b), r0) * arm_scale

    if V1 > 0:
      dr_un1 = tl.dot(do0_b, fb01)
      if K > 64:
        dr_un1 += tl.dot(do1_b, fb11)
      dr_un1 = dr_un1 * inv
      dc_row += tl.sum(tl.sum(dr_un1 * r1.to(tl.float32), 1), 0)
      dr1 = dr_un1 * arm_scale
      dmt01 += tl.dot(tl.trans(do0_b), r1) * arm_scale
      if K > 64:
        dmt11 += tl.dot(tl.trans(do1_b), r1) * arm_scale

    # Gate, Up, and Static-Key per-row gradients
    dgd = tl.sum(dr0 * uh0, 1)
    dgs = tl.sum(dr0 * sk0, 1)
    if V1 > 0:
      dgd += tl.sum(dr1 * uh1, 1)
      dgs += tl.sum(dr1 * sk1, 1)
    tl.store(DGATES + row * s_dgt + cg, (dgd * gd * (1.0 - sig_d)).to(tl.bfloat16), mask=cm)
    tl.store(DGATES + row * s_dgt + 4 * N + cg, (dgs * gs * (1.0 - sig_s)).to(tl.bfloat16), mask=cm)

    duh0 = dr0 * gd[:, None]
    dot_u = tl.sum(duh0 * uh0, 1)
    if V1 > 0:
      duh1 = dr1 * gd[:, None]
      dot_u += tl.sum(duh1 * uh1, 1)
    dot_u = dot_u / V
    tl.store(DUP + row * s_dup + cg[:, None] * V + a[None, :],
             ((duh0 - uh0 * dot_u[:, None]) * inv_up[:, None]).to(tl.bfloat16), mask=am2)
    tl.store(DSK_ROWS + row * s_dskr + cg[:, None] * V + a[None, :],
             (dr0 * gs[:, None]).to(tl.bfloat16), mask=am2)
    if V1 > 0:
      tl.store(DUP + row * s_dup + cg[:, None] * V + b[None, :],
               ((duh1 - uh1 * dot_u[:, None]) * inv_up[:, None]).to(tl.bfloat16), mask=bm2)
      tl.store(DSK_ROWS + row * s_dskr + cg[:, None] * V + b[None, :],
               (dr1 * gs[:, None]).to(tl.bfloat16), mask=bm2)

  kl._store_m(
      DMT_G + (group * rows + row) * s_dmt,
      dmt00.to(tl.bfloat16), dmt01.to(tl.bfloat16), dmt10.to(tl.bfloat16), dmt11.to(tl.bfloat16),
      K, K1P, V, V0, V1,
  )
  tl.store(DC_ROWS + group * rows + row, dc_row)


@triton.jit
def _mshift_mnorm_bwd_kernel(
    M, s_m, DMT_G, s_dmt, S_BUF, s_sb,
    DM, s_dm, DSHIFT, s_dsh,
    rows, T, CH, eps,
    R: tl.constexpr,
    K: tl.constexpr, K1P: tl.constexpr, V: tl.constexpr, V0: tl.constexpr, V1: tl.constexpr,
):
  """Fused MNorm backward + MShift backward walking `R` rows in descending time (carry kept in registers)."""
  pid = tl.program_id(0)
  seq = pid // CH
  t_lo = (pid % CH) * R
  t_start = tl.minimum(t_lo + R, T - 1)

  a = tl.arange(0, V0)
  am = a < V
  if V1 > 0:
    b = V0 + tl.arange(0, V1)
    bm = b < V

  cb00 = tl.zeros((64, V0), tl.bfloat16)
  cb10 = tl.zeros((K1P, V0), tl.bfloat16)
  sn0 = tl.zeros((V0,), tl.float32)
  if V1 > 0:
    cb01 = tl.zeros((64, V1), tl.bfloat16)
    cb11 = tl.zeros((K1P, V1), tl.bfloat16)
    sn1 = tl.zeros((V1,), tl.float32)
  else:
    cb01 = tl.zeros((64, 16), tl.bfloat16)
    cb11 = tl.zeros((K1P, 16), tl.bfloat16)
    sn1 = tl.zeros((16,), tl.float32)

  for i in range(0, R + 1):
    t = t_lo + R - i
    if t <= t_start:
      row = seq * T + t
      pp = M + (row - 1) * s_m if t > 0 else M + row * s_m
      p00, p01, p10, p11 = kl._load_m(pp, K, K1P, V, V0, V1)
      m00, m01, m10, m11 = kl._load_m(M + row * s_m, K, K1P, V, V0, V1)
      g00, g01, g10, g11 = kl._load_m(DMT_G + row * s_dmt, K, K1P, V, V0, V1)
      h00, h01, h10, h11 = kl._load_m(DMT_G + (rows + row) * s_dmt, K, K1P, V, V0, V1)

      s0_1d = tl.load(S_BUF + row * s_sb + a, mask=am, other=0.0)
      s0 = s0_1d[None, :]
      m00_f = m00.to(tl.float32)
      pm00 = p00.to(tl.float32) - m00_f
      f00 = (m00_f + s0 * pm00).to(tl.bfloat16).to(tl.float32)
      ss = tl.sum(tl.sum(f00 * f00, 1), 0)
      dmt00 = g00.to(tl.float32) + h00.to(tl.float32)
      dot_m = tl.sum(tl.sum(dmt00 * f00, 1), 0)
      if K > 64:
        m10_f = m10.to(tl.float32)
        pm10 = p10.to(tl.float32) - m10_f
        f10 = (m10_f + s0 * pm10).to(tl.bfloat16).to(tl.float32)
        ss += tl.sum(tl.sum(f10 * f10, 1), 0)
        dmt10 = g10.to(tl.float32) + h10.to(tl.float32)
        dot_m += tl.sum(tl.sum(dmt10 * f10, 1), 0)
      if V1 > 0:
        s1_1d = tl.load(S_BUF + row * s_sb + b, mask=bm, other=0.0)
        s1 = s1_1d[None, :]
        m01_f = m01.to(tl.float32)
        pm01 = p01.to(tl.float32) - m01_f
        f01 = (m01_f + s1 * pm01).to(tl.bfloat16).to(tl.float32)
        ss += tl.sum(tl.sum(f01 * f01, 1), 0)
        dmt01 = g01.to(tl.float32) + h01.to(tl.float32)
        dot_m += tl.sum(tl.sum(dmt01 * f01, 1), 0)
        if K > 64:
          m11_f = m11.to(tl.float32)
          pm11 = p11.to(tl.float32) - m11_f
          f11 = (m11_f + s1 * pm11).to(tl.bfloat16).to(tl.float32)
          ss += tl.sum(tl.sum(f11 * f11, 1), 0)
          dmt11 = g11.to(tl.float32) + h11.to(tl.float32)
          dot_m += tl.sum(tl.sum(dmt11 * f11, 1), 0)
      inv = tl.rsqrt(ss / (K * V) + eps)
      coeff = (dot_m / (K * V)) * (inv * inv)

      db00 = ((dmt00 - f00 * coeff) * inv).to(tl.bfloat16)
      db10 = ((dmt10 - f10 * coeff) * inv).to(tl.bfloat16) if K > 64 else tl.zeros((K1P, V0), tl.bfloat16)
      if V1 > 0:
        db01 = ((dmt01 - f01 * coeff) * inv).to(tl.bfloat16)
        db11 = ((dmt11 - f11 * coeff) * inv).to(tl.bfloat16) if K > 64 else tl.zeros((K1P, V1), tl.bfloat16)
      else:
        db01 = tl.zeros((64, 16), tl.bfloat16)
        db11 = tl.zeros((K1P, 16), tl.bfloat16)

      if i > 0:
        d00_f = db00.to(tl.float32)
        ob00 = (d00_f * (1.0 - s0) + cb00.to(tl.float32) * sn0[None, :]).to(tl.bfloat16)
        ds0 = tl.sum(d00_f * pm00, 0)
        if K > 64:
          d10_f = db10.to(tl.float32)
          ob10 = (d10_f * (1.0 - s0) + cb10.to(tl.float32) * sn0[None, :]).to(tl.bfloat16)
          ds0 += tl.sum(d10_f * pm10, 0)
        else:
          ob10 = tl.zeros((K1P, V0), tl.bfloat16)
        tl.store(DSHIFT + row * s_dsh + a, (ds0 * s0_1d * (1.0 - s0_1d)).to(tl.bfloat16), mask=am)
        if V1 > 0:
          d01_f = db01.to(tl.float32)
          ob01 = (d01_f * (1.0 - s1) + cb01.to(tl.float32) * sn1[None, :]).to(tl.bfloat16)
          ds1 = tl.sum(d01_f * pm01, 0)
          if K > 64:
            d11_f = db11.to(tl.float32)
            ob11 = (d11_f * (1.0 - s1) + cb11.to(tl.float32) * sn1[None, :]).to(tl.bfloat16)
            ds1 += tl.sum(d11_f * pm11, 0)
          else:
            ob11 = tl.zeros((K1P, V1), tl.bfloat16)
          tl.store(DSHIFT + row * s_dsh + b, (ds1 * s1_1d * (1.0 - s1_1d)).to(tl.bfloat16), mask=bm)
        else:
          ob01 = tl.zeros((64, 16), tl.bfloat16)
          ob11 = tl.zeros((K1P, 16), tl.bfloat16)
        kl._store_m(DM + row * s_dm, ob00, ob01, ob10, ob11, K, K1P, V, V0, V1)

      cb00, cb01, cb10, cb11 = db00, db01, db10, db11
      sn0 = s0_1d
      if V1 > 0:
        sn1 = s1_1d


@triton.jit
def _read_m_prep_kernel(
    M, s_m, SHIFT, s_sh, XB,
    FB, s_fb, S_BUF, s_sb, INV_BUF,
    T, CH, eps,
    R: tl.constexpr,
    K: tl.constexpr, K1P: tl.constexpr, V: tl.constexpr, V0: tl.constexpr, V1: tl.constexpr,
):
  """Single-pass MShift + MNorm prep walking `R` ascending rows (carries M[t-1] in registers -> 1x M read)."""
  pid = tl.program_id(0)
  seq = pid // CH
  t_lo = (pid % CH) * R

  a = tl.arange(0, V0)
  am = a < V
  xb0 = tl.load(XB + a, mask=am, other=0.0).to(tl.float32)
  if V1 > 0:
    b = V0 + tl.arange(0, V1)
    bm = b < V
    xb1 = tl.load(XB + b, mask=bm, other=0.0).to(tl.float32)

  r0 = seq * T + (t_lo - 1 if t_lo > 0 else 0)
  p00, p01, p10, p11 = kl._load_m(M + r0 * s_m, K, K1P, V, V0, V1)

  for i in range(0, R):
    t = t_lo + i
    if t < T:
      row = seq * T + t
      m00, m01, m10, m11 = kl._load_m(M + row * s_m, K, K1P, V, V0, V1)
      pv = 1.0 if t > 0 else 0.0
      s0_1d = pv * tl.sigmoid(tl.load(SHIFT + row * s_sh + a, mask=am, other=0.0).to(tl.float32) + xb0)
      tl.store(S_BUF + row * s_sb + a, s0_1d, mask=am)
      s0 = s0_1d[None, :]
      m00_f = m00.to(tl.float32)
      fb00 = (m00_f + s0 * (p00.to(tl.float32) - m00_f)).to(tl.bfloat16)
      f00 = fb00.to(tl.float32)
      ss = tl.sum(tl.sum(f00 * f00, 1), 0)
      if K > 64:
        m10_f = m10.to(tl.float32)
        fb10 = (m10_f + s0 * (p10.to(tl.float32) - m10_f)).to(tl.bfloat16)
        f10 = fb10.to(tl.float32)
        ss += tl.sum(tl.sum(f10 * f10, 1), 0)
      else:
        fb10 = tl.zeros((K1P, V0), tl.bfloat16)
      if V1 > 0:
        s1_1d = pv * tl.sigmoid(tl.load(SHIFT + row * s_sh + b, mask=bm, other=0.0).to(tl.float32) + xb1)
        tl.store(S_BUF + row * s_sb + b, s1_1d, mask=bm)
        s1 = s1_1d[None, :]
        m01_f = m01.to(tl.float32)
        fb01 = (m01_f + s1 * (p01.to(tl.float32) - m01_f)).to(tl.bfloat16)
        f01 = fb01.to(tl.float32)
        ss += tl.sum(tl.sum(f01 * f01, 1), 0)
        if K > 64:
          m11_f = m11.to(tl.float32)
          fb11 = (m11_f + s1 * (p11.to(tl.float32) - m11_f)).to(tl.bfloat16)
          f11 = fb11.to(tl.float32)
          ss += tl.sum(tl.sum(f11 * f11, 1), 0)
        else:
          fb11 = tl.zeros((K1P, V1), tl.bfloat16)
      else:
        fb01 = tl.zeros((64, 16), tl.bfloat16)
        fb11 = tl.zeros((K1P, 16), tl.bfloat16)
      tl.store(INV_BUF + row, tl.rsqrt(ss / (K * V) + eps))
      kl._store_m(FB + row * s_fb, fb00, fb01, fb10, fb11, K, K1P, V, V0, V1)
      p00, p01, p10, p11 = m00, m01, m10, m11


@triton.jit
def _read_fwd_from_fb_kernel(
    FB, s_fb, INV_BUF, STD, s_std, UP, s_up, GATES, s_gt, XB, SKEY, C_QK, C_VO, COS, SIN,
    Q_SUB, K_SUB, V_SUB, O_M,
    T, key_eps, key_scale, static_scale, q_mult,
    N: tl.constexpr, NP: tl.constexpr, K: tl.constexpr, K1P: tl.constexpr,
    V: tl.constexpr, V0: tl.constexpr, V1: tl.constexpr,
    H: tl.constexpr, HP: tl.constexpr, QW: tl.constexpr,
    K_TAIL_P: tl.constexpr, Q_TAIL_P: tl.constexpr,
):
  """Zero-spill 1-warp read forward consuming precomputed `(FB, INV_BUF)` for multi-tile `M`."""
  row = tl.program_id(0)
  seq = row // T
  t = row % T

  a = tl.arange(0, V0)
  am = a < V
  if V1 > 0:
    b = V0 + tl.arange(0, V1)
    bm = b < V

  fb00, fb01, fb10, fb11 = kl._load_m(FB + row * s_fb, K, K1P, V, V0, V1)
  inv = tl.load(INV_BUF + row)
  qk_scale = tl.load(C_QK).to(tl.float32)
  vo_scale = tl.load(C_VO).to(tl.float32)
  GB = XB + V

  k0 = tl.arange(0, 64)
  km0 = k0 < K
  if K > 64:
    k1 = 64 + tl.arange(0, K1P)
    km1 = k1 < K
  if H > 0:
    j = tl.arange(0, HP)
    jm = j < H
    cs = tl.load(COS + t * H + j, mask=jm, other=0.0).to(tl.float32)
    sn = tl.load(SIN + t * H + j, mask=jm, other=0.0).to(tl.float32)

  head = tl.arange(0, NP)
  hm = head < N
  rm = hm[:, None]
  am2 = rm & am[None, :]
  if V1 > 0:
    bm2 = rm & bm[None, :]

  row_q = seq * T + (t - 1)
  row_k = seq * T + t
  row_pad = seq * T + (T - 1)

  for arm_idx in tl.static_range(4):
    cg = arm_idx * N + head
    gd = key_scale * tl.sigmoid(
        tl.load(GATES + row * s_gt + cg, mask=hm, other=0.0).to(tl.float32)
        + tl.load(GB + cg, mask=hm, other=0.0).to(tl.float32)
    )
    gs = static_scale * tl.sigmoid(
        tl.load(GATES + row * s_gt + 4 * N + cg, mask=hm, other=0.0).to(tl.float32)
        + tl.load(GB + 4 * N + cg, mask=hm, other=0.0).to(tl.float32)
    )
    up0 = tl.load(UP + row * s_up + cg[:, None] * V + a[None, :], mask=am2, other=0.0).to(tl.float32)
    ss_up = tl.sum(up0 * up0, 1)
    if V1 > 0:
      up1 = tl.load(UP + row * s_up + cg[:, None] * V + b[None, :], mask=bm2, other=0.0).to(tl.float32)
      ss_up += tl.sum(up1 * up1, 1)
    gk = (gd * tl.rsqrt(ss_up / V + key_eps))[:, None]
    sk0 = tl.load(SKEY + cg[:, None] * V + a[None, :], mask=am2, other=0.0).to(tl.float32)
    r0 = (up0 * gk + gs[:, None] * sk0).to(tl.bfloat16)
    o0 = tl.dot(r0, tl.trans(fb00))
    if K > 64:
      o1 = tl.dot(r0, tl.trans(fb10))
    if V1 > 0:
      sk1 = tl.load(SKEY + cg[:, None] * V + b[None, :], mask=bm2, other=0.0).to(tl.float32)
      r1 = (up1 * gk + gs[:, None] * sk1).to(tl.bfloat16)
      o0 += tl.dot(r1, tl.trans(fb01))
      if K > 64:
        o1 += tl.dot(r1, tl.trans(fb11))

    if arm_idx < 2:
      qm = q_mult if arm_idx == 0 else 1.0
      sq = inv * qk_scale * qm
      y0 = (o0 * sq).to(tl.bfloat16)
      if K > 64:
        y1 = (o1 * sq).to(tl.bfloat16)
      if H > 0:
        sa = tl.load(STD + row * s_std + cg[:, None] * (2 * H) + j[None, :], mask=rm & jm[None, :], other=0.0).to(tl.float32)
        sb = tl.load(STD + row * s_std + cg[:, None] * (2 * H) + H + j[None, :], mask=rm & jm[None, :], other=0.0).to(tl.float32)
        ra = ((sa * cs[None, :] - sb * sn[None, :]) * qm).to(tl.bfloat16)
        rb = ((sb * cs[None, :] + sa * sn[None, :]) * qm).to(tl.bfloat16)
      if arm_idx == 0:
        if t > 0:
          base = Q_SUB + row_q * (N * QW) + head[:, None] * QW
          tl.store(base + k0[None, :], y0, mask=rm & km0[None, :])
          if K > 64:
            tl.store(base + k1[None, :], y1, mask=rm & km1[None, :])
          if H > 0:
            tl.store(base + K + j[None, :], ra, mask=rm & jm[None, :])
            tl.store(base + K + H + j[None, :], rb, mask=rm & jm[None, :])
          if QW > K + 2 * H:
            zp = tl.arange(0, Q_TAIL_P)
            tl.store(base + K + 2 * H + zp[None, :], tl.zeros((NP, Q_TAIL_P), tl.bfloat16),
                     mask=rm & (zp[None, :] < QW - K - 2 * H))
        elif T > 1:
          _zero_qw_row(Q_SUB + row_pad * (N * QW) + head[:, None] * QW, rm, NP, QW)
      else:
        if t + 1 < T:
          base = K_SUB + row_k * (N * QW) + head[:, None] * QW
          tl.store(base + k0[None, :], y0, mask=rm & km0[None, :])
          if K > 64:
            tl.store(base + k1[None, :], y1, mask=rm & km1[None, :])
          if H > 0:
            tl.store(base + K + j[None, :], ra, mask=rm & jm[None, :])
            tl.store(base + K + H + j[None, :], rb, mask=rm & jm[None, :])
          if QW > K + 2 * H:
            zp = tl.arange(0, Q_TAIL_P)
            tl.store(base + K + 2 * H + zp[None, :], tl.zeros((NP, Q_TAIL_P), tl.bfloat16),
                     mask=rm & (zp[None, :] < QW - K - 2 * H))
        elif T > 1:
          _zero_qw_row(K_SUB + row_pad * (N * QW) + head[:, None] * QW, rm, NP, QW)
    else:
      sv = inv * vo_scale
      y0 = (o0 * sv).to(tl.bfloat16)
      if K > 64:
        y1 = (o1 * sv).to(tl.bfloat16)
      if arm_idx == 2:
        if t + 1 < T:
          base = V_SUB + row_k * (N * QW) + head[:, None] * QW
          tl.store(base + k0[None, :], y0, mask=rm & km0[None, :])
          if K > 64:
            tl.store(base + k1[None, :], y1, mask=rm & km1[None, :])
          if QW > K:
            zt = tl.arange(0, K_TAIL_P)
            tl.store(base + K + zt[None, :], tl.zeros((NP, K_TAIL_P), tl.bfloat16),
                     mask=rm & (zt[None, :] < QW - K))
        elif T > 1:
          _zero_qw_row(V_SUB + row_pad * (N * QW) + head[:, None] * QW, rm, NP, QW)
      else:
        base = O_M + row * (N * K) + head[:, None] * K
        tl.store(base + k0[None, :], y0, mask=rm & km0[None, :])
        if K > 64:
          tl.store(base + k1[None, :], y1, mask=rm & km1[None, :])


@triton.jit
def _read_bwd_group_wide_from_fb_kernel(
    FB, s_fb, INV_BUF, UP, s_up, GATES, s_gt, XB, SKEY, C_QK, C_VO, COS, SIN,
    DQ_IN, sdq_b, sdq_t, sdq_n,
    DK_IN, sdk_b, sdk_t, sdk_n,
    DV_IN, sdv_b, sdv_t, sdv_n,
    DO_M,
    DMT_G, s_dmt,
    DSTD, s_dstd, DUP, s_dup, DGATES, s_dgt,
    DSK_ROWS, s_dskr, DC_ROWS,
    rows, T, key_eps, key_scale, static_scale, q_mult,
    N: tl.constexpr, K: tl.constexpr, K1P: tl.constexpr,
    V: tl.constexpr, V0: tl.constexpr, V1: tl.constexpr,
    H: tl.constexpr, HP: tl.constexpr, CRP: tl.constexpr,
):
  """Split-group wide read backward consuming precomputed `(FB, INV_BUF)` (no MShift/MNorm recompute)."""
  row = tl.program_id(0)
  group = tl.program_id(1)
  seq = row // T
  t = row % T

  a = tl.arange(0, V0)
  am = a < V
  k0 = tl.arange(0, 64)
  km0 = k0 < K
  if K > 64:
    k1 = 64 + tl.arange(0, K1P)
    km1 = k1 < K
  if V1 > 0:
    b = V0 + tl.arange(0, V1)
    bm = b < V

  fb00, fb01, fb10, fb11 = kl._load_m(FB + row * s_fb, K, K1P, V, V0, V1)
  inv = tl.load(INV_BUF + row)

  arm_scale = tl.load(C_QK).to(tl.float32) if group == 0 else tl.load(C_VO).to(tl.float32)
  GB = XB + V

  c = tl.arange(0, CRP)
  cm = c < 2 * N
  rm = cm[:, None]
  am2 = rm & am[None, :]
  if V1 > 0:
    bm2 = rm & bm[None, :]
  cg = group * 2 * N + c
  arm = c // N
  head = c % N

  sig_d = tl.sigmoid(
      tl.load(GATES + row * s_gt + cg, mask=cm, other=0.0).to(tl.float32)
      + tl.load(GB + cg, mask=cm, other=0.0).to(tl.float32)
  )
  sig_s = tl.sigmoid(
      tl.load(GATES + row * s_gt + 4 * N + cg, mask=cm, other=0.0).to(tl.float32)
      + tl.load(GB + 4 * N + cg, mask=cm, other=0.0).to(tl.float32)
  )
  gd = key_scale * sig_d
  gs = static_scale * sig_s

  up0_a = tl.load(UP + row * s_up + cg[:, None] * V + a[None, :], mask=am2, other=0.0).to(tl.float32)
  ss_up = tl.sum(up0_a * up0_a, 1)
  if V1 > 0:
    up1_a = tl.load(UP + row * s_up + cg[:, None] * V + b[None, :], mask=bm2, other=0.0).to(tl.float32)
    ss_up += tl.sum(up1_a * up1_a, 1)
  inv_up = tl.rsqrt(ss_up / V + key_eps)
  sk0_a = tl.load(SKEY + cg[:, None] * V + a[None, :], mask=am2, other=0.0).to(tl.float32)
  r0 = ((up0_a * inv_up[:, None]) * gd[:, None] + sk0_a * gs[:, None]).to(tl.bfloat16)
  if V1 > 0:
    sk1_a = tl.load(SKEY + cg[:, None] * V + b[None, :], mask=bm2, other=0.0).to(tl.float32)
    r1 = ((up1_a * inv_up[:, None]) * gd[:, None] + sk1_a * gs[:, None]).to(tl.bfloat16)

  if group == 0:
    is_q = arm == 0
    qm = tl.where(is_q, q_mult, 1.0)
    qb = DQ_IN + seq * sdq_b + (t - 1) * sdq_t + head[:, None] * sdq_n
    kb = DK_IN + seq * sdk_b + t * sdk_t + head[:, None] * sdk_n
    vm = (cm & ((is_q & (t > 0)) | ((~is_q) & (t + 1 < T))))[:, None]
    base = tl.where(is_q[:, None], qb, kb)
    do0_b = (tl.load(base + k0[None, :], mask=vm & km0[None, :], other=0.0).to(tl.float32) * qm[:, None]).to(tl.bfloat16)
    if K > 64:
      do1_b = (tl.load(base + k1[None, :], mask=vm & km1[None, :], other=0.0).to(tl.float32) * qm[:, None]).to(tl.bfloat16)
    if H > 0:
      j = tl.arange(0, HP)
      jm = j < H
      dra = tl.load(base + K + j[None, :], mask=vm & jm[None, :], other=0.0).to(tl.float32) * qm[:, None]
      drb = tl.load(base + K + H + j[None, :], mask=vm & jm[None, :], other=0.0).to(tl.float32) * qm[:, None]
      cs = tl.load(COS + t * H + j, mask=jm, other=0.0).to(tl.float32)
      sn = tl.load(SIN + t * H + j, mask=jm, other=0.0).to(tl.float32)
      tl.store(DSTD + row * s_dstd + c[:, None] * (2 * H) + j[None, :], (dra * cs[None, :] + drb * sn[None, :]).to(tl.bfloat16), mask=rm & jm[None, :])
      tl.store(DSTD + row * s_dstd + c[:, None] * (2 * H) + H + j[None, :], (drb * cs[None, :] - dra * sn[None, :]).to(tl.bfloat16), mask=rm & jm[None, :])
  else:
    is_v = arm == 0
    vb = DV_IN + seq * sdv_b + t * sdv_t + head[:, None] * sdv_n
    ob = DO_M + row * (N * K) + head[:, None] * K
    vm = (cm & ((is_v & (t + 1 < T)) | (~is_v)))[:, None]
    base = tl.where(is_v[:, None], vb, ob)
    do0_b = tl.load(base + k0[None, :], mask=vm & km0[None, :], other=0.0)
    if K > 64:
      do1_b = tl.load(base + k1[None, :], mask=vm & km1[None, :], other=0.0)

  dmt_ptr = DMT_G + (group * rows + row) * s_dmt
  tl.store(dmt_ptr + k0[:, None] * V + a[None, :], (tl.dot(tl.trans(do0_b), r0) * arm_scale).to(tl.bfloat16), mask=km0[:, None] & am[None, :])
  if K > 64:
    tl.store(dmt_ptr + k1[:, None] * V + a[None, :], (tl.dot(tl.trans(do1_b), r0) * arm_scale).to(tl.bfloat16), mask=km1[:, None] & am[None, :])
  if V1 > 0:
    tl.store(dmt_ptr + k0[:, None] * V + b[None, :], (tl.dot(tl.trans(do0_b), r1) * arm_scale).to(tl.bfloat16), mask=km0[:, None] & bm[None, :])
    if K > 64:
      tl.store(dmt_ptr + k1[:, None] * V + b[None, :], (tl.dot(tl.trans(do1_b), r1) * arm_scale).to(tl.bfloat16), mask=km1[:, None] & bm[None, :])

  dr_un0 = tl.dot(do0_b, fb00)
  if K > 64:
    dr_un0 += tl.dot(do1_b, fb10)
  dr_un0 = dr_un0 * inv
  dc_row = tl.sum(tl.sum(dr_un0 * r0.to(tl.float32), 1), 0)
  dr0 = dr_un0 * arm_scale

  if V1 > 0:
    dr_un1 = tl.dot(do0_b, fb01)
    if K > 64:
      dr_un1 += tl.dot(do1_b, fb11)
    dr_un1 = dr_un1 * inv
    dc_row += tl.sum(tl.sum(dr_un1 * r1.to(tl.float32), 1), 0)
    dr1 = dr_un1 * arm_scale

  uh0 = tl.load(UP + row * s_up + cg[:, None] * V + a[None, :], mask=am2, other=0.0).to(tl.float32) * inv_up[:, None]
  sk0 = tl.load(SKEY + cg[:, None] * V + a[None, :], mask=am2, other=0.0).to(tl.float32)
  dgd = tl.sum(dr0 * uh0, 1)
  dgs = tl.sum(dr0 * sk0, 1)
  duh0 = dr0 * gd[:, None]
  dot_u = tl.sum(duh0 * uh0, 1)
  if V1 > 0:
    uh1 = tl.load(UP + row * s_up + cg[:, None] * V + b[None, :], mask=bm2, other=0.0).to(tl.float32) * inv_up[:, None]
    sk1 = tl.load(SKEY + cg[:, None] * V + b[None, :], mask=bm2, other=0.0).to(tl.float32)
    dgd += tl.sum(dr1 * uh1, 1)
    dgs += tl.sum(dr1 * sk1, 1)
    duh1 = dr1 * gd[:, None]
    dot_u += tl.sum(duh1 * uh1, 1)

  tl.store(DGATES + row * s_dgt + cg, (dgd * gd * (1.0 - sig_d)).to(tl.bfloat16), mask=cm)
  tl.store(DGATES + row * s_dgt + 4 * N + cg, (dgs * gs * (1.0 - sig_s)).to(tl.bfloat16), mask=cm)

  dot_u = dot_u / V
  tl.store(DUP + row * s_dup + cg[:, None] * V + a[None, :],
           ((duh0 - uh0 * dot_u[:, None]) * inv_up[:, None]).to(tl.bfloat16), mask=am2)
  tl.store(DSK_ROWS + row * s_dskr + cg[:, None] * V + a[None, :],
           (dr0 * gs[:, None]).to(tl.bfloat16), mask=am2)
  if V1 > 0:
    tl.store(DUP + row * s_dup + cg[:, None] * V + b[None, :],
             ((duh1 - uh1 * dot_u[:, None]) * inv_up[:, None]).to(tl.bfloat16), mask=bm2)
    tl.store(DSK_ROWS + row * s_dskr + cg[:, None] * V + b[None, :],
             (dr1 * gs[:, None]).to(tl.bfloat16), mask=bm2)
  tl.store(DC_ROWS + group * rows + row, dc_row)


@triton.jit
def _mshift_mnorm_bwd_inv_kernel(
    M, s_m, DMT_G, s_dmt, S_BUF, s_sb, INV_BUF,
    DM, s_dm, DSHIFT, s_dsh,
    rows, T, CH,
    R: tl.constexpr,
    K: tl.constexpr, K1P: tl.constexpr, V: tl.constexpr, V0: tl.constexpr, V1: tl.constexpr,
):
  """Fused MNorm + MShift backward consuming precomputed `INV_BUF` (drops `ss` reduction + `rsqrt`)."""
  pid = tl.program_id(0)
  seq = pid // CH
  t_lo = (pid % CH) * R
  t_start = tl.minimum(t_lo + R, T - 1)

  a = tl.arange(0, V0)
  am = a < V
  if V1 > 0:
    b = V0 + tl.arange(0, V1)
    bm = b < V

  cb00 = tl.zeros((64, V0), tl.bfloat16)
  cb10 = tl.zeros((K1P, V0), tl.bfloat16)
  sn0 = tl.zeros((V0,), tl.float32)
  if V1 > 0:
    cb01 = tl.zeros((64, V1), tl.bfloat16)
    cb11 = tl.zeros((K1P, V1), tl.bfloat16)
    sn1 = tl.zeros((V1,), tl.float32)
  else:
    cb01 = tl.zeros((64, 16), tl.bfloat16)
    cb11 = tl.zeros((K1P, 16), tl.bfloat16)
    sn1 = tl.zeros((16,), tl.float32)

  for i in range(0, R + 1):
    t = t_lo + R - i
    if t <= t_start:
      row = seq * T + t
      pp = M + (row - 1) * s_m if t > 0 else M + row * s_m
      p00, p01, p10, p11 = kl._load_m(pp, K, K1P, V, V0, V1)
      m00, m01, m10, m11 = kl._load_m(M + row * s_m, K, K1P, V, V0, V1)
      g00, g01, g10, g11 = kl._load_m(DMT_G + row * s_dmt, K, K1P, V, V0, V1)
      h00, h01, h10, h11 = kl._load_m(DMT_G + (rows + row) * s_dmt, K, K1P, V, V0, V1)

      s0_1d = tl.load(S_BUF + row * s_sb + a, mask=am, other=0.0)
      s0 = s0_1d[None, :]
      m00_f = m00.to(tl.float32)
      pm00 = p00.to(tl.float32) - m00_f
      f00 = (m00_f + s0 * pm00).to(tl.bfloat16).to(tl.float32)
      dmt00 = g00.to(tl.float32) + h00.to(tl.float32)
      dot_m = tl.sum(tl.sum(dmt00 * f00, 1), 0)
      if K > 64:
        m10_f = m10.to(tl.float32)
        pm10 = p10.to(tl.float32) - m10_f
        f10 = (m10_f + s0 * pm10).to(tl.bfloat16).to(tl.float32)
        dmt10 = g10.to(tl.float32) + h10.to(tl.float32)
        dot_m += tl.sum(tl.sum(dmt10 * f10, 1), 0)
      if V1 > 0:
        s1_1d = tl.load(S_BUF + row * s_sb + b, mask=bm, other=0.0)
        s1 = s1_1d[None, :]
        m01_f = m01.to(tl.float32)
        pm01 = p01.to(tl.float32) - m01_f
        f01 = (m01_f + s1 * pm01).to(tl.bfloat16).to(tl.float32)
        dmt01 = g01.to(tl.float32) + h01.to(tl.float32)
        dot_m += tl.sum(tl.sum(dmt01 * f01, 1), 0)
        if K > 64:
          m11_f = m11.to(tl.float32)
          pm11 = p11.to(tl.float32) - m11_f
          f11 = (m11_f + s1 * pm11).to(tl.bfloat16).to(tl.float32)
          dmt11 = g11.to(tl.float32) + h11.to(tl.float32)
          dot_m += tl.sum(tl.sum(dmt11 * f11, 1), 0)
      inv = tl.load(INV_BUF + row)
      coeff = (dot_m / (K * V)) * (inv * inv)

      db00 = ((dmt00 - f00 * coeff) * inv).to(tl.bfloat16)
      db10 = ((dmt10 - f10 * coeff) * inv).to(tl.bfloat16) if K > 64 else tl.zeros((K1P, V0), tl.bfloat16)
      if V1 > 0:
        db01 = ((dmt01 - f01 * coeff) * inv).to(tl.bfloat16)
        db11 = ((dmt11 - f11 * coeff) * inv).to(tl.bfloat16) if K > 64 else tl.zeros((K1P, V1), tl.bfloat16)
      else:
        db01 = tl.zeros((64, 16), tl.bfloat16)
        db11 = tl.zeros((K1P, 16), tl.bfloat16)

      if i > 0:
        d00_f = db00.to(tl.float32)
        dm00 = ((1.0 - s0) * d00_f + sn0[None, :] * cb00.to(tl.float32)).to(tl.bfloat16)
        ds0 = tl.sum(d00_f * pm00, 0)
        if K > 64:
          d10_f = db10.to(tl.float32)
          dm10 = ((1.0 - s0) * d10_f + sn0[None, :] * cb10.to(tl.float32)).to(tl.bfloat16)
          ds0 += tl.sum(d10_f * pm10, 0)
        else:
          dm10 = tl.zeros((K1P, V0), tl.bfloat16)
        tl.store(DSHIFT + row * s_dsh + a, (ds0 * s0_1d * (1.0 - s0_1d)).to(tl.bfloat16), mask=am)

        if V1 > 0:
          d01_f = db01.to(tl.float32)
          dm01 = ((1.0 - s1) * d01_f + sn1[None, :] * cb01.to(tl.float32)).to(tl.bfloat16)
          ds1 = tl.sum(d01_f * pm01, 0)
          if K > 64:
            d11_f = db11.to(tl.float32)
            dm11 = ((1.0 - s1) * d11_f + sn1[None, :] * cb11.to(tl.float32)).to(tl.bfloat16)
            ds1 += tl.sum(d11_f * pm11, 0)
          else:
            dm11 = tl.zeros((K1P, V1), tl.bfloat16)
          tl.store(DSHIFT + row * s_dsh + b, (ds1 * s1_1d * (1.0 - s1_1d)).to(tl.bfloat16), mask=bm)
        else:
          dm01 = tl.zeros((64, 16), tl.bfloat16)
          dm11 = tl.zeros((K1P, 16), tl.bfloat16)

        kl._store_m(DM + row * s_dm, dm00, dm01, dm10, dm11, K, K1P, V, V0, V1)

      cb00, cb01, cb10, cb11 = db00, db01, db10, db11
      sn0 = s0_1d
      if V1 > 0:
        sn1 = s1_1d


@torch.library.custom_op("loom_train::loom_read_fwd", mutates_args=())
def _loom_read_fwd_op(
    m: torch.Tensor, shift: torch.Tensor, std: torch.Tensor, up: torch.Tensor,
    gates: torch.Tensor, static_key: torch.Tensor, xb: torch.Tensor,
    c_qk: torch.Tensor, c_vo: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
    batch: int, n: int, k: int, v: int, s: int,
    eps: float, key_eps: float, key_scale: float, static_scale: float, q_mult: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  m, shift, std, up, gates = (_unit_inner(x) for x in (m, shift, std, up, gates))
  static_key, xb, c_qk, c_vo, cos, sin = (x.contiguous() for x in (static_key, xb, c_qk, c_vo, cos, sin))
  rows = m.shape[0]
  t = rows // batch
  t_attn = t if t > 1 else 0
  qw = _fa_head_dim(k, s)
  q_sub = m.new_empty((batch, t_attn, n, qw))
  k_sub = m.new_empty((batch, t_attn, n, qw))
  v_sub = m.new_empty((batch, t_attn, n, qw))
  o_m = m.new_empty((rows, n, k))
  cfg = _select_kernel_config(n, k, v)
  _read_fwd_train_kernel[(rows,)](
      m, m.stride(0), shift, shift.stride(0), std, std.stride(0), up, up.stride(0),
      gates, gates.stride(0), xb, static_key, c_qk, c_vo, cos, sin,
      q_sub, k_sub, v_sub, o_m,
      t, eps, key_eps, key_scale, static_scale, q_mult,
      N=n, NP=kl._pow2(n), K=k, K1P=cfg.k1p, V=v, V0=cfg.v0, V1=cfg.v1,
      H=s // 2, HP=kl._pow2(max(1, s // 2)), QW=qw,
      K_TAIL_P=max(8, kl._pow2(max(1, qw - k))),
      Q_TAIL_P=max(8, kl._pow2(max(1, qw - k - s))),
      num_warps=cfg.read_fwd_warps,
  )
  return q_sub, k_sub, v_sub, o_m


@_loom_read_fwd_op.register_fake
def _(
    m, shift, std, up, gates, static_key, xb, c_qk, c_vo, cos, sin,
    batch: int, n: int, k: int, v: int, s: int,
    eps: float, key_eps: float, key_scale: float, static_scale: float, q_mult: float,
):
  rows = m.shape[0]
  t = rows // batch
  t_attn = t if t > 1 else 0
  qw = _fa_head_dim(k, s)
  return (
      m.new_empty((batch, t_attn, n, qw)),
      m.new_empty((batch, t_attn, n, qw)),
      m.new_empty((batch, t_attn, n, qw)),
      m.new_empty((rows, n, k)),
  )


@torch.library.custom_op("loom_train::loom_read_fwd_fb", mutates_args=())
def _loom_read_fwd_fb_op(
    m: torch.Tensor, shift: torch.Tensor, std: torch.Tensor, up: torch.Tensor,
    gates: torch.Tensor, static_key: torch.Tensor, xb: torch.Tensor,
    c_qk: torch.Tensor, c_vo: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
    batch: int, n: int, k: int, v: int, s: int,
    eps: float, key_eps: float, key_scale: float, static_scale: float, q_mult: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  m, shift, std, up, gates = (_unit_inner(x) for x in (m, shift, std, up, gates))
  static_key, xb, c_qk, c_vo, cos, sin = (x.contiguous() for x in (static_key, xb, c_qk, c_vo, cos, sin))
  rows = m.shape[0]
  t = rows // batch
  t_attn = t if t > 1 else 0
  qw = _fa_head_dim(k, s)
  q_sub = m.new_empty((batch, t_attn, n, qw))
  k_sub = m.new_empty((batch, t_attn, n, qw))
  v_sub = m.new_empty((batch, t_attn, n, qw))
  o_m = m.new_empty((rows, n, k))
  fb = m.new_empty(m.shape)
  s_buf = torch.empty((rows, v), device=m.device, dtype=torch.float32)
  inv_buf = torch.empty((rows,), device=m.device, dtype=torch.float32)
  cfg = _select_kernel_config(n, k, v)
  r_prep = 8
  ch_prep = -(-t // r_prep)
  _read_m_prep_kernel[(batch * ch_prep,)](
      m, m.stride(0), shift, shift.stride(0), xb,
      fb, fb.stride(0), s_buf, s_buf.stride(0), inv_buf,
      t, ch_prep, eps, R=r_prep, K=k, K1P=cfg.k1p, V=v, V0=cfg.v0, V1=cfg.v1,
      num_warps=cfg.m_prep_warps, num_stages=1,
  )
  _read_fwd_from_fb_kernel[(rows,)](
      fb, fb.stride(0), inv_buf, std, std.stride(0), up, up.stride(0),
      gates, gates.stride(0), xb, static_key, c_qk, c_vo, cos, sin,
      q_sub, k_sub, v_sub, o_m,
      t, key_eps, key_scale, static_scale, q_mult,
      N=n, NP=kl._pow2(n), K=k, K1P=cfg.k1p, V=v, V0=cfg.v0, V1=cfg.v1,
      H=s // 2, HP=kl._pow2(max(1, s // 2)), QW=qw,
      K_TAIL_P=max(8, kl._pow2(max(1, qw - k))),
      Q_TAIL_P=max(8, kl._pow2(max(1, qw - k - s))),
      num_warps=cfg.read_fwd_fb_warps,
  )
  return q_sub, k_sub, v_sub, o_m, fb, s_buf, inv_buf


@_loom_read_fwd_fb_op.register_fake
def _(
    m, shift, std, up, gates, static_key, xb, c_qk, c_vo, cos, sin,
    batch: int, n: int, k: int, v: int, s: int,
    eps: float, key_eps: float, key_scale: float, static_scale: float, q_mult: float,
):
  rows = m.shape[0]
  t = rows // batch
  t_attn = t if t > 1 else 0
  qw = _fa_head_dim(k, s)
  return (
      m.new_empty((batch, t_attn, n, qw)),
      m.new_empty((batch, t_attn, n, qw)),
      m.new_empty((batch, t_attn, n, qw)),
      m.new_empty((rows, n, k)),
      m.new_empty(m.shape),
      torch.empty((rows, v), device=m.device, dtype=torch.float32),
      torch.empty((rows,), device=m.device, dtype=torch.float32),
  )


def _run_read_bwd_from_fb(
    dq_sub: torch.Tensor, dk_sub: torch.Tensor, dv_sub: torch.Tensor, do_m: torch.Tensor,
    m: torch.Tensor, fb: torch.Tensor, s_buf: torch.Tensor, inv_buf: torch.Tensor,
    up: torch.Tensor, gates: torch.Tensor,
    static_key: torch.Tensor, xb: torch.Tensor, c_qk: torch.Tensor, c_vo: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor,
    batch: int, n: int, k: int, v: int, s: int,
    key_eps: float, key_scale: float, static_scale: float, q_mult: float,
):
  rows = m.shape[0]
  t = rows // batch
  cfg = _select_kernel_config(n, k, v)
  dstd = m.new_empty((rows, 2 * n * s))
  dmt_g = m.new_empty((2 * rows, k * v))
  dm = m.new_empty(m.shape)
  dshift = m.new_empty((rows, v))
  dup = up.new_empty(up.shape)
  dgates = gates.new_empty(gates.shape)
  dsk_rows = m.new_empty((rows, 4 * n, v))
  dc_rows = torch.empty((2, rows), device=m.device, dtype=torch.float32)

  _read_bwd_group_wide_from_fb_kernel[(rows, 2)](
      fb, fb.stride(0), inv_buf, up, up.stride(0), gates, gates.stride(0),
      xb, static_key, c_qk, c_vo, cos, sin,
      dq_sub, dq_sub.stride(0), dq_sub.stride(1), dq_sub.stride(2),
      dk_sub, dk_sub.stride(0), dk_sub.stride(1), dk_sub.stride(2),
      dv_sub, dv_sub.stride(0), dv_sub.stride(1), dv_sub.stride(2),
      do_m,
      dmt_g, dmt_g.stride(0),
      dstd, dstd.stride(0), dup, dup.stride(0), dgates, dgates.stride(0),
      dsk_rows, dsk_rows.stride(0), dc_rows,
      rows, t, key_eps, key_scale, static_scale, q_mult,
      N=n, K=k, K1P=cfg.k1p, V=v, V0=cfg.v0, V1=cfg.v1,
      H=s // 2, HP=kl._pow2(max(1, s // 2)), CRP=kl._pow2(2 * n),
      num_warps=cfg.read_bwd_fb_warps, num_stages=2,
  )
  r = _MSHIFT_BWD_ROWS
  ch = -(-t // r)
  _mshift_mnorm_bwd_inv_kernel[(batch * ch,)](
      m, m.stride(0), dmt_g, dmt_g.stride(0), s_buf, s_buf.stride(0), inv_buf,
      dm, dm.stride(0), dshift, dshift.stride(0),
      rows, t, ch, R=r, K=k, K1P=cfg.k1p, V=v, V0=cfg.v0, V1=cfg.v1,
      num_warps=cfg.mshift_bwd_warps, num_stages=1,
  )
  dstatic_key = dsk_rows.sum(dim=0, dtype=torch.float32).to(static_key.dtype)
  dxb = torch.cat((dshift.sum(0, dtype=torch.float32), dgates.sum(0, dtype=torch.float32)), dim=0).to(xb.dtype)
  dc_qk = dc_rows[0].sum().to(c_qk.dtype).view_as(c_qk)
  dc_vo = dc_rows[1].sum().to(c_vo.dtype).view_as(c_vo)
  return dm, dshift, dstd, dup, dgates, dstatic_key, dxb, dc_qk, dc_vo


@torch.library.custom_op("loom_train::loom_read_bwd_fb", mutates_args=())
def _loom_read_bwd_fb_op(
    dq_sub: torch.Tensor, dk_sub: torch.Tensor, dv_sub: torch.Tensor, do_m: torch.Tensor,
    m: torch.Tensor, fb: torch.Tensor, s_buf: torch.Tensor, inv_buf: torch.Tensor,
    up: torch.Tensor, gates: torch.Tensor,
    static_key: torch.Tensor, xb: torch.Tensor, c_qk: torch.Tensor, c_vo: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor,
    batch: int, n: int, k: int, v: int, s: int,
    key_eps: float, key_scale: float, static_scale: float, q_mult: float,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
]:
  m, fb, up, gates = (_unit_inner(x) for x in (m, fb, up, gates))
  s_buf, inv_buf, static_key, xb, c_qk, c_vo, cos, sin = (
      x.contiguous() for x in (s_buf, inv_buf, static_key, xb, c_qk, c_vo, cos, sin)
  )
  if m.shape[0] // batch > 1:
    dq_sub, dk_sub, dv_sub = (_unit_inner(x) for x in (dq_sub, dk_sub, dv_sub))
  do_m = do_m.contiguous()
  return _run_read_bwd_from_fb(
      dq_sub, dk_sub, dv_sub, do_m, m, fb, s_buf, inv_buf, up, gates,
      static_key, xb, c_qk, c_vo, cos, sin,
      batch, n, k, v, s, key_eps, key_scale, static_scale, q_mult,
  )


@_loom_read_bwd_fb_op.register_fake
def _(
    dq_sub, dk_sub, dv_sub, do_m,
    m, fb, s_buf, inv_buf, up, gates, static_key, xb, c_qk, c_vo, cos, sin,
    batch: int, n: int, k: int, v: int, s: int,
    key_eps: float, key_scale: float, static_scale: float, q_mult: float,
):
  rows = m.shape[0]
  return (
      m.new_empty(m.shape), m.new_empty((rows, v)), m.new_empty((rows, 2 * n * s)),
      up.new_empty(up.shape), gates.new_empty(gates.shape), static_key.new_empty(static_key.shape),
      xb.new_empty(xb.shape), c_qk.new_empty(c_qk.shape), c_vo.new_empty(c_vo.shape),
  )


@torch.library.custom_op("loom_train::loom_read_bwd", mutates_args=())
def _loom_read_bwd_op(
    dq_sub: torch.Tensor, dk_sub: torch.Tensor, dv_sub: torch.Tensor, do_m: torch.Tensor,
    m: torch.Tensor, shift: torch.Tensor, up: torch.Tensor, gates: torch.Tensor,
    static_key: torch.Tensor, xb: torch.Tensor, c_qk: torch.Tensor, c_vo: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor,
    batch: int, n: int, k: int, v: int, s: int,
    eps: float, key_eps: float, key_scale: float, static_scale: float, q_mult: float,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
]:
  m, shift, up, gates = (_unit_inner(x) for x in (m, shift, up, gates))
  static_key, xb, c_qk, c_vo, cos, sin = (x.contiguous() for x in (static_key, xb, c_qk, c_vo, cos, sin))
  rows = m.shape[0]
  t = rows // batch
  cfg = _select_kernel_config(n, k, v)

  if t > 1:
    dq_sub, dk_sub, dv_sub = (_unit_inner(x) for x in (dq_sub, dk_sub, dv_sub))
  do_m = do_m.contiguous()

  if cfg.use_fb_read:
    fb = m.new_empty(m.shape)
    s_buf = torch.empty((rows, v), device=m.device, dtype=torch.float32)
    inv_buf = torch.empty((rows,), device=m.device, dtype=torch.float32)
    r_prep = 8
    ch_prep = -(-t // r_prep)
    _read_m_prep_kernel[(batch * ch_prep,)](
        m, m.stride(0), shift, shift.stride(0), xb,
        fb, fb.stride(0), s_buf, s_buf.stride(0), inv_buf,
        t, ch_prep, eps, R=r_prep, K=k, K1P=cfg.k1p, V=v, V0=cfg.v0, V1=cfg.v1,
        num_warps=cfg.m_prep_warps, num_stages=1,
    )
    return _run_read_bwd_from_fb(
        dq_sub, dk_sub, dv_sub, do_m, m, fb, s_buf, inv_buf, up, gates,
        static_key, xb, c_qk, c_vo, cos, sin,
        batch, n, k, v, s, key_eps, key_scale, static_scale, q_mult,
    )

  dstd = m.new_empty((rows, 2 * n * s))
  dmt_g = m.new_empty((2 * rows, k * v))
  s_buf = torch.empty((rows, v), device=m.device, dtype=torch.float32)
  dm = m.new_empty(m.shape)
  dshift = shift.new_empty(shift.shape)
  dup = up.new_empty(up.shape)
  dgates = gates.new_empty(gates.shape)
  dsk_rows = m.new_empty((rows, 4 * n, v))
  dc_rows = torch.empty((2, rows), device=m.device, dtype=torch.float32)

  if 2 * n <= 32 and cfg.v1 == 0:
    _read_bwd_group_kernel[(rows, 2)](
        m, m.stride(0), shift, shift.stride(0), up, up.stride(0), gates, gates.stride(0),
        xb, static_key, c_qk, c_vo, cos, sin,
        dq_sub, dq_sub.stride(0), dq_sub.stride(1), dq_sub.stride(2),
        dk_sub, dk_sub.stride(0), dk_sub.stride(1), dk_sub.stride(2),
        dv_sub, dv_sub.stride(0), dv_sub.stride(1), dv_sub.stride(2),
        do_m,
        dmt_g, dmt_g.stride(0), s_buf, s_buf.stride(0),
        dstd, dstd.stride(0), dup, dup.stride(0), dgates, dgates.stride(0),
        dsk_rows, dsk_rows.stride(0), dc_rows,
        rows, t, eps, key_eps, key_scale, static_scale, q_mult,
        SUB=1, N=n, K=k, K1P=cfg.k1p, V=v, V0=cfg.v0, V1=cfg.v1,
        H=s // 2, HP=kl._pow2(max(1, s // 2)), CR=2 * n, CRP=kl._pow2(2 * n),
        num_warps=2, num_stages=1,
    )
  else:
    sub = -(-(2 * n) // 32)
    cr = -(-(2 * n) // sub)
    _read_bwd_group_kernel[(rows, 2)](
        m, m.stride(0), shift, shift.stride(0), up, up.stride(0), gates, gates.stride(0),
        xb, static_key, c_qk, c_vo, cos, sin,
        dq_sub, dq_sub.stride(0), dq_sub.stride(1), dq_sub.stride(2),
        dk_sub, dk_sub.stride(0), dk_sub.stride(1), dk_sub.stride(2),
        dv_sub, dv_sub.stride(0), dv_sub.stride(1), dv_sub.stride(2),
        do_m,
        dmt_g, dmt_g.stride(0), s_buf, s_buf.stride(0),
        dstd, dstd.stride(0), dup, dup.stride(0), dgates, dgates.stride(0),
        dsk_rows, dsk_rows.stride(0), dc_rows,
        rows, t, eps, key_eps, key_scale, static_scale, q_mult,
        SUB=sub, N=n, K=k, K1P=cfg.k1p, V=v, V0=cfg.v0, V1=cfg.v1,
        H=s // 2, HP=kl._pow2(max(1, s // 2)), CR=cr, CRP=kl._pow2(cr),
        num_warps=4,
    )

  r = _MSHIFT_BWD_ROWS
  ch = -(-t // r)
  _mshift_mnorm_bwd_kernel[(batch * ch,)](
      m, m.stride(0), dmt_g, dmt_g.stride(0), s_buf, s_buf.stride(0),
      dm, dm.stride(0), dshift, dshift.stride(0),
      rows, t, ch, eps,
      R=r, K=k, K1P=cfg.k1p, V=v, V0=cfg.v0, V1=cfg.v1,
      num_warps=cfg.mshift_bwd_warps, num_stages=1,
  )

  dstatic_key = dsk_rows.sum(dim=0, dtype=torch.float32).to(static_key.dtype)
  dxb = torch.cat((dshift.sum(0, dtype=torch.float32), dgates.sum(0, dtype=torch.float32)), dim=0).to(xb.dtype)
  dc_qk = dc_rows[0].sum().to(c_qk.dtype).view_as(c_qk)
  dc_vo = dc_rows[1].sum().to(c_vo.dtype).view_as(c_vo)
  return dm, dshift, dstd, dup, dgates, dstatic_key, dxb, dc_qk, dc_vo


@_loom_read_bwd_op.register_fake
def _(
    dq_sub, dk_sub, dv_sub, do_m,
    m, shift, up, gates, static_key, xb, c_qk, c_vo, cos, sin,
    batch: int, n: int, k: int, v: int, s: int,
    eps: float, key_eps: float, key_scale: float, static_scale: float, q_mult: float,
):
  rows = m.shape[0]
  return (
      m.new_empty(m.shape), shift.new_empty(shift.shape), m.new_empty((rows, 2 * n * s)),
      up.new_empty(up.shape), gates.new_empty(gates.shape), static_key.new_empty(static_key.shape),
      xb.new_empty(xb.shape), c_qk.new_empty(c_qk.shape), c_vo.new_empty(c_vo.shape),
  )


def _loom_read_setup(ctx, inputs, output):
  m, shift, _std, up, gates, static_key, xb, c_qk, c_vo, cos, sin, *meta = inputs
  ctx.save_for_backward(m, shift, up, gates, static_key, xb, c_qk, c_vo, cos, sin)
  ctx.meta = tuple(meta)


def _loom_read_backward(ctx, dq_sub, dk_sub, dv_sub, do_m):
  m, shift, up, gates, static_key, xb, c_qk, c_vo, cos, sin = ctx.saved_tensors
  batch, n, k, v, s, eps, key_eps, key_scale, static_scale, q_mult = ctx.meta
  rows = m.shape[0]
  t = rows // batch
  t_attn = t if t > 1 else 0
  qw = _fa_head_dim(k, s)
  if dq_sub is None:
    dq_sub = m.new_zeros((batch, t_attn, n, qw))
  if dk_sub is None:
    dk_sub = m.new_zeros((batch, t_attn, n, qw))
  if dv_sub is None:
    dv_sub = m.new_zeros((batch, t_attn, n, qw))
  if do_m is None:
    do_m = m.new_zeros((rows, n, k))
  dm, dshift, dstd, dup, dgates, dstatic_key, dxb, dc_qk, dc_vo = _loom_read_bwd_op(
      dq_sub, dk_sub, dv_sub, do_m,
      m, shift, up, gates, static_key, xb, c_qk, c_vo, cos, sin,
      batch, n, k, v, s, eps, key_eps, key_scale, static_scale, q_mult,
  )
  return (
      dm, dshift, dstd, dup, dgates, dstatic_key, dxb, dc_qk, dc_vo, None, None,
      None, None, None, None, None, None, None, None, None, None,
  )


_loom_read_fwd_op.register_autograd(_loom_read_backward, setup_context=_loom_read_setup)


def _loom_read_fb_setup(ctx, inputs, output):
  m, _shift, _std, up, gates, static_key, xb, c_qk, c_vo, cos, sin, *meta = inputs
  _q, _k, _v, _o, fb, s_buf, inv_buf = output
  ctx.save_for_backward(m, fb, s_buf, inv_buf, up, gates, static_key, xb, c_qk, c_vo, cos, sin)
  ctx.meta = tuple(meta)


def _loom_read_fb_backward(ctx, dq_sub, dk_sub, dv_sub, do_m, _dfb, _ds_buf, _dinv_buf):
  m, fb, s_buf, inv_buf, up, gates, static_key, xb, c_qk, c_vo, cos, sin = ctx.saved_tensors
  batch, n, k, v, s, _eps, key_eps, key_scale, static_scale, q_mult = ctx.meta
  rows = m.shape[0]
  t = rows // batch
  t_attn = t if t > 1 else 0
  qw = _fa_head_dim(k, s)
  if dq_sub is None:
    dq_sub = m.new_zeros((batch, t_attn, n, qw))
  if dk_sub is None:
    dk_sub = m.new_zeros((batch, t_attn, n, qw))
  if dv_sub is None:
    dv_sub = m.new_zeros((batch, t_attn, n, qw))
  if do_m is None:
    do_m = m.new_zeros((rows, n, k))
  dm, dshift, dstd, dup, dgates, dstatic_key, dxb, dc_qk, dc_vo = _loom_read_bwd_fb_op(
      dq_sub, dk_sub, dv_sub, do_m,
      m, fb, s_buf, inv_buf, up, gates, static_key, xb, c_qk, c_vo, cos, sin,
      batch, n, k, v, s, key_eps, key_scale, static_scale, q_mult,
  )
  return (
      dm, dshift, dstd, dup, dgates, dstatic_key, dxb, dc_qk, dc_vo, None, None,
      None, None, None, None, None, None, None, None, None, None,
  )


_loom_read_fwd_fb_op.register_autograd(_loom_read_fb_backward, setup_context=_loom_read_fb_setup)


def loom_read_triton(
    m: torch.Tensor,
    shift: torch.Tensor,
    std: torch.Tensor,
    up: torch.Tensor,
    gates: torch.Tensor,
    static_key: torch.Tensor,
    xb: torch.Tensor,
    c_qk: torch.Tensor,
    c_vo: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    batch: int, n: int, k: int, v: int, s: int,
    eps: float, key_eps: float, key_scale: float, static_scale: float, q_mult: float,
    cache_fb: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Autograd + `torch.compile`-enabled fused Triton MShift + MNorm + 4-arm MRead + RoPE."""
  dt = m.dtype
  cfg = _select_kernel_config(n, k, v)
  if cache_fb and torch.is_grad_enabled() and cfg.use_fb_read:
    q_sub, k_sub, v_sub, o_m, _fb, _s_buf, _inv_buf = _loom_read_fwd_fb_op(
        m, shift, std, up, gates, static_key.to(dt), xb.to(dt), c_qk.to(dt), c_vo.to(dt), cos, sin,
        batch, n, k, v, s, eps, key_eps, key_scale, static_scale, q_mult,
    )
    return q_sub, k_sub, v_sub, o_m
  return _loom_read_fwd_op(
      m, shift, std, up, gates, static_key.to(dt), xb.to(dt), c_qk.to(dt), c_vo.to(dt), cos, sin,
      batch, n, k, v, s, eps, key_eps, key_scale, static_scale, q_mult,
  )
