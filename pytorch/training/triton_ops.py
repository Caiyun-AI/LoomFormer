"""Fused Triton forward & backward autograd kernels for LoomFormer training (Llama backbone).

Reuses the forward Triton primitives from `fast_inference.kernels_llama` (`_write_kernel`,
`_load_m`, `_store_m`, `_group_keys`, `_vsplit`, `_ksplit`, `_pow2`, `qk_width`) and pairs them
with atomic-free, zero-spill Triton backward kernels and contiguous `[B, T-1, N, QW]`
FlashAttention-2 layout.

Physical memory layout matches `fast_inference/kernels_llama.py`:
  - `M`: `[rows, K * V]` (`rows = B * T`, `K = head_dim >= 64`, `V = d_k`, k-major)
  - `Q_sub, K_sub, V_sub`: `[B, T - 1, N, QW]` (1-step shifted so native `is_causal=True`
    FlashAttention-2 computes strictly causal `s < t` with zero transpose/pad copies)
  - `O_M`: `[rows, N, K]` (per-token local vertical bypass added to attention output)
"""
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import triton
import triton.language as tl

from fast_inference import kernels_llama as kl


def _unit_inner(t: torch.Tensor) -> torch.Tensor:
  """Ensure 2-D tensor has unit inner stride (`stride(-1) == 1`) without copying column slices."""
  return t if t.stride(-1) == 1 else t.contiguous()


def can_use_triton_loom(tensor: torch.Tensor, head_dim: int) -> bool:
  """Check whether the fused Triton Loom kernels can run on this tensor & head geometry."""
  return tensor.is_cuda and tensor.dtype == torch.bfloat16 and head_dim >= 64


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


def strict_causal_attention_add_train(
    q_sub: torch.Tensor, k_sub: torch.Tensor, v_sub: torch.Tensor, o_m: torch.Tensor,
    batch: int, t: int, n: int, k: int,
) -> torch.Tensor:
  """Strictly causal (`s < t`, t=0 -> 0) FlashAttention-2 on shifted `[B, T-1, N, QW]` + `o_m`."""
  if t <= 1:
    return o_m.reshape(batch * t, n * k)
  y_sub = _sdpa_flash_or_cudnn(q_sub.transpose(1, 2), k_sub.transpose(1, 2), v_sub.transpose(1, 2))
  # Left-pad position t=0 with zeros and add the local vertical bypass `o_m`
  y = F.pad(y_sub[..., :k].transpose(1, 2), (0, 0, 0, 0, 1, 0))
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

  c0 = tl.load(C + row * s_c + n[:, None] * K + k0[None, :], mask=nm[:, None], other=0.0).to(tl.float32)
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

  # Gate gradient: d_sig = sum_k d(c_eff) * c_hat
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
  tl.store(DC + row * s_dc + n[:, None] * K + k0[None, :], dc0.to(tl.bfloat16), mask=nm[:, None])
  if K > 64:
    tl.store(DC + row * s_dc + n[:, None] * K + k1[None, :], dc1.to(tl.bfloat16), mask=nm[:, None] & km1[None, :])

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
  v0, v1 = kl._vsplit(v)
  kl._write_kernel[(rows,)](
      m_old, m_old.stride(0), m_new,
      c1, c1.stride(0), m_old, 0, 0, g1, g1.stride(0), a1, a1.stride(0),
      c2, c2.stride(0), g2, g2.stride(0), a2, a2.stride(0),
      wb, write_eps, False,
      N=n, NP=kl._pow2(n), K=k, K1P=kl._ksplit(k), V=v, V0=v0, V1=v1,
      num_warps=1 if v1 == 0 and v0 <= 32 else 2,
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
  v0, v1 = kl._vsplit(v)
  _write_bwd_kernel[(rows,)](
      dm_new, dm_new.stride(0),
      c1, c1.stride(0), g1, g1.stride(0), a1, a1.stride(0),
      c2, c2.stride(0), g2, g2.stride(0), a2, a2.stride(0),
      wb,
      dc1, dc1.stride(0), dg1, dg1.stride(0), da1, da1.stride(0),
      dc2, dc2.stride(0), dg2, dg2.stride(0), da2, da2.stride(0),
      write_eps,
      N=n, NP=kl._pow2(n), K=k, K1P=kl._ksplit(k), V=v, V0=v0, V1=v1,
      num_warps=1 if v1 == 0 and v0 <= 32 else 2,
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
  return _loom_write_fwd_op(m_old, c1, g1, a1, c2, g2, a2, wb, n, k, v, write_eps)


# ==============================================================================
# 3. Loom Read (MShift + MNorm + 4-Arm MRead + RoPE, Forward & Zero-Spill Bwd)
# ==============================================================================

_MSHIFT_BWD_ROWS = 16  # consecutive token rows per MShift/MNorm backward program (carry kept in registers)


@triton.jit
def _group_read_direct(
    group, row, seq, t, T, c, cm, r0, r1, f00, f01, f10, f11, inv,
    STD, s_std, COS, SIN,
    Q_SUB, K_SUB, V_SUB, O_M,
    qk_scale, vo_scale, q_mult,
    N: tl.constexpr, K: tl.constexpr, K1P: tl.constexpr, V1: tl.constexpr,
    H: tl.constexpr, HP: tl.constexpr, QW: tl.constexpr, K_TAIL_P: tl.constexpr, CRP: tl.constexpr,
):
  """Compute 2N read arms (group 0: Q/K + RoPE, group 1: V/O_M) and store shifted `[B, T-1, N, QW]`."""
  k0 = tl.arange(0, 64)
  o0 = tl.dot(r0, tl.trans(f00))
  if K > 64:
    k1 = 64 + tl.arange(0, K1P)
    km1 = k1 < K
    o1 = tl.dot(r0, tl.trans(f10))
  if V1 > 0:
    o0 += tl.dot(r1, tl.trans(f01))
    if K > 64:
      o1 += tl.dot(r1, tl.trans(f11))

  arm = c // N
  head = c % N
  sm = cm[:, None]
  tm1 = T - 1
  row_q = seq * tm1 + (t - 1)
  row_k = seq * tm1 + t

  if group == 0:
    is_q = arm == 0
    qm = tl.where(is_q, q_mult, 1.0)
    sq = inv * qk_scale * qm
    y0 = (o0 * sq[:, None]).to(tl.bfloat16)
    if K > 64:
      y1 = (o1 * sq[:, None]).to(tl.bfloat16)

    j = tl.arange(0, HP)
    jm = j < H
    sa = tl.load(STD + row * s_std + c[:, None] * (2 * H) + j[None, :], mask=sm & jm[None, :], other=0.0).to(tl.float32)
    sb = tl.load(STD + row * s_std + c[:, None] * (2 * H) + H + j[None, :], mask=sm & jm[None, :], other=0.0).to(tl.float32)
    cs = tl.load(COS + t * H + j, mask=jm, other=0.0).to(tl.float32)
    sn = tl.load(SIN + t * H + j, mask=jm, other=0.0).to(tl.float32)
    ra = ((sa * cs[None, :] - sb * sn[None, :]) * qm[:, None]).to(tl.bfloat16)
    rb = ((sb * cs[None, :] + sa * sn[None, :]) * qm[:, None]).to(tl.bfloat16)

    qb = Q_SUB + row_q * (N * QW) + head[:, None] * QW
    kb = K_SUB + row_k * (N * QW) + head[:, None] * QW
    vm = (cm & ((is_q & (t > 0)) | ((~is_q) & (t + 1 < T))))[:, None]
    base = tl.where(is_q[:, None], qb, kb)

    tl.store(base + k0[None, :], y0, mask=vm)
    if K > 64:
      tl.store(base + k1[None, :], y1, mask=vm & km1[None, :])
    tl.store(base + K + j[None, :], ra, mask=vm & jm[None, :])
    tl.store(base + K + H + j[None, :], rb, mask=vm & jm[None, :])
    if QW > K + 2 * H:
      zp = tl.arange(0, 8)
      tl.store(base + K + 2 * H + zp[None, :], tl.zeros((CRP, 8), tl.bfloat16),
               mask=vm & (zp[None, :] < QW - K - 2 * H))
  else:
    sv = inv * vo_scale
    y0 = (o0 * sv).to(tl.bfloat16)
    is_v = arm == 0

    vb = V_SUB + row_k * (N * QW) + head[:, None] * QW
    ob = O_M + row * (N * K) + head[:, None] * K
    v_valid = cm & is_v & (t + 1 < T)
    o_valid = cm & (~is_v)

    base = tl.where(is_v[:, None], vb, ob)
    vm = (v_valid | o_valid)[:, None]
    tl.store(base + k0[None, :], y0, mask=vm)
    if K > 64:
      y1 = (o1 * sv).to(tl.bfloat16)
      tl.store(base + k1[None, :], y1, mask=vm & km1[None, :])
    if QW > K:
      zt = tl.arange(0, K_TAIL_P)
      tl.store(vb + K + zt[None, :], tl.zeros((CRP, K_TAIL_P), tl.bfloat16),
               mask=v_valid[:, None] & (zt[None, :] < QW - K))


@triton.jit
def _read_fwd_train_kernel(
    M, s_m, SHIFT, s_sh, STD, s_std, UP, s_up, GATES, s_gt, XB, SKEY, C_QK, C_VO, COS, SIN,
    Q_SUB, K_SUB, V_SUB, O_M,
    T, eps, key_eps, key_scale, static_scale, q_mult,
    SUB: tl.constexpr, GROUPS: tl.constexpr,
    N: tl.constexpr, K: tl.constexpr, K1P: tl.constexpr,
    V: tl.constexpr, V0: tl.constexpr, V1: tl.constexpr,
    H: tl.constexpr, HP: tl.constexpr, QW: tl.constexpr, K_TAIL_P: tl.constexpr,
    CR: tl.constexpr, CRP: tl.constexpr,
):
  """Fused MShift + MNorm + 4-arm MRead + RoPE writing to shifted `[B, T-1, N, QW]` & `[rows, N, K]`."""
  ci = tl.arange(0, CRP)
  blk = tl.program_id(1)
  c = (blk % SUB) * CR + ci
  cm = (ci < CR) & (c < 2 * N)
  row = tl.program_id(0)
  seq = row // T
  t = row % T
  g0 = (blk // SUB) * GROUPS

  ra0, ra1 = kl._group_keys(g0, row, c, cm, UP, s_up, GATES, s_gt, XB + V, SKEY, key_eps, key_scale, static_scale, N, V, V0, V1, CRP)
  if GROUPS == 2:
    rb0, rb1 = kl._group_keys(g0 + 1, row, c, cm, UP, s_up, GATES, s_gt, XB + V, SKEY, key_eps, key_scale, static_scale, N, V, V0, V1, CRP)

  pv = 1.0 if t > 0 else 0.0
  pp = M + (row - 1) * s_m if t > 0 else M + row * s_m
  p00, p01, p10, p11 = kl._load_m(pp, K, K1P, V, V0, V1)
  a = tl.arange(0, V0)
  s0 = (pv * tl.sigmoid(
      tl.load(SHIFT + row * s_sh + a, mask=a < V, other=0.0).to(tl.float32)
      + tl.load(XB + a, mask=a < V, other=0.0).to(tl.float32)
  ))[None, :]
  if V1 > 0:
    b = V0 + tl.arange(0, V1)
    s1 = (pv * tl.sigmoid(
        tl.load(SHIFT + row * s_sh + b, mask=b < V, other=0.0).to(tl.float32)
        + tl.load(XB + b, mask=b < V, other=0.0).to(tl.float32)
    ))[None, :]

  m00, m01, m10, m11 = kl._load_m(M + row * s_m, K, K1P, V, V0, V1)
  f00 = m00.to(tl.float32)
  f00 = (f00 + s0 * (p00.to(tl.float32) - f00)).to(tl.bfloat16).to(tl.float32)
  ss = tl.sum(tl.sum(f00 * f00, 1), 0)
  if K > 64:
    f10 = m10.to(tl.float32)
    f10 = (f10 + s0 * (p10.to(tl.float32) - f10)).to(tl.bfloat16).to(tl.float32)
    ss += tl.sum(tl.sum(f10 * f10, 1), 0)
    fb10 = f10.to(tl.bfloat16)
  else:
    fb10 = tl.zeros((K1P, V0), tl.bfloat16)
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
    else:
      fb11 = tl.zeros((K1P, V1), tl.bfloat16)
  else:
    fb01 = tl.zeros((64, 16), tl.bfloat16)
    fb11 = tl.zeros((K1P, 16), tl.bfloat16)
  inv = tl.rsqrt(ss / (K * V) + eps)
  fb00 = f00.to(tl.bfloat16)

  qk_scale = tl.load(C_QK).to(tl.float32)
  vo_scale = tl.load(C_VO).to(tl.float32)

  _group_read_direct(
      g0, row, seq, t, T, c, cm, ra0, ra1, fb00, fb01, fb10, fb11, inv,
      STD, s_std, COS, SIN, Q_SUB, K_SUB, V_SUB, O_M,
      qk_scale, vo_scale, q_mult,
      N, K, K1P, V1, H, HP, QW, K_TAIL_P, CRP,
  )
  if GROUPS == 2:
    _group_read_direct(
        g0 + 1, row, seq, t, T, c, cm, rb0, rb1, fb00, fb01, fb10, fb11, inv,
        STD, s_std, COS, SIN, Q_SUB, K_SUB, V_SUB, O_M,
        qk_scale, vo_scale, q_mult,
        N, K, K1P, V1, H, HP, QW, K_TAIL_P, CRP,
    )


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

      do0_b = (tl.load(base + k0[None, :], mask=vm, other=0.0).to(tl.float32) * qm[:, None]).to(tl.bfloat16)
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
      do0_b = tl.load(base + k0[None, :], mask=vm, other=0.0)
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
  tm1 = max(0, t - 1)
  qw = kl.qk_width(k, s)
  q_sub = m.new_empty((batch, tm1, n, qw))
  k_sub = m.new_empty((batch, tm1, n, qw))
  v_sub = m.new_empty((batch, tm1, n, qw))
  o_m = m.new_empty((rows, n, k))
  v0, v1 = kl._vsplit(v)
  sub = 1 if 2 * n <= 32 else -(-(2 * n) // 32)
  cr = -(-(2 * n) // sub)
  _read_fwd_train_kernel[(rows, sub)](
      m, m.stride(0), shift, shift.stride(0), std, std.stride(0), up, up.stride(0),
      gates, gates.stride(0), xb, static_key, c_qk, c_vo, cos, sin,
      q_sub, k_sub, v_sub, o_m,
      t, eps, key_eps, key_scale, static_scale, q_mult,
      SUB=sub, GROUPS=2,
      N=n, K=k, K1P=kl._ksplit(k), V=v, V0=v0, V1=v1,
      H=s // 2, HP=kl._pow2(s // 2), QW=qw, K_TAIL_P=kl._pow2(max(1, qw - k)),
      CR=cr, CRP=kl._pow2(cr),
      num_warps=2 if (v1 == 0 and cr <= 32) else 4,
  )
  return q_sub, k_sub, v_sub, o_m


@_loom_read_fwd_op.register_fake
def _(
    m, shift, std, up, gates, static_key, xb, c_qk, c_vo, cos, sin,
    batch: int, n: int, k: int, v: int, s: int,
    eps: float, key_eps: float, key_scale: float, static_scale: float, q_mult: float,
):
  rows = m.shape[0]
  tm1 = max(0, rows // batch - 1)
  qw = kl.qk_width(k, s)
  return (
      m.new_empty((batch, tm1, n, qw)),
      m.new_empty((batch, tm1, n, qw)),
      m.new_empty((batch, tm1, n, qw)),
      m.new_empty((rows, n, k)),
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
  tm1 = max(0, t - 1)
  v0, v1 = kl._vsplit(v)
  k1p = kl._ksplit(k)
  sub = 1 if 2 * n <= 32 else -(-(2 * n) // 32)
  cr = -(-(2 * n) // sub)

  if tm1 > 0:
    dq_sub, dk_sub, dv_sub = (_unit_inner(x) for x in (dq_sub, dk_sub, dv_sub))
  do_m = do_m.contiguous()

  dstd = m.new_empty((rows, 2 * n * s))
  dmt_g = m.new_empty((2 * rows, k * v))
  s_buf = torch.empty((rows, v), device=m.device, dtype=torch.float32)
  dm = m.new_empty(m.shape)
  dshift = shift.new_empty(shift.shape)
  dup = up.new_empty(up.shape)
  dgates = gates.new_empty(gates.shape)
  dsk_rows = m.new_empty((rows, 4 * n, v))
  dc_rows = torch.empty((2, rows), device=m.device, dtype=torch.float32)

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
      SUB=sub, N=n, K=k, K1P=k1p, V=v, V0=v0, V1=v1,
      H=s // 2, HP=kl._pow2(s // 2), CR=cr, CRP=kl._pow2(cr),
      num_warps=2 if (v1 == 0 and cr <= 32) else 4,
  )

  r = _MSHIFT_BWD_ROWS
  ch = -(-t // r)
  _mshift_mnorm_bwd_kernel[(batch * ch,)](
      m, m.stride(0), dmt_g, dmt_g.stride(0), s_buf, s_buf.stride(0),
      dm, dm.stride(0), dshift, dshift.stride(0),
      rows, t, ch, eps,
      R=r, K=k, K1P=k1p, V=v, V0=v0, V1=v1,
      num_warps=2 if v1 == 0 else 4,
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
  tm1 = max(0, rows // batch - 1)
  qw = kl.qk_width(k, s)
  if dq_sub is None:
    dq_sub = m.new_zeros((batch, tm1, n, qw))
  if dk_sub is None:
    dk_sub = m.new_zeros((batch, tm1, n, qw))
  if dv_sub is None:
    dv_sub = m.new_zeros((batch, tm1, n, qw))
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Autograd + `torch.compile`-enabled fused Triton MShift + MNorm + 4-arm MRead + RoPE."""
  return _loom_read_fwd_op(
      m, shift, std, up, gates, static_key, xb, c_qk, c_vo, cos, sin,
      batch, n, k, v, s, eps, key_eps, key_scale, static_scale, q_mult,
  )
