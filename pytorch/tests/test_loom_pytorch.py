"""Unit, numerical parity, KV+M cache decode, and Triton fast-inference tests for LoomFormer & LoomPythia."""
import os
import sys
import unittest

import numpy as np
import torch
import torch.nn.functional as F

PYTORCH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PYTORCH_ROOT not in sys.path:
  sys.path.insert(0, PYTORCH_ROOT)

import fast_inference  # noqa: E402
from loomformer import LoomFormerConfig, LoomFormerForCausalLM  # noqa: E402
from loompythia import LoomPythiaConfig, LoomPythiaForCausalLM  # noqa: E402


def _rel_rms(a: torch.Tensor, b: torch.Tensor) -> float:
  af, bf = a.float(), b.float()
  return float(((af - bf).pow(2).mean().sqrt() / bf.pow(2).mean().sqrt().clamp(min=1e-12)).item())


def _mean_kl(logits_p: torch.Tensor, logits_q: torch.Tensor) -> float:
  lp = F.log_softmax(logits_p.float(), dim=-1)
  lq = F.log_softmax(logits_q.float(), dim=-1)
  return float((lp.exp() * (lp - lq)).sum(dim=-1).mean().item())


def _perturb_nonzero(model: torch.nn.Module, seed: int = 0):
  """Make all zero-initialized gates/static keys nonzero so every circuit branch is exercised."""
  gen = torch.Generator(device="cpu").manual_seed(seed)
  with torch.no_grad():
    for name, p in model.named_parameters():
      if any(tag in name for tag in ("W_rg", "W_sg", "q_bar")):
        p.add_(0.15 * torch.randn(p.shape, generator=gen, dtype=torch.float32).to(device=p.device, dtype=p.dtype))


class LoomPyTorchTest(unittest.TestCase):

  def setUp(self):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

  def _assert_cache_and_padding_invariants(self, model, vocab_size: int):
    device = next(model.parameters()).device
    torch.manual_seed(123)
    ids = torch.randint(1, vocab_size, (2, 24), device=device)

    with torch.no_grad():
      full_logits = model(ids, use_cache=False).logits

      # 1. Incremental token-by-token decode after 16-token prefill
      out_pre = model(ids[:, :16], use_cache=True)
      cache = out_pre.past_key_values
      dec_pieces = [out_pre.logits]
      for t in range(16, 24):
        step_out = model(ids[:, t : t + 1], past_key_values=cache, use_cache=True)
        cache = step_out.past_key_values
        dec_pieces.append(step_out.logits)
      inc_logits = torch.cat(dec_pieces, dim=1)
      self.assertLess(_rel_rms(inc_logits, full_logits), 5e-6, "Incremental KV+M decode mismatch vs full forward")

      # 2. Chunked prefill (8 + 8 + 8)
      c_out = model(ids[:, :8], use_cache=True)
      c_cache = c_out.past_key_values
      c_pieces = [c_out.logits]
      for s in (8, 16):
        c_step = model(ids[:, s : s + 8], past_key_values=c_cache, use_cache=True)
        c_cache = c_step.past_key_values
        c_pieces.append(c_step.logits)
      chunk_logits = torch.cat(c_pieces, dim=1)
      self.assertLess(_rel_rms(chunk_logits, full_logits), 5e-6, "Chunked prefill mismatch vs full forward")

      # 3. Left-padded sequence invariance
      pad_len = 5
      padded_ids = torch.cat([torch.zeros((2, pad_len), dtype=ids.dtype, device=device), ids], dim=1)
      attn_mask = torch.cat([torch.zeros((2, pad_len), dtype=torch.long, device=device),
                             torch.ones_like(ids, dtype=torch.long)], dim=1)
      pad_logits = model(padded_ids, attention_mask=attn_mask, use_cache=False).logits[:, pad_len:]
      self.assertLess(_rel_rms(pad_logits, full_logits), 5e-6, "Left-padding invariance failed")

      # 4. HF `generate()` greedy argmax matches uncached greedy rollout
      gen_ids = model.generate(ids[:, :10], max_new_tokens=6, do_sample=False, use_cache=True)
      cur = ids[:, :10]
      for _ in range(6):
        nxt = model(cur, use_cache=False).logits[:, -1:].argmax(dim=-1)
        cur = torch.cat([cur, nxt], dim=1)
      self.assertTrue(torch.equal(gen_ids, cur), "HF generate() greedy tokens diverged from uncached rollout")

  def test_loompythia_cache_chunk_padding_and_generate_fp32(self):
    cfg = LoomPythiaConfig(
        vocab_size=256, hidden_size=160, num_hidden_layers=3, num_attention_heads=4,
        head_dim=40, d_k=16, d_r=32, rope_dim=12, intermediate_size=320,
        last_ffn_x_dim=448, ffn_m_dim=160,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LoomPythiaForCausalLM(cfg).to(device=device, dtype=torch.float32).eval()
    _perturb_nonzero(model, seed=1)
    self._assert_cache_and_padding_invariants(model, cfg.vocab_size)

  def test_loomformer_variants_fp32(self):
    """Test Llama-backbone LoomFormer (odd head_dim=75, SiTU, and MHA baseline) cache/padding/generate."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    configs = [
        # Odd head_dim=75 (Medium head geometry)
        LoomFormerConfig(
            vocab_size=256, hidden_size=300, num_hidden_layers=3, num_attention_heads=4,
            head_dim=75, d_k=16, d_r=64, rope_dim=18, intermediate_size=512,
            last_ffn_x_dim=768, ffn_m_dim=256, loom_enabled=True,
        ),
        # SiTU-GLU variant
        LoomFormerConfig(
            vocab_size=256, hidden_size=384, num_hidden_layers=3, num_attention_heads=4,
            head_dim=96, d_k=16, d_r=64, rope_dim=24, intermediate_size=512,
            last_ffn_x_dim=768, ffn_m_dim=256, loom_enabled=True, situ_beta1=4.0, situ_beta2=7.0,
        ),
        # Standard Transformer MHA baseline with odd head_dim=75 (`mha_nope_dim=1`)
        LoomFormerConfig(
            vocab_size=256, hidden_size=300, num_hidden_layers=3, num_attention_heads=4,
            head_dim=75, mha_nope_dim=1, intermediate_size=512, loom_enabled=False,
        ),
    ]
    for idx, cfg in enumerate(configs):
      model = LoomFormerForCausalLM(cfg).to(device=device, dtype=torch.float32).eval()
      _perturb_nonzero(model, seed=10 + idx)
      self._assert_cache_and_padding_invariants(model, cfg.vocab_size)

  def test_triton_fast_inference_loompythia_geometry(self):
    """Verify fused Triton `FastLoomPythia` (80x64 geometry) matches `LoomPythiaForCausalLM` on prefill & decode."""
    if not torch.cuda.is_available():
      self.skipTest("CUDA required for Triton fast_inference test.")
    # 3-layer model with exact 2.8B head/slot geometry (head_dim=80, d_k=64, rope_dim=20) so specialized Triton kernels run
    cfg = LoomPythiaConfig(
        vocab_size=512, hidden_size=1280, num_hidden_layers=3, num_attention_heads=16,
        head_dim=80, d_k=64, d_r=256, rope_dim=20, intermediate_size=1024,
        last_ffn_x_dim=1280, ffn_m_dim=512,
    )
    hf = LoomPythiaForCausalLM(cfg).to(device="cuda", dtype=torch.bfloat16).eval()
    _perturb_nonzero(hf, seed=42)
    fast = fast_inference.build(hf)
    self.assertTrue(fast.kernels, "Expected specialized Triton kernels to be active on (80, 64, 20) BF16 geometry")

    for bs in (1, 4):
      ids = torch.randint(1, cfg.vocab_size, (bs, 192), device="cuda")
      with torch.no_grad():
        ref_full = hf(ids, use_cache=False).logits
        state = fast.new_state(bs, 256)
        fast_pre = fast.prefill(ids[:, :128], state, last_only=False)
        self.assertLess(_mean_kl(ref_full[:, :128], fast_pre), 5e-4, f"Prefill KL too high at bs={bs}")

        dec_logits = []
        for t in range(128, 192):
          dec_logits.append(fast.decode(ids[:, t : t + 1], state))
        fast_dec = torch.cat(dec_logits, dim=1)
        self.assertLess(_mean_kl(ref_full[:, 128:], fast_dec), 5e-4, f"Decode KL too high at bs={bs}")

  def test_triton_fast_inference_loomformer_odd_and_even_heads(self):
    """Verify generic fused Triton `FastLoomFormer` on odd `head_dim=75` (Medium) and `head_dim=96` (Large/XL)."""
    if not torch.cuda.is_available():
      self.skipTest("CUDA required for Triton fast_inference test.")
    for hd, dk, rdim in [(75, 32, 18), (96, 40, 24)]:
      n = 8
      cfg = LoomFormerConfig(
          vocab_size=512, hidden_size=n * hd, num_hidden_layers=3, num_attention_heads=n,
          head_dim=hd, d_k=dk, d_r=128, rope_dim=rdim, intermediate_size=512,
          last_ffn_x_dim=768, ffn_m_dim=256, loom_enabled=True,
      )
      hf = LoomFormerForCausalLM(cfg).to(device="cuda", dtype=torch.bfloat16).eval()
      _perturb_nonzero(hf, seed=77 + hd)
      fast = fast_inference.build(hf)
      self.assertTrue(fast.kernels, f"Expected generic Triton kernels active for head_dim={hd}")

      for bs in (1, 4):
        ids = torch.randint(1, cfg.vocab_size, (bs, 160), device="cuda")
        with torch.no_grad():
          ref_full = hf(ids, use_cache=False).logits
          state = fast.new_state(bs, 256)
          fast_pre = fast.prefill(ids[:, :128], state, last_only=False)
          self.assertLess(_mean_kl(ref_full[:, :128], fast_pre), 5e-4, f"LoomFormer hd={hd} prefill KL")
          dec_logits = [fast.decode(ids[:, t : t + 1], state) for t in range(128, 160)]
          fast_dec = torch.cat(dec_logits, dim=1)
          self.assertLess(_mean_kl(ref_full[:, 128:], fast_dec), 5e-4, f"LoomFormer hd={hd} decode KL")


if __name__ == "__main__":
  unittest.main()
