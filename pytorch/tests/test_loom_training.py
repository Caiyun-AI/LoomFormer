"""Unit & integration tests for LoomLlama PyTorch training, Triton fwd+bwd kernels, and `fast_inference` alignment."""
import math
import os
import sys
import tempfile
import unittest

import numpy as np
import torch
import torch.nn.functional as F

PYTORCH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PYTORCH_ROOT not in sys.path:
  sys.path.insert(0, PYTORCH_ROOT)

import fast_inference  # noqa: E402
from loomformer import LoomFormerConfig, LoomFormerForCausalLM  # noqa: E402
from train_loom import build_optimizer, verify_inference_alignment  # noqa: E402
from training.configs import (  # noqa: E402
    build_loomllama_160m_config,
    build_loomllama_medium_config,
    count_parameters,
)
from training.data import FINEWEB_MAGIC, FINEWEB_VERSION, FineWebShardLoader, HEADER_INT32  # noqa: E402
from training.engine import build_train_model  # noqa: E402


def _mean_kl(p: torch.Tensor, q: torch.Tensor) -> float:
  lp, lq = F.log_softmax(p.float(), dim=-1), F.log_softmax(q.float(), dim=-1)
  return float((lp.exp() * (lp - lq)).sum(dim=-1).mean().item())


class LoomTrainingTest(unittest.TestCase):

  def test_jax_aligned_initialization_and_wd_mask(self):
    """Verify JAX-aligned gate/q_bar/clone_vo initialization and AdamW weight-decay grouping."""
    cfg = build_loomllama_160m_config(loom_enabled=True, vocab_size=512, num_hidden_layers=3)
    model = LoomFormerForCausalLM(cfg)
    mha0 = model.model.layers[0].loom_mha

    self.assertEqual(mha0.W_sg.weight.abs().max().item(), 0.0)
    self.assertAlmostEqual(mha0.W_sg.bias.mean().item(), 0.0, places=5)
    self.assertAlmostEqual(model.model.embed_write.W_wg.bias.mean().item(), math.log(0.1 / 0.9), places=5)
    self.assertAlmostEqual(mha0.W_wg.bias.mean().item(), math.log(0.1 / 0.9), places=5)
    self.assertAlmostEqual(model.model.layers[0].ffn_W_wg.bias.mean().item(), math.log(0.1 / 0.9), places=5)
    for arm in ("q", "k", "o", "v"):
      self.assertEqual(getattr(mha0, f"W_rg_{arm}").weight.abs().max().item(), 0.0)
      self.assertAlmostEqual(getattr(mha0, f"W_rg_{arm}").bias.mean().item(), math.log(0.05 / 0.95), places=5)
      self.assertEqual(getattr(mha0, f"W_rg_bar_{arm}").weight.abs().max().item(), 0.0)
      self.assertAlmostEqual(getattr(mha0, f"W_rg_bar_{arm}").bias.mean().item(), math.log(0.99 / 0.11), places=5)
    self.assertGreater(mha0.q_bar_v.std().item(), 0.1)
    self.assertEqual(mha0.q_bar_q.abs().max().item(), 0.0)
    self.assertTrue(torch.equal(mha0.W_r2_v.weight, mha0.W_r2_o.weight))

    opt = build_optimizer(model, lr=1e-3, weight_decay=0.1)
    no_decay_ids = {id(p) for p in opt.param_groups[1]["params"]}
    decay_ids = {id(p) for p in opt.param_groups[0]["params"]}
    self.assertIn(id(mha0.c_qk), no_decay_ids)
    self.assertIn(id(mha0.c_vo), no_decay_ids)
    self.assertIn(id(mha0.W_sg.bias), no_decay_ids)
    self.assertIn(id(model.model.layers[0].ln_1.weight), no_decay_ids)
    self.assertIn(id(mha0.q_bar_v), decay_ids)
    self.assertIn(id(mha0.W_o.weight), decay_ids)

  def test_triton_train_grad_parity_and_inference_alignment(self):
    """Verify full LoomLlama & Llama fwd+bwd parity (Triton vs PyTorch ref) and `fast_inference` alignment."""
    if not torch.cuda.is_available():
      self.skipTest("CUDA required for Triton training test.")

    for loom_on in (True, False):
      cfg = build_loomllama_160m_config(loom_enabled=loom_on, vocab_size=1024, num_hidden_layers=4)
      torch.manual_seed(123 + int(loom_on))
      model = LoomFormerForCausalLM(cfg).cuda()
      with torch.no_grad():
        for name, p in model.named_parameters():
          if any(tag in name for tag in ("W_rg", "W_sg", "q_bar")):
            p.add_(0.05 * torch.randn_like(p))

      ids = torch.randint(1, cfg.vocab_size, (2, 128), device="cuda")
      targets = torch.randint(1, cfg.vocab_size, (2, 128), device="cuda")

      # Reference forward + backward (pure PyTorch HF model)
      model.zero_grad()
      with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits_ref = model(ids, use_cache=False).logits
        loss_ref = F.cross_entropy(logits_ref.flatten(0, 1), targets.flatten())
      loss_ref.backward()
      grads_ref = {k: p.grad.detach().clone() for k, p in model.named_parameters() if p.grad is not None}

      # Triton accelerated forward + backward (`build_train_model` sharing live parameters)
      train_model = build_train_model(model)
      model.zero_grad()
      with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits_tri = train_model(ids)
        loss_tri = F.cross_entropy(logits_tri.flatten(0, 1), targets.flatten())
      loss_tri.backward()
      grads_tri = {k: p.grad.detach().clone() for k, p in model.named_parameters() if p.grad is not None}

      self.assertLess(abs(loss_ref.item() - loss_tri.item()), 2.5e-3)
      self.assertLess(_mean_kl(logits_ref, logits_tri), 1e-4)
      self.assertEqual(set(grads_ref.keys()), set(grads_tri.keys()))
      for k in grads_ref:
        a, b = grads_tri[k].float().flatten(), grads_ref[k].float().flatten()
        self.assertTrue(torch.isfinite(a).all(), f"Non-finite grad in {k}")
        if b.norm().item() > 1e-5 and b.numel() > 1:
          cos = F.cosine_similarity(a[None], b[None]).item()
          self.assertGreater(cos, 0.99, f"Low grad cosine {cos:.5f} in {k} (loom={loom_on})")

      # Verify alignment with `fast_inference.build(...)`
      align = verify_inference_alignment(model, seq_len=128, prefill_len=96, batch_size=2)
      if loom_on:
        self.assertEqual(align["fast_kernels_active"], 1.0, "Expected fast_inference Triton kernels active")
      self.assertLess(align["kl_train_vs_fast_full"], 5e-4, f"Alignment KL too high (loom={loom_on}): {align}")
      self.assertLess(align["kl_hf_ref_vs_fast_full"], 5e-4, f"HF ref vs fast KL too high (loom={loom_on}): {align}")

  def test_fineweb_shard_loader(self):
    """Verify binary shard header validation and DDP rank slicing in `FineWebShardLoader`."""
    with tempfile.TemporaryDirectory() as tmp:
      shard_path = os.path.join(tmp, "fineweb_train_000001.bin")
      num_toks = 4096
      header = np.zeros(HEADER_INT32, dtype=np.int32)
      header[0] = FINEWEB_MAGIC
      header[1] = FINEWEB_VERSION
      header[2] = num_toks
      toks = np.arange(num_toks, dtype=np.uint16)
      with open(shard_path, "wb") as f:
        f.write(header.tobytes())
        f.write(toks.tobytes())

      l0 = FineWebShardLoader([shard_path], micro_batch_size=2, seq_len=64, rank=0, world_size=2, device="cpu")
      l1 = FineWebShardLoader([shard_path], micro_batch_size=2, seq_len=64, rank=1, world_size=2, device="cpu")
      x0, y0 = l0.next_batch()
      x1, y1 = l1.next_batch()
      self.assertEqual(tuple(x0.shape), (2, 64))
      self.assertTrue(torch.equal(y0[:, :-1], x0[:, 1:]))
      # Rank 0 covers [0..127], Rank 1 covers [128..255]
      self.assertEqual(int(x0[0, 0]), 0)
      self.assertEqual(int(x1[0, 0]), 128)
      # Next step advances by world_size * B * T = 256
      x0_next, _ = l0.next_batch()
      self.assertEqual(int(x0_next[0, 0]), 256)

  def test_training_steps_decrease_loss_and_align_after_updates(self):
    """Run 10 real AdamW steps on synthetic data for both LoomLlama and Llama and verify loss drops & inference aligns."""
    if not torch.cuda.is_available():
      self.skipTest("CUDA required for training smoke test.")

    for loom_on in (True, False):
      cfg = build_loomllama_160m_config(loom_enabled=loom_on, vocab_size=512, num_hidden_layers=3)
      torch.manual_seed(7)
      model = LoomFormerForCausalLM(cfg).cuda()
      train_model = build_train_model(model)
      opt = build_optimizer(model, lr=1e-3, weight_decay=0.1)

      x = torch.randint(1, cfg.vocab_size, (4, 128), device="cuda")
      y = torch.roll(x, -1, dims=1)
      losses = []
      for _ in range(10):
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
          logits = train_model(x)
          loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.item()))

      self.assertLess(losses[-1], losses[0], f"Loss failed to decrease (loom={loom_on}): {losses}")
      align = verify_inference_alignment(model, seq_len=128, prefill_len=96, batch_size=2)
      self.assertLess(align["kl_train_vs_fast_full"], 5e-4, f"Post-train alignment failed (loom={loom_on}): {align}")

      # Verify HF save_pretrained -> from_pretrained preserves exact trained weights (no re-init in transformers 5.x)
      with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        reloaded = LoomFormerForCausalLM.from_pretrained(tmp).cuda().eval()
        for name, p in model.named_parameters():
          self.assertTrue(
              torch.equal(p.data, reloaded.get_parameter(name).data),
              f"from_pretrained altered parameter {name} (loom={loom_on})",
          )
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
          l_orig = F.cross_entropy(train_model.eval()(x).flatten(0, 1), y.flatten()).item()
          l_load = F.cross_entropy(build_train_model(reloaded).eval()(x).flatten(0, 1), y.flatten()).item()
        self.assertAlmostEqual(l_orig, l_load, places=5, msg=f"from_pretrained loss mismatch (loom={loom_on})")

  def test_torch_compile_triton_zero_breaks_and_parity(self):
    """Verify `torch.compile` traces `FastTrainModel` with 0 graph breaks and matches eager grads."""
    if not torch.cuda.is_available():
      self.skipTest("CUDA required for torch.compile test.")

    for loom_on in (True, False):
      torch._dynamo.reset()
      cfg = build_loomllama_160m_config(loom_enabled=loom_on, vocab_size=512, num_hidden_layers=3)
      torch.manual_seed(99 + int(loom_on))
      model = LoomFormerForCausalLM(cfg).cuda().train()
      train_model = build_train_model(model).train()
      ids = torch.randint(1, cfg.vocab_size, (2, 128), device="cuda")
      targets = torch.randint(1, cfg.vocab_size, (2, 128), device="cuda")

      def _step(x, y):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
          logits = train_model(x)
          return F.cross_entropy(logits.flatten(0, 1), y.flatten())

      exp = torch._dynamo.explain(_step)(ids, targets)
      self.assertEqual(exp.graph_break_count, 0, f"Unexpected Dynamo graph breaks (loom={loom_on}): {exp.break_reasons}")

      model.zero_grad(set_to_none=True)
      loss_eager = _step(ids, targets)
      loss_eager.backward()
      g_eager = {k: p.grad.detach().clone().float() for k, p in model.named_parameters() if p.grad is not None}

      compiled_model = torch.compile(train_model)
      compiled_ce = torch.compile(lambda lg, tg: F.cross_entropy(lg.flatten(0, 1), tg.flatten()))
      model.zero_grad(set_to_none=True)
      with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits_c = compiled_model(ids)
        loss_comp = compiled_ce(logits_c, targets)
      loss_comp.backward()
      g_comp = {k: p.grad.detach().clone().float() for k, p in model.named_parameters() if p.grad is not None}

      self.assertLess(abs(loss_eager.item() - loss_comp.item()), 2.5e-3)
      self.assertEqual(set(g_eager.keys()), set(g_comp.keys()))
      for k in g_eager:
        a, b = g_comp[k].flatten(), g_eager[k].flatten()
        self.assertTrue(torch.isfinite(a).all(), f"Non-finite compiled grad in {k}")
        if b.norm().item() > 1e-5 and b.numel() > 1:
          cos = F.cosine_similarity(a[None], b[None]).item()
          self.assertGreater(cos, 0.99, f"Low compiled grad cosine {cos:.5f} in {k} (loom={loom_on})")

  def test_160m_head_dim_64_triton_compile_and_inference(self):
    """Verify 162M config (head_dim=64, d_k=24, rope_dim=16) param match, Triton fwd/bwd, compile & fast_inference."""
    cfg_loom = build_loomllama_160m_config(loom_enabled=True)
    cfg_llama = build_loomllama_160m_config(loom_enabled=False)
    with torch.device("meta"):
      c_loom = count_parameters(LoomFormerForCausalLM(cfg_loom))["total"]
      c_llama = count_parameters(LoomFormerForCausalLM(cfg_llama))["total"]
    self.assertEqual(c_llama, 162_220_800)
    self.assertEqual(c_loom, 162_223_020)
    self.assertLess(abs(c_loom - c_llama) / c_llama, 1e-4)

    if not torch.cuda.is_available():
      self.skipTest("CUDA required for head_dim=64 Triton test.")

    torch._dynamo.reset()
    cfg = build_loomllama_160m_config(loom_enabled=True, vocab_size=512, num_hidden_layers=3)
    torch.manual_seed(2026)
    model = LoomFormerForCausalLM(cfg).cuda().train()
    train_model = build_train_model(model).train()
    with torch.no_grad():
      for name, p in model.named_parameters():
        if any(tag in name for tag in ("W_rg", "W_sg", "q_bar")):
          p.add_(0.05 * torch.randn_like(p))

    ids = torch.randint(1, cfg.vocab_size, (2, 128), device="cuda")
    targets = torch.randint(1, cfg.vocab_size, (2, 128), device="cuda")

    # 1. Pure PyTorch reference fwd + bwd
    model.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
      logits_ref = model(ids, use_cache=False).logits
      loss_ref = F.cross_entropy(logits_ref.flatten(0, 1), targets.flatten())
    loss_ref.backward()
    grads_ref = {k: p.grad.detach().clone().float() for k, p in model.named_parameters() if p.grad is not None}

    # 2. Triton head_dim=64 fwd + bwd
    model.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
      logits_tri = train_model(ids)
      loss_tri = F.cross_entropy(logits_tri.flatten(0, 1), targets.flatten())
    loss_tri.backward()
    grads_tri = {k: p.grad.detach().clone().float() for k, p in model.named_parameters() if p.grad is not None}

    self.assertLess(abs(loss_ref.item() - loss_tri.item()), 2.5e-3)
    self.assertLess(_mean_kl(logits_ref, logits_tri), 1e-4)
    self.assertEqual(set(grads_ref.keys()), set(grads_tri.keys()))
    for k in grads_ref:
      a, b = grads_tri[k].flatten(), grads_ref[k].flatten()
      self.assertTrue(torch.isfinite(a).all(), f"Non-finite grad in {k}")
      if b.norm().item() > 1e-5 and b.numel() > 1:
        cos = F.cosine_similarity(a[None], b[None]).item()
        self.assertGreater(cos, 0.99, f"Low grad cosine {cos:.5f} in {k}")

    # 3. torch.compile 0 graph breaks + parity
    def _step(x, y):
      with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        return F.cross_entropy(train_model(x).flatten(0, 1), y.flatten())

    exp = torch._dynamo.explain(_step)(ids, targets)
    self.assertEqual(exp.graph_break_count, 0, f"Unexpected Dynamo breaks: {exp.break_reasons}")

    compiled_model = torch.compile(train_model)
    model.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
      loss_comp = F.cross_entropy(compiled_model(ids).flatten(0, 1), targets.flatten())
    loss_comp.backward()
    self.assertLess(abs(loss_tri.item() - loss_comp.item()), 2.5e-3)

    # 4. fast_inference Triton prefill + decode alignment (`fast_kernels_active == 1.0`)
    align = verify_inference_alignment(model, seq_len=128, prefill_len=96, batch_size=2)
    self.assertEqual(align["fast_kernels_active"], 1.0, "Expected fast_inference Triton kernels active for head_dim=64")
    self.assertLess(align["kl_train_vs_fast_full"], 5e-4, f"Alignment KL too high: {align}")
    self.assertLess(align["kl_hf_ref_vs_fast_full"], 5e-4, f"HF ref vs fast KL too high: {align}")

  def test_medium_head_dim_75_triton_compile_and_inference(self):
    """Verify paper Medium (~432M, odd head_dim=75) param match, Triton fwd/bwd, compile & fast_inference."""
    with torch.device("meta"):
      c_loom = count_parameters(LoomFormerForCausalLM(build_loomllama_medium_config(loom_enabled=True)))["total"]
      c_llama = count_parameters(LoomFormerForCausalLM(build_loomllama_medium_config(loom_enabled=False)))["total"]
    self.assertEqual(c_llama, 432_121_200)
    self.assertEqual(c_loom, 432_112_228)
    self.assertLess(abs(c_loom - c_llama) / c_llama, 1e-4)

    if not torch.cuda.is_available():
      self.skipTest("CUDA required for head_dim=75 Triton test.")

    for loom_on in (True, False):
      torch._dynamo.reset()
      cfg = build_loomllama_medium_config(loom_enabled=loom_on, vocab_size=512, num_hidden_layers=2)
      torch.manual_seed(75 + int(loom_on))
      model = LoomFormerForCausalLM(cfg).cuda().train()
      train_model = build_train_model(model).train()
      ids = torch.randint(1, cfg.vocab_size, (2, 128), device="cuda")
      targets = torch.randint(1, cfg.vocab_size, (2, 128), device="cuda")

      compiled = torch.compile(train_model)
      model.zero_grad(set_to_none=True)
      with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        loss = F.cross_entropy(compiled(ids).flatten(0, 1), targets.flatten())
      loss.backward()
      for k, p in model.named_parameters():
        if p.grad is not None:
          self.assertTrue(torch.isfinite(p.grad).all(), f"Non-finite compiled grad in {k} (loom={loom_on})")

      align = verify_inference_alignment(model, seq_len=128, prefill_len=96, batch_size=2)
      if loom_on:
        self.assertEqual(align["fast_kernels_active"], 1.0)
      self.assertLess(align["kl_train_vs_fast_full"], 5e-4, f"Medium alignment KL too high (loom={loom_on}): {align}")

  def test_grad_checkpoint_compile_parity(self):
    """Verify `grad_checkpoint=True` matches `grad_checkpoint=False` grads with 0 Dynamo breaks."""
    if not torch.cuda.is_available():
      self.skipTest("CUDA required for grad_checkpoint test.")

    for loom_on in (True, False):
      torch._dynamo.reset()
      cfg = build_loomllama_160m_config(loom_enabled=loom_on, vocab_size=512, num_hidden_layers=3)
      torch.manual_seed(321 + int(loom_on))
      model = LoomFormerForCausalLM(cfg).cuda().train()
      ids = torch.randint(1, cfg.vocab_size, (2, 128), device="cuda")
      targets = torch.randint(1, cfg.vocab_size, (2, 128), device="cuda")

      tm_no_ckpt = build_train_model(model, grad_checkpoint=False).train()
      model.zero_grad(set_to_none=True)
      with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        loss_ref = F.cross_entropy(tm_no_ckpt(ids).flatten(0, 1), targets.flatten())
      loss_ref.backward()
      g_ref = {k: p.grad.detach().clone().float() for k, p in model.named_parameters() if p.grad is not None}

      tm_ckpt = build_train_model(model, grad_checkpoint=True).train()
      def _step(x, y):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
          return F.cross_entropy(tm_ckpt(x).flatten(0, 1), y.flatten())

      exp = torch._dynamo.explain(_step)(ids, targets)
      self.assertEqual(exp.graph_break_count, 0, f"Unexpected Dynamo breaks with grad_ckpt (loom={loom_on}): {exp.break_reasons}")

      compiled = torch.compile(tm_ckpt)
      model.zero_grad(set_to_none=True)
      with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        loss_ckpt = F.cross_entropy(compiled(ids).flatten(0, 1), targets.flatten())
      loss_ckpt.backward()
      g_ckpt = {k: p.grad.detach().clone().float() for k, p in model.named_parameters() if p.grad is not None}

      self.assertLess(abs(loss_ref.item() - loss_ckpt.item()), 2.5e-3)
      for k in g_ref:
        a, b = g_ckpt[k].flatten(), g_ref[k].flatten()
        self.assertTrue(torch.isfinite(a).all(), f"Non-finite grad in {k}")
        if b.norm().item() > 1e-5 and b.numel() > 1:
          cos = F.cosine_similarity(a[None], b[None]).item()
          self.assertGreater(cos, 0.995, f"Low grad_ckpt cosine {cos:.5f} in {k} (loom={loom_on})")


if __name__ == "__main__":
  unittest.main()
