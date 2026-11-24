"""Benchmark prefill and CUDA-graph decode throughput for LoomPythia / LoomFormer and Pythia / Transformer baselines.

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 python pytorch/scripts/benchmark_inference.py \
      --model-dir /path/to/LoomPythia-2.8B \
      --compile max-autotune
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM

PYTORCH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PYTORCH_ROOT not in sys.path:
  sys.path.insert(0, PYTORCH_ROOT)

import fast_inference  # noqa: E402


def _sync_time(fn, warmup: int = 3, iters: int = 10) -> float:
  for _ in range(warmup):
    fn()
  torch.cuda.synchronize()
  times = []
  for _ in range(iters):
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    times.append((time.perf_counter() - t0) * 1000.0)
  return float(np.median(times))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model-dir", required=True, help="HF model path or repo id")
  parser.add_argument("--compile", default="max-autotune", choices=["none", "reduce-overhead", "max-autotune"])
  parser.add_argument("--seq-len", type=int, default=2048)
  parser.add_argument("--decode-context", type=int, default=1984)
  parser.add_argument("--decode-batches", default="1,16,64")
  args = parser.parse_args()

  hf = AutoModelForCausalLM.from_pretrained(args.model_dir, trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()
  fast = fast_inference.build(hf)
  vocab = hf.config.vocab_size

  print(f"GPU: {torch.cuda.get_device_name(0)} | Model: {args.model_dir} | compile={args.compile}")
  with torch.no_grad():
    for bs in (1, 8):
      ids = torch.randint(1, vocab, (bs, args.seq_len), device="cuda")
      ms = _sync_time(lambda: fast.prefill(ids, state=None, last_only=True))
      print(f"  prefill {bs:2d} x {args.seq_len}: {ms:7.2f} ms")

    for bs in [int(x) for x in args.decode_batches.split(",") if x.strip()]:
      state = fast.new_state(bs, args.seq_len)
      prompt = torch.randint(1, vocab, (bs, args.decode_context), device="cuda")
      tok = torch.randint(1, vocab, (bs, 1), device="cuda")
      fast.prefill(prompt, state)

      def step_fn(x):
        out = fast.decode(x, state)
        state.pos.fill_(args.decode_context)
        return out

      if args.compile != "none":
        step_fn = torch.compile(step_fn, mode=args.compile, fullgraph=True, dynamic=False)
      ms = _sync_time(lambda: step_fn(tok), warmup=5, iters=20)
      print(f"  decode  bs={bs:2d} @ ctx={args.decode_context}: {ms:7.2f} ms/step")


if __name__ == "__main__":
  main()
