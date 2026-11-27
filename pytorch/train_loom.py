"""PyTorch training entrypoint for LoomFormer (`--loom`) and parameter-matched Llama (`--no-loom`).

Supports single-GPU and multi-GPU DDP (`torchrun`), BF16 mixed-precision training with FP32 master weights,
the fused Triton training engine (`pytorch/training/engine.py` + `pytorch/training/triton_ops.py`, reusing
`fast_inference.kernels_llama`), `torch.compile`, JAX-aligned AdamW weight-decay masking & initialization,
FineWeb10B (`modded-nanogpt`) data, HuggingFace checkpoint saving, and post-training `fast_inference` alignment.

Built-in YAML configs (`pytorch/configs/`, passed via `--config`):
  - `loomformer_160m.yml`   (~162.2M, D=768,  L=12, n=12, d=64, d_k=24, d_r=128, r=16)
  - `loomformer_medium.yml` (~432.1M, D=1200, L=18, n=16, d=75, d_k=32, d_r=256, r=18)
  - `loomformer_large.yml`  (~777.8M, D=1536, L=22, n=16, d=96, d_k=32, d_r=256, r=24)
  - `loomformer_xl.yml`     (~1.432B, D=1920, L=28, n=20, d=96, d_k=40, d_r=400, r=24)
  - `loomformer_2p8b.yml`   (~2.774B, D=2560, L=32, n=32, d=80, d_k=64, d_r=640, r=20)

Examples:
  # 1. Train any scale directly from its YAML config (8 GPUs):
  torchrun --standalone --nproc_per_node=8 pytorch/train_loom.py --config pytorch/configs/loomformer_160m.yml

  # 2. Parameter-matched Llama baseline (any CLI flag overrides the YAML):
  torchrun --standalone --nproc_per_node=8 pytorch/train_loom.py --config pytorch/configs/loomformer_160m.yml --no-loom \
      --out-dir pytorch/runs/llama_162m_2b_seq4096_b128

  # 3. Override any hyperparameter on the fly (e.g. FA2 HEADDIM=96 alignment on 2.8B):
  torchrun --standalone --nproc_per_node=8 pytorch/train_loom.py --config pytorch/configs/loomformer_2p8b.yml --rope-dim 16
"""
import argparse
from contextlib import nullcontext
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import yaml

PYTORCH_ROOT = os.path.dirname(os.path.abspath(__file__))
if PYTORCH_ROOT not in sys.path:
  sys.path.insert(0, PYTORCH_ROOT)

import fast_inference  # noqa: E402
from loomformer import LoomFormerForCausalLM  # noqa: E402
from training.configs import MODEL_PRESETS, build_preset_config, count_parameters  # noqa: E402
from training.data import FineWebShardLoader, download_fineweb10b_shards  # noqa: E402
from training.engine import build_train_model  # noqa: E402


# ==============================================================================
# Optimizer, Weight-Decay Mask, and LR Schedule (aligned with `jax/configs/*.yml`)
# ==============================================================================

NO_DECAY_TAGS = (".bias", "c_qk", "c_vo")


def build_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float,
    beta1: float = 0.9,
    beta2: float = 0.95,
    eps: float = 1e-8,
) -> torch.optim.AdamW:
  """JAX-aligned AdamW matching `adamw_mask` (no weight decay on 1D params, biases, `c_qk`, `c_vo`)."""
  decay_params, no_decay_params = [], []
  for name, p in model.named_parameters():
    if not p.requires_grad:
      continue
    if p.ndim < 2 or any(name.endswith(tag) for tag in NO_DECAY_TAGS):
      no_decay_params.append(p)
    else:
      decay_params.append(p)

  groups = [
      {"params": decay_params, "weight_decay": weight_decay},
      {"params": no_decay_params, "weight_decay": 0.0},
  ]
  fused = torch.cuda.is_available() and all(p.is_cuda for p in decay_params + no_decay_params)
  return torch.optim.AdamW(groups, lr=lr, betas=(beta1, beta2), eps=eps, fused=fused)


def cosine_lr(step: int, total_steps: int, max_lr: float, min_lr_ratio: float, warmup_steps: int) -> float:
  if warmup_steps > 0 and step < warmup_steps:
    return max_lr * (step + 1) / warmup_steps
  if step >= total_steps:
    return max_lr * min_lr_ratio
  progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
  cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
  return max_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


# ==============================================================================
# Evaluation & `fast_inference` Alignment Verification
# ==============================================================================

def _cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
  return F.cross_entropy(logits.flatten(0, 1), targets.flatten())


@torch.no_grad()
def evaluate_val(
    model: torch.nn.Module,
    val_loader: FineWebShardLoader,
    num_val_batches: int,
    amp_ctx,
    ddp: bool,
    loss_fn=_cross_entropy_loss,
) -> Tuple[float, float]:
  """Compute mean cross-entropy loss and perplexity over `num_val_batches` global val batches."""
  model.eval()
  val_loader.reset()
  loss_sum = torch.zeros((), device=val_loader.device, dtype=torch.float64)
  for _ in range(num_val_batches):
    x, y = val_loader.next_batch()
    with amp_ctx():
      logits = model(x)
      loss = loss_fn(logits, y)
    loss_sum += loss.double()
  if ddp:
    dist.all_reduce(loss_sum, op=dist.ReduceOp.AVG)
  mean_loss = float((loss_sum / num_val_batches).item())
  model.train()
  return mean_loss, math.exp(min(mean_loss, 20.0))


@torch.no_grad()
def verify_inference_alignment(
    raw_model: LoomFormerForCausalLM,
    seq_len: int = 128,
    prefill_len: int = 96,
    batch_size: int = 2,
) -> Dict[str, float]:
  """Verify `build_train_model(raw_model)` matches pure HF reference AND `fast_inference.build(model)` (prefill + decode)."""
  device = next(raw_model.parameters()).device
  was_training = raw_model.training
  raw_model.eval()

  # Clone weights into BF16 HF model for inference parity test (leaves FP32 training weights untouched)
  hf_bf16 = LoomFormerForCausalLM(raw_model.config).to(device=device, dtype=torch.bfloat16).eval()
  hf_bf16.load_state_dict({k: v.detach().to(torch.bfloat16) for k, v in raw_model.state_dict().items()})

  torch.manual_seed(2026)
  ids = torch.randint(1, min(raw_model.config.vocab_size, 50257), (batch_size, seq_len), device=device)

  def _mean_kl(p: torch.Tensor, q: torch.Tensor) -> float:
    lp, lq = F.log_softmax(p.float(), dim=-1), F.log_softmax(q.float(), dim=-1)
    return float((lp.exp() * (lp - lq)).sum(dim=-1).mean().item())

  # 1. Pure PyTorch HF reference forward
  logits_hf_ref = hf_bf16(ids, use_cache=False).logits

  # 2. Training Triton engine forward path (under BF16 autocast on FP32 master weights)
  train_engine = build_train_model(raw_model).eval()
  with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    logits_train_triton = train_engine(ids)

  # 3. fast_inference Triton engine (prefill + autoregressive decode)
  fast = fast_inference.build(hf_bf16)
  state = fast.new_state(batch_size, seq_len + 16)
  logits_fast_pre = fast.prefill(ids[:, :prefill_len], state, last_only=False)
  dec_steps = [fast.decode(ids[:, t : t + 1], state) for t in range(prefill_len, seq_len)]
  logits_fast_full = torch.cat([logits_fast_pre] + dec_steps, dim=1)

  metrics = {
      "fast_kernels_active": float(getattr(fast, "kernels", False)),
      "kl_train_vs_hf_ref": _mean_kl(logits_train_triton, logits_hf_ref),
      "kl_train_vs_fast_prefill": _mean_kl(logits_train_triton[:, :prefill_len], logits_fast_pre),
      "kl_train_vs_fast_full": _mean_kl(logits_train_triton, logits_fast_full),
      "kl_hf_ref_vs_fast_full": _mean_kl(logits_hf_ref, logits_fast_full),
      "max_abs_diff_prefill": float((logits_train_triton[:, :prefill_len] - logits_fast_pre).abs().max().item()),
  }

  if was_training:
    raw_model.train()
  del hf_bf16, train_engine, fast, state
  return metrics


# ==============================================================================
# Main Training Loop
# ==============================================================================

def _load_yaml_config(path: str) -> Dict:
  """Load a flat YAML training config into an argparse-compatible dict."""
  with open(path, "r", encoding="utf-8") as f:
    raw = yaml.safe_load(f)
  if raw is None:
    return {}
  if not isinstance(raw, dict):
    raise ValueError(f"Expected a YAML mapping in {path!r}, got {type(raw).__name__}")
  return {str(k).replace("-", "_"): v for k, v in raw.items()}


def _cli_has_flag(argv: List[str], *flags: str) -> bool:
  return any(tok in flags or any(tok.startswith(f + "=") for f in flags) for tok in argv)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
  raw_argv = list(sys.argv[1:] if argv is None else argv)

  # Stage 1: parse optional `--config` / `-c` first so YAML sets defaults and CLI flags always override YAML.
  cfg_pre = argparse.ArgumentParser(add_help=False)
  cfg_pre.add_argument("-c", "--config", type=str, default="", help="Path to YAML training config (e.g. pytorch/configs/loomformer_160m.yml)")
  pre_args, _ = cfg_pre.parse_known_args(raw_argv)

  p = argparse.ArgumentParser(
      description="Train LoomFormer or parameter-matched Llama on FineWeb10B (via `--config` YAML or CLI flags).",
      parents=[cfg_pre],
  )
  # Architecture switches
  loom_grp = p.add_mutually_exclusive_group()
  loom_grp.add_argument("--loom", dest="loom_enabled", action="store_true", help="Enable Loom Attention (LoomFormer)")
  loom_grp.add_argument("--no-loom", dest="loom_enabled", action="store_false", help="Disable Loom Attention (parameter-matched Llama baseline)")
  p.set_defaults(loom_enabled=True)
  p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False, help="Enable torch.compile on model and fused cross-entropy loss")
  p.add_argument("--grad-checkpoint", action=argparse.BooleanOptionalAction, default=False, help="Enable full per-layer activation checkpointing (matches JAX remat_policy=full)")
  p.add_argument(
      "--compile-mode",
      type=str,
      default="default",
      choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
      help="torch.compile optimization mode (default: 'default')",
  )

  # Model preset & optional overrides (0 = inherit from --preset)
  p.add_argument("--preset", type=str, default="160m", choices=sorted(MODEL_PRESETS),
                 help="Parameter-matched model preset: 160m (~162.2M), medium (~432.1M), large (~777.8M), xl (~1.43B), 2.8b (~2.77B)")
  p.add_argument("--vocab-size", type=int, default=0, help="Override vocab_size (0 = use preset)")
  p.add_argument("--hidden-size", type=int, default=0, help="Override hidden_size (0 = use preset)")
  p.add_argument("--num-layers", type=int, default=0, help="Override num_hidden_layers (0 = use preset)")
  p.add_argument("--num-heads", type=int, default=0, help="Override num_attention_heads (0 = use preset)")
  p.add_argument("--head-dim", type=int, default=0, help="Per-head dim (0 = infer hidden_size // num_heads; >0 asserts hidden_size == num_heads * head_dim)")
  p.add_argument("--intermediate-size", type=int, default=0, help="Override FFN intermediate_size (0 = use preset)")
  p.add_argument("--ffn-m-dim", type=int, default=0, help="Override Loom FFN_M width (0 = use preset)")
  p.add_argument("--last-ffn-x-dim", type=int, default=0, help="Override last-layer FFN_X width (0 = use preset)")
  p.add_argument("--d-k", type=int, default=0, help="Override Loom memory width d_k (0 = use preset)")
  p.add_argument("--d-r", type=int, default=0, help="Override Loom read bottleneck d_r (0 = use preset)")
  p.add_argument("--rope-dim", type=int, default=0, help="Override Loom RoPE width r (0 = use preset)")
  p.add_argument("--situ-beta1", type=float, default=0.0)
  p.add_argument("--situ-beta2", type=float, default=0.0)
  p.add_argument("--init-std", type=float, default=0.0, help="Override initializer_range (0 = use preset)")

  # Data (FineWeb10B from modded-nanogpt)
  p.add_argument("--data-dir", type=str, default=os.path.join(PYTORCH_ROOT, "data", "fineweb10B"))
  p.add_argument("--num-train-shards", type=int, default=2, help="Number of 100M-token training shards to download/use")
  p.add_argument("--skip-download", action=argparse.BooleanOptionalAction, default=False, help="Assume FineWeb shards already exist in --data-dir")

  # Training hyperparameters
  p.add_argument("--steps", type=int, default=1500, help="Total optimizer steps")
  p.add_argument("--batch-size", type=int, default=64, help="Global batch size in sequences")
  p.add_argument("--micro-batch-size", type=int, default=8, help="Per-GPU micro-batch size in sequences")
  p.add_argument("--grad-accum", type=int, default=0,
                 help="Gradient accumulation steps (0 = auto-compute batch_size // (micro_batch_size * world_size); "
                      ">0 asserts batch_size == micro_batch_size * world_size * grad_accum)")
  p.add_argument("--seq-len", type=int, default=1024, help="Sequence length in tokens")
  p.add_argument("--lr", type=float, default=6e-4, help="Peak AdamW learning rate")
  p.add_argument("--min-lr-ratio", type=float, default=0.1, help="Final cosine LR fraction of peak LR")
  p.add_argument("--warmup-steps", type=int, default=50, help="Linear warmup steps")
  p.add_argument("--weight-decay", type=float, default=0.1)
  p.add_argument("--beta1", type=float, default=0.9)
  p.add_argument("--beta2", type=float, default=0.95)
  p.add_argument("--grad-clip", type=float, default=1.0)
  p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float32"])
  p.add_argument("--seed", type=int, default=42)

  # Logging, evaluation & saving
  p.add_argument("--log-every", type=int, default=25)
  p.add_argument("--eval-every", type=int, default=250)
  p.add_argument("--val-tokens", type=int, default=2_097_152, help="Tokens per intermediate val eval (default 2M; modded-nanogpt full val is 10,485,760)")
  p.add_argument("--final-val-tokens", type=int, default=10_485_760, help="Tokens for final val eval (10,485,760 matches modded-nanogpt)")
  p.add_argument("--out-dir", type=str, default="", help="Optional output directory to save metrics and final HF checkpoint")
  p.add_argument("--save-ckpt", action=argparse.BooleanOptionalAction, default=False, help="Save final HuggingFace checkpoint into --out-dir")
  p.add_argument("--verify-inference", action=argparse.BooleanOptionalAction, default=True, help="Verify fast_inference alignment after training")

  yaml_loom_enabled = None
  if pre_args.config:
    yaml_cfg = _load_yaml_config(pre_args.config)
    valid_dests = {a.dest for a in p._actions} - {"help", "config"}
    unknown = sorted(set(yaml_cfg.keys()) - valid_dests)
    if unknown:
      raise ValueError(f"Unknown key(s) in config {pre_args.config!r}: {unknown}")
    yaml_loom_enabled = yaml_cfg.get("loom_enabled", None)
    p.set_defaults(**yaml_cfg)

  args = p.parse_args(raw_argv)

  # If a Loom YAML (`loom_enabled: true`) is switched to `--no-loom` on CLI without an explicit
  # `--intermediate-size`, fall back to the preset's parameter-matched Llama FFN width.
  if yaml_loom_enabled is True and not args.loom_enabled:
    if not _cli_has_flag(raw_argv, "--intermediate-size"):
      args.intermediate_size = 0
    if not _cli_has_flag(raw_argv, "--last-ffn-x-dim"):
      args.last_ffn_x_dim = 0
  return args


def build_model_config_from_args(args: argparse.Namespace):
  """Construct and validate a `LoomFormerConfig` from parsed YAML/CLI arguments."""
  for field in ("vocab_size", "hidden_size", "num_layers", "num_heads", "head_dim",
                "intermediate_size", "ffn_m_dim", "last_ffn_x_dim", "d_k", "d_r", "rope_dim"):
    val = getattr(args, field)
    if val < 0:
      raise ValueError(f"{field} ({val}) must be >= 0")

  base_tmp = build_preset_config(args.preset, loom_enabled=args.loom_enabled)
  h_eff = args.hidden_size if args.hidden_size > 0 else base_tmp.hidden_size
  n_eff = args.num_heads if args.num_heads > 0 else base_tmp.num_attention_heads
  if args.head_dim > 0:
    d_eff = args.head_dim
    if h_eff != n_eff * d_eff:
      raise ValueError(
          f"hidden_size ({h_eff}) != num_heads ({n_eff}) * head_dim ({d_eff}) = {n_eff * d_eff}. "
          f"Ensure hidden_size, num_heads, and head_dim are consistent."
      )
  else:
    if h_eff % n_eff != 0:
      raise ValueError(f"hidden_size ({h_eff}) must be divisible by num_heads ({n_eff}).")
    d_eff = h_eff // n_eff

  extra_cfg = dict(
      hidden_size=h_eff,
      num_attention_heads=n_eff,
      head_dim=d_eff,
      max_position_embeddings=max(2048, args.seq_len),
      situ_beta1=args.situ_beta1,
      situ_beta2=args.situ_beta2,
  )
  if args.vocab_size > 0:
    extra_cfg["vocab_size"] = args.vocab_size
  if args.num_layers > 0:
    extra_cfg["num_hidden_layers"] = args.num_layers
  if args.init_std > 0:
    extra_cfg["initializer_range"] = args.init_std
  if args.intermediate_size > 0:
    extra_cfg["intermediate_size"] = args.intermediate_size
    if not args.loom_enabled:
      extra_cfg["last_ffn_x_dim"] = args.intermediate_size
  if args.loom_enabled:
    if args.d_k > 0:
      extra_cfg["d_k"] = args.d_k
    if args.d_r > 0:
      extra_cfg["d_r"] = args.d_r
    if args.rope_dim > 0:
      if args.rope_dim % 2 != 0:
        raise ValueError(f"rope_dim ({args.rope_dim}) must be even.")
      extra_cfg["rope_dim"] = args.rope_dim
    if args.ffn_m_dim > 0:
      extra_cfg["ffn_m_dim"] = args.ffn_m_dim
    if args.last_ffn_x_dim > 0:
      extra_cfg["last_ffn_x_dim"] = args.last_ffn_x_dim
  elif d_eff != base_tmp.head_dim:
    extra_cfg["mha_nope_dim"] = d_eff % 2
  return build_preset_config(args.preset, loom_enabled=args.loom_enabled, **extra_cfg)


def resolve_grad_accum(batch_size: int, micro_batch_size: int, world_size: int, grad_accum: int) -> int:
  """Validate `batch_size == micro_batch_size * world_size * grad_accum` (or infer `grad_accum` when `grad_accum == 0`)."""
  if batch_size <= 0 or micro_batch_size <= 0 or world_size <= 0:
    raise ValueError(
        f"batch_size ({batch_size}), micro_batch_size ({micro_batch_size}), and world_size ({world_size}) must be positive"
    )
  if grad_accum < 0:
    raise ValueError(f"grad_accum ({grad_accum}) must be >= 0")
  if grad_accum > 0:
    expected = micro_batch_size * world_size * grad_accum
    if batch_size != expected:
      raise ValueError(
          f"batch_size ({batch_size}) != micro_batch_size ({micro_batch_size}) * world_size ({world_size}) * "
          f"grad_accum ({grad_accum}) = {expected}. Adjust --grad-accum / --micro-batch-size for {world_size} GPU(s) "
          f"(or pass --grad-accum 0 to auto-compute)."
      )
    return grad_accum
  denom = micro_batch_size * world_size
  if batch_size % denom != 0:
    raise ValueError(
        f"batch_size ({batch_size}) must be divisible by micro_batch_size ({micro_batch_size}) * world_size ({world_size}) = {denom}"
    )
  return batch_size // denom


def main(argv: Optional[List[str]] = None) -> Dict:
  args = parse_args(argv)

  # DDP setup (works seamlessly with or without torchrun)
  ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
  if ddp:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
  else:
    rank, world_size, local_rank = 0, 1, 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
  is_master = rank == 0

  grad_accum = resolve_grad_accum(args.batch_size, args.micro_batch_size, world_size, args.grad_accum)
  tokens_per_step = args.batch_size * args.seq_len

  torch.manual_seed(args.seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

  # Download FineWeb shards on rank 0, then barrier
  if is_master and not args.skip_download:
    download_fineweb10b_shards(args.data_dir, num_train_shards=args.num_train_shards)
  if ddp:
    dist.barrier()

  train_files = [os.path.join(args.data_dir, f"fineweb_train_{i:06d}.bin") for i in range(1, args.num_train_shards + 1)]
  val_files = [os.path.join(args.data_dir, "fineweb_val_000000.bin")]
  train_loader = FineWebShardLoader(train_files, args.micro_batch_size, args.seq_len, rank, world_size, device)
  val_loader = FineWebShardLoader(val_files, args.micro_batch_size, args.seq_len, rank, world_size, device)

  # Build model config from `--config` / `--preset` plus any explicit CLI overrides
  config = build_model_config_from_args(args)
  raw_model = LoomFormerForCausalLM(config).to(device=device, dtype=torch.float32)
  train_model = build_train_model(raw_model, grad_checkpoint=args.grad_checkpoint)
  param_stats = count_parameters(raw_model)

  ddp_model = (
      DDP(train_model, device_ids=[local_rank], gradient_as_bucket_view=True, bucket_cap_mb=64)
      if ddp
      else train_model
  )
  if args.compile:
    model = torch.compile(ddp_model, mode=args.compile_mode)
    loss_fn = torch.compile(_cross_entropy_loss, mode=args.compile_mode)
  else:
    model = ddp_model
    loss_fn = _cross_entropy_loss
  optimizer = build_optimizer(raw_model, args.lr, args.weight_decay, args.beta1, args.beta2)

  use_bf16 = args.dtype == "bfloat16" and device.startswith("cuda")
  amp_ctx = (lambda: torch.amp.autocast("cuda", dtype=torch.bfloat16)) if use_bf16 else nullcontext

  val_step_tokens = args.micro_batch_size * world_size * args.seq_len
  val_batches = max(1, args.val_tokens // val_step_tokens)
  final_val_batches = max(1, args.final_val_tokens // val_step_tokens)

  arch_name = "LoomLlama" if args.loom_enabled else "Llama"
  log_fh = None
  jsonl_fh = None
  if is_master and args.out_dir:
    os.makedirs(args.out_dir, exist_ok=True)
    log_fh = open(os.path.join(args.out_dir, "train.log"), "w", buffering=1)
    jsonl_fh = open(os.path.join(args.out_dir, "metrics.jsonl"), "w", buffering=1)

  def log_print(msg: str) -> None:
    if not is_master:
      return
    print(msg, flush=True)
    if log_fh is not None:
      log_fh.write(msg + "\n")
      log_fh.flush()

  history: List[Dict] = []

  def record_metric(entry: Dict) -> None:
    if not is_master:
      return
    history.append(entry)
    if jsonl_fh is not None:
      jsonl_fh.write(json.dumps(entry) + "\n")
      jsonl_fh.flush()

  if is_master:
    log_print("=" * 84)
    if args.config:
      log_print(f"YAML Config: {args.config}")
    log_print(
        f"Training {arch_name} | params={param_stats['total']:,} (non-emb={param_stats['non_embedding']:,}) "
        f"| AdamW lr={args.lr:.2e} | compile={args.compile_mode if args.compile else False} "
        f"| grad_ckpt={args.grad_checkpoint} | dtype={args.dtype}"
    )
    log_print(
        f"Config: D={config.hidden_size}, L={config.num_hidden_layers}, n={config.num_attention_heads}, "
        f"d={config.head_dim}, ffn_x={config.intermediate_size}"
        + (f", ffn_m={config.ffn_m_dim}, last_ffn_x={config.last_ffn_x_dim}, d_k={config.d_k}, d_r={config.d_r}, r={config.rope_dim}"
           if config.loom_enabled else "")
    )
    log_print(
        f"Schedule: steps={args.steps}, global_bs={args.batch_size}, micro_bs={args.micro_batch_size}, "
        f"world={world_size}, grad_accum={grad_accum}, seq_len={args.seq_len}, tok/step={tokens_per_step:,}, "
        f"total_tokens={args.steps * tokens_per_step / 1e6:.2f}M"
    )
    log_print(
        f"Data: train_shards={len(train_files)} ({train_loader.total_tokens / 1e6:.1f}M tok), "
        f"val_eval={val_batches * val_step_tokens / 1e6:.2f}M tok (final {final_val_batches * val_step_tokens / 1e6:.2f}M tok)"
    )
    log_print("=" * 84)

  model.train()
  if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

  # Initial val loss at step 0
  init_val_loss, init_val_ppl = evaluate_val(model, val_loader, val_batches, amp_ctx, ddp, loss_fn=loss_fn)
  if is_master:
    log_print(f"[step    0/{args.steps}] val_loss={init_val_loss:.4f} | val_ppl={init_val_ppl:.2f}")
    record_metric({"step": 0, "tokens_seen": 0, "val_loss": init_val_loss, "val_ppl": init_val_ppl})

  t_window = time.perf_counter()
  steps_in_window = 0
  loss_accum_t = torch.zeros((), device=device, dtype=torch.float64)
  train_start = time.perf_counter()

  for step in range(1, args.steps + 1):
    lr_now = cosine_lr(step - 1, args.steps, args.lr, args.min_lr_ratio, args.warmup_steps)
    for pg in optimizer.param_groups:
      pg["lr"] = lr_now

    optimizer.zero_grad(set_to_none=True)
    for micro in range(grad_accum):
      x, y = train_loader.next_batch()
      sync_ctx = ddp_model.no_sync() if (ddp and micro < grad_accum - 1) else nullcontext()
      with sync_ctx:
        with amp_ctx():
          logits = model(x)
          loss = loss_fn(logits, y) / grad_accum
        loss.backward()
      loss_accum_t += loss.detach().double()

    grad_norm_t = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
    optimizer.step()
    steps_in_window += 1

    if step % args.log_every == 0 or step == 1 or step == args.steps:
      if ddp:
        dist.all_reduce(loss_accum_t, op=dist.ReduceOp.AVG)
      if torch.cuda.is_available():
        torch.cuda.synchronize(device)
      dt = time.perf_counter() - t_window
      ms_per_step = 1000.0 * dt / steps_in_window
      tok_per_sec = (tokens_per_step * steps_in_window) / max(dt, 1e-6)
      avg_loss = float(loss_accum_t.item()) / steps_in_window
      grad_norm = float(grad_norm_t.item())
      peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if torch.cuda.is_available() else 0.0
      if is_master:
        elapsed = time.perf_counter() - train_start
        eta_min = (args.steps - step) * (ms_per_step / 1000.0) / 60.0
        log_print(
            f"[step {step:4d}/{args.steps}] train_loss={avg_loss:.4f} | lr={lr_now:.2e} | gnorm={grad_norm:.3f} "
            f"| {ms_per_step:.1f} ms/step | {tok_per_sec / 1e3:.1f}k tok/s | mem={peak_mb:.0f} MB | eta={eta_min:.1f}m"
        )
        record_metric({
            "step": step,
            "tokens_seen": step * tokens_per_step,
            "train_loss": avg_loss,
            "lr": lr_now,
            "grad_norm": grad_norm,
            "ms_per_step": ms_per_step,
            "tok_per_sec": tok_per_sec,
            "elapsed_sec": elapsed,
        })
      t_window = time.perf_counter()
      steps_in_window = 0
      loss_accum_t.zero_()

    if args.eval_every > 0 and step % args.eval_every == 0 and step < args.steps:
      val_loss, val_ppl = evaluate_val(model, val_loader, val_batches, amp_ctx, ddp, loss_fn=loss_fn)
      if is_master:
        log_print(f"  >>> [eval step {step:4d}] val_loss={val_loss:.4f} | val_ppl={val_ppl:.2f}")
        record_metric({
            "step": step,
            "tokens_seen": step * tokens_per_step,
            "val_loss": val_loss,
            "val_ppl": val_ppl,
            "elapsed_sec": time.perf_counter() - train_start,
        })
      t_window = time.perf_counter()

  if torch.cuda.is_available():
    torch.cuda.synchronize(device)
  total_train_sec = time.perf_counter() - train_start

  # Final full validation
  final_val_loss, final_val_ppl = evaluate_val(model, val_loader, final_val_batches, amp_ctx, ddp, loss_fn=loss_fn)
  summary: Dict = {
      "arch": arch_name,
      "loom_enabled": args.loom_enabled,
      "compile": args.compile,
      "compile_mode": args.compile_mode if args.compile else None,
      "grad_checkpoint": args.grad_checkpoint,
      "params": param_stats,
      "steps": args.steps,
      "tokens_seen": args.steps * tokens_per_step,
      "train_seconds": total_train_sec,
      "avg_tok_per_sec": (args.steps * tokens_per_step) / max(total_train_sec, 1e-6),
      "final_val_loss": final_val_loss,
      "final_val_ppl": final_val_ppl,
      "peak_mem_mb": torch.cuda.max_memory_allocated(device) / (1024 ** 2) if torch.cuda.is_available() else 0.0,
  }

  if is_master:
    record_metric({
        "step": args.steps,
        "tokens_seen": args.steps * tokens_per_step,
        "val_loss": final_val_loss,
        "val_ppl": final_val_ppl,
        "is_final": True,
        "elapsed_sec": total_train_sec,
    })
    log_print("=" * 84)
    log_print(
        f"DONE {arch_name} | final_val_loss={final_val_loss:.4f} | final_val_ppl={final_val_ppl:.2f} "
        f"| time={total_train_sec:.1f}s | avg_speed={summary['avg_tok_per_sec'] / 1e3:.1f}k tok/s"
    )

    if args.verify_inference and device.startswith("cuda"):
      align = verify_inference_alignment(raw_model)
      summary["inference_alignment"] = align
      log_print(
          f"INFERENCE ALIGNMENT: fast_kernels={bool(align['fast_kernels_active'])} "
          f"| KL(train, hf_ref)={align['kl_train_vs_hf_ref']:.2e} "
          f"| KL(train, fast_full)={align['kl_train_vs_fast_full']:.2e} "
          f"| max_abs_diff_prefill={align['max_abs_diff_prefill']:.2e}"
      )

    if args.out_dir:
      os.makedirs(args.out_dir, exist_ok=True)
      with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump({"summary": summary, "history": history, "args": vars(args)}, f, indent=2)
      if args.save_ckpt:
        ckpt_dir = os.path.join(args.out_dir, "hf_checkpoint")
        hf_save = LoomFormerForCausalLM(raw_model.config)
        hf_save.load_state_dict({k: v.detach().cpu().to(torch.bfloat16) for k, v in raw_model.state_dict().items()})
        hf_save.save_pretrained(ckpt_dir)
        log_print(f"Saved HuggingFace BF16 checkpoint -> {ckpt_dir}")
    log_print("=" * 84)
    if log_fh is not None:
      log_fh.close()
    if jsonl_fh is not None:
      jsonl_fh.close()

  if ddp:
    dist.barrier()
    dist.destroy_process_group()
  return summary


if __name__ == "__main__":
  main()
