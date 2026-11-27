"""Plot and compare training / validation loss curves from one or more run directories.

Reads live `metrics.jsonl` (preferred, works while training is running) or
completed `metrics.json` from each run directory and outputs a comparison PNG.

Usage:
  python plot_curves.py \\
    --runs pytorch/runs/loomllama_162m_2b pytorch/runs/llama_162m_2b \\
    --labels "LoomLlama-162M" "Llama-162M" \\
    --out runs/loss_comparison.png
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_run_history(run_dir: str) -> Tuple[str, List[Dict]]:
  jsonl_path = os.path.join(run_dir, "metrics.jsonl")
  json_path = os.path.join(run_dir, "metrics.json")
  label = os.path.basename(os.path.normpath(run_dir))

  if os.path.exists(jsonl_path):
    history: List[Dict] = []
    with open(jsonl_path) as f:
      for line in f:
        line = line.strip()
        if line:
          history.append(json.loads(line))
    if os.path.exists(json_path):
      try:
        with open(json_path) as f:
          arch = json.load(f).get("summary", {}).get("arch")
          if arch:
            label = f"{arch} ({label})"
      except Exception:
        pass
    return label, history

  if os.path.exists(json_path):
    with open(json_path) as f:
      data = json.load(f)
    arch = data.get("summary", {}).get("arch")
    if arch:
      label = f"{arch} ({label})"
    return label, data.get("history", [])

  raise FileNotFoundError(f"Neither metrics.jsonl nor metrics.json found in {run_dir}")


def extract_series(history: List[Dict], metric: str, x_axis: str) -> Tuple[List[float], List[float]]:
  xs: List[float] = []
  ys: List[float] = []
  for row in history:
    if metric not in row:
      continue
    if x_axis == "tokens" and "tokens_seen" in row:
      xs.append(row["tokens_seen"] / 1e6)
    elif x_axis == "time" and "elapsed_sec" in row:
      xs.append(row["elapsed_sec"] / 60.0)
    else:
      xs.append(float(row["step"]))
    ys.append(float(row[metric]))
  return xs, ys


def main() -> None:
  p = argparse.ArgumentParser(description="Compare LoomLlama vs Llama training & validation loss curves")
  p.add_argument("--runs", nargs="+", required=True, help="One or more run directories")
  p.add_argument("--labels", nargs="*", default=None, help="Optional legend labels matching --runs")
  p.add_argument("--x-axis", choices=["tokens", "step", "time"], default="tokens",
                 help="X-axis unit: tokens (millions), step, or time (wall-clock minutes)")
  p.add_argument("--target-loss", type=float, default=0.0, help="Optional horizontal reference line (<=0 disables)")
  p.add_argument("--out", type=str, default="runs/loss_comparison.png", help="Output PNG path")
  args = p.parse_args()

  if args.labels is not None and len(args.labels) != len(args.runs):
    raise ValueError(f"--labels length ({len(args.labels)}) must match --runs ({len(args.runs)})")

  fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
  ax_train, ax_val = axes
  x_label = {"tokens": "Tokens Seen (M)", "step": "Step", "time": "Wall-Clock Time (min)"}[args.x_axis]

  for idx, run_dir in enumerate(args.runs):
    default_label, history = load_run_history(run_dir)
    label = args.labels[idx] if args.labels is not None else default_label

    tx, ty = extract_series(history, "train_loss", args.x_axis)
    if tx:
      ax_train.plot(tx, ty, linewidth=1.8, label=f"{label} (last {ty[-1]:.4f})")

    vx, vy = extract_series(history, "val_loss", args.x_axis)
    if vx:
      # Skip step 0 (~10.9) when plotting zoom if later points exist
      if len(vx) > 1 and vx[0] == 0:
        vx, vy = vx[1:], vy[1:]
      ax_val.plot(vx, vy, marker="o", markersize=4, linewidth=2.0, label=f"{label} (last {vy[-1]:.4f})")

  if args.target_loss > 0:
    ax_val.axhline(args.target_loss, color="gray", linestyle="--", linewidth=1.2, label=f"Target ({args.target_loss:.2f})")

  ax_train.set_title("Training Loss")
  ax_train.set_xlabel(x_label)
  ax_train.set_ylabel("Cross-Entropy Loss")
  ax_train.grid(True, alpha=0.3)
  ax_train.legend()

  ax_val.set_title("Validation Loss")
  ax_val.set_xlabel(x_label)
  ax_val.set_ylabel("Cross-Entropy Loss")
  ax_val.grid(True, alpha=0.3)
  ax_val.legend()

  fig.tight_layout()
  os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
  fig.savefig(args.out)
  print(f"Saved curve comparison -> {args.out}")


if __name__ == "__main__":
  main()
