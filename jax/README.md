# LoomFormer — JAX / TPU Training (`jax/`)

This directory contains the standalone JAX/Flax training implementation of **LoomFormer** and **LoomPythia** on top of official **[AI-Hypercomputer/maxtext (`maxtext-v0.2.0`)](https://github.com/AI-Hypercomputer/maxtext/releases/tag/maxtext-v0.2.0)** as a pure external dependency (**zero patches or forked MaxText files**).

All modules follow a PyTorch-familiar `setup()` (define parameters & submodules) + `__call__()` (forward pass) layout, with `einsum` / shape annotations matching Paper Section 3.2 and Appendix C (`loomformer_pseudo.py`).

## Directory Structure

```text
jax/
├── requirements_tpu.txt          # Pins official maxtext[tpu] @ maxtext-v0.2.0 (jax[tpu]==0.11.0)
├── train_loom.py                 # Training entrypoint plugging LoomFormer / LoomPythia into MaxText's train_loop
├── configs/                      # Paper model YAML configurations
│   ├── loomformer_medium.yml     # LoomFormer-Medium (432M, D=1200, n=16, d=75, d_k=32, D_r=256, r=18)
│   ├── loomformer_large.yml      # LoomFormer-Large  (778M, D=1536, n=16, d=96, d_k=32, D_r=256, r=24)
│   ├── loomformer_xl.yml         # LoomFormer-XL     (1.43B, D=1920, n=20, d=96, d_k=40, D_r=400, r=24)
│   ├── loomformer_xl_situ.yml    # LoomFormer-XL with bounded SiTU-GLU (beta_1=4, beta_2=7)
│   └── loompythia_2p8b.yml       # LoomPythia-2.8B   (2.78B, D=2560, n=32, d=80, d_k=64, D_r=640, r=20)
├── loomformer/
│   ├── config.py                 # LoomConfig dataclass & YAML/CLI config splitter
│   ├── loomformer.py             # Core Loom primitives (MShift, MRead, MWrite, LoomEmbedWrite) and
│   │                             # Llama-backbone LoomFormer (LoomMHA, LoomDecoderLayer, LoomDecoder, LoomFormer)
│   ├── loompythia.py             # Pythia-backbone LoomPythia (LayerNorm + GELU + QKNorm + bf16_dot_fp32_output)
│   └── loom_pallas.py            # Fused forward/backward TPU Pallas `read` & `write` kernels + Splash adapters
└── tests/
    └── loomformer_test.py        # Parameter counts, pure-JAX vs. Pallas parity, and official train_step tests
```

## Installation

Requires `python >= 3.12` on Cloud TPU v5p / v6e:

```bash
pip install -r jax/requirements_tpu.txt
```

## Training

`train_loom.py` accepts any config in `jax/configs/*.yml` plus standard [MaxText CLI overrides](https://github.com/AI-Hypercomputer/maxtext) (`dataset_type=hf`, `grain`, `tfds`, or `synthetic`):

```bash
# Pretraining LoomPythia-2.8B on the Pile via HuggingFace streaming / local shards
python jax/train_loom.py jax/configs/loompythia_2p8b.yml \
  run_name=loompythia_2p8b \
  base_output_directory=gs://your-bucket/loomformer-runs \
  dataset_type=hf \
  hf_path=EleutherAI/pile

# Quick synthetic-data smoke test on a single TPU host
python jax/train_loom.py jax/configs/loomformer_medium.yml \
  run_name=local_smoke \
  base_output_directory=/tmp/loom_runs \
  dataset_type=synthetic \
  enable_checkpointing=False \
  per_device_batch_size=2 \
  steps=10
```

Set `loom_pallas=False` on the command line to switch from the fused TPU Pallas kernels (`loom_pallas.py`) to the pure-JAX `einsum` reference path (`loomformer.py` / `loompythia.py`) on the identical parameter tree.

## Running Tests

```bash
python jax/tests/loomformer_test.py -v
```
