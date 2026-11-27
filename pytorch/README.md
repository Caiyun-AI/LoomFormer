# LoomFormer — PyTorch Training, Inference & Triton Kernels (`pytorch/`)

This directory provides HuggingFace-compatible PyTorch implementations of **LoomPythia** and **LoomFormer**, together with:

1. **Fused Triton Training Engine & DDP Pretraining** (`training/engine.py`, `training/triton_ops.py`, `train_loom.py`, `configs/`): separates the fast Triton training wrapper (`build_train_model(hf_model, grad_checkpoint=False)`, mirroring `fast_inference/engine.py` with live HF parameters) from the pure HuggingFace reference in `loomformer/`, reusing `fast_inference/kernels_llama.py` with shape-adaptive Triton dispatch (`160m` → `2.8b`, zero-spill autograd backward kernels & precomputed-FB pipeline), FlashAttention-2 native head-dim bucketing, `torch.compile` (0 graph breaks), optional `--grad-checkpoint`, standalone YAML configs (`configs/loomformer_*.yml`), and FineWeb10B scripts.
2. **Fused Triton + CUDA-Graph Fast Inference Engine** (`fast_inference/`): static KV/$M$ cache wrappers for high-throughput prefill and autoregressive decode.
3. **MaxText-to-HuggingFace Converter** (`scripts/convert_maxtext_to_hf.py`): converts Orbax OCDBT MaxText checkpoints directly to standalone HuggingFace `safetensors` directories.

We release **[LoomPythia-2.8B](https://huggingface.co/Caiyun-AI/LoomPythia-2.8B)** and the matched 4-stream mHC comparison baseline **[mHC-Pythia-2.8B](https://huggingface.co/Caiyun-AI/mHC-Pythia-2.8B)** (both pretrained on 300B Pile tokens) on HuggingFace 🤗.

---



## Installation

```bash
cd pytorch
pip install -r requirements.txt
```

*(Tested with Python 3.10/3.11,* `torch >= 2.5`*,* `triton >= 3.1`*,* `transformers >= 4.46` *on NVIDIA A800 / RTX PRO 6000 GPUs.)*

---



## Directory Overview

```text
pytorch/
├── configs/           # Standalone YAML training configs (`160m`, `medium`, `large`, `xl`, `2.8b`)
├── loompythia/        # HF LoomPythiaConfig & LoomPythiaForCausalLM (Pythia backbone + QKNorm)
├── loomformer/        # Pure HF LoomFormerForCausalLM reference (LoomLlama & matched Llama baselines)
├── fast_inference/    # Fused Triton + CUDA-graph prefill & decode engine (`fast_inference.build`)
├── training/          # Shape-adaptive Triton training engine, autograd ops, presets & FineWeb10B loader
├── scripts/           # MaxText-to-HF checkpoint converter & inference latency benchmark
├── tests/             # Unit & integration tests (KV+M cache, Triton grad parity, torch.compile, alignment)
├── train_loom.py      # Single-GPU / multi-GPU (`torchrun` DDP) training & alignment entrypoint
└── plot_curves.py     # Plot train/val loss curves vs. tokens, steps, or wall-clock time
```

---



## 1. PyTorch Pretraining (LoomFormer vs. Matched Llama)

`train_loom.py` trains parameter-matched **LoomFormer** (`--loom`) and standard **Llama** (`--no-loom`) models on GPT-2 tokenized FineWeb10B binary shards from [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) (`[kjj0/fineweb10B-gpt2](https://huggingface.co/datasets/kjj0/fineweb10B-gpt2)`).

**Data download:** Shards (~200 MB / 100M tokens each; `--num-train-shards 20` ≈ 4.2 GB including val) are downloaded automatically on rank 0 into `pytorch/data/fineweb10B` (override with `--data-dir`). On slow connections or large shard counts, you can pre-download ahead of `torchrun` (set `HF_ENDPOINT` if using a mirror) to avoid multi-GPU NCCL barrier timeouts, then pass `--skip-download`:

```bash
python -c "import sys; sys.path.insert(0, 'pytorch'); from training.data import download_fineweb10b_shards; download_fineweb10b_shards('pytorch/data/fineweb10B', num_train_shards=20)"
```



### Reproducing the Pretraining Experiments

Pass any YAML config via `--config` (any CLI flag overrides the YAML; pass `--no-loom` to automatically run the parameter-matched Llama baseline for that scale):

```bash
# Train LoomFormer directly from its YAML config (replace with medium / large / xl / 2p8b as needed)
torchrun --standalone --nproc_per_node=8 pytorch/train_loom.py \
  --config pytorch/configs/loomformer_160m.yml

# Train the parameter-matched Llama baseline from the same YAML config (`--no-loom` switches FFN to matched Llama width)
torchrun --standalone --nproc_per_node=8 pytorch/train_loom.py \
  --config pytorch/configs/loomformer_160m.yml --no-loom \
  --out-dir pytorch/runs/llama_162m_2b_seq4096_b128

# Optional: override any YAML field on the CLI (e.g. FA2 HEADDIM=96 alignment on 2.8B via `--rope-dim 16`)
torchrun --standalone --nproc_per_node=8 pytorch/train_loom.py \
  --config pytorch/configs/loomformer_2p8b.yml --rope-dim 16

# Plot comparison curves (by tokens, step, or wall-clock time)
python pytorch/plot_curves.py \
  --runs pytorch/runs/llama_162m_2b_seq4096_b128 pytorch/runs/loomllama_162m_2b_seq4096_b128 \
  --labels "Llama-162M (162.22M, T=4096)" "LoomLlama-162M (162.22M, T=4096)" \
  --x-axis tokens --out pytorch/runs/compare_162m_2b_seq4096_b128.png
```

*(The default YAMLs target 8 GPUs and explicitly enforce* `batch_size == micro_batch_size * world_size * grad_accum`*. On a different GPU count—e.g. a single GPU—adjust* `--grad-accum` *or pass* `--grad-accum 0` *to auto-compute it.)*

After training completes, `--verify-inference` (enabled by default) converts the trained weights to BF16, builds `fast_inference.build(hf_model)`, and checks end-to-end KL alignment across the Triton training engine (`build_train_model`), pure PyTorch HuggingFace reference (`LoomFormerForCausalLM`), and `fast_inference` prefill + autoregressive decode.

### 8×A800 Training Throughput & FineWeb 2B-Token Results

Measured on `8×A800-80GB` (`seq_len=4096`, `batch_size=128`, `--compile`, `--grad-checkpoint`, BF16) with shape-adaptive Triton kernel dispatch (`_select_kernel_config` in `training/triton_ops.py`). Throughput and Peak Mem are reported relative to the parameter-matched Llama baseline (`100%`):


| Model                                            | Params  | FA2 Bucket (`Llama → Loom`) | Throughput (`vs. Llama`) | Peak Mem (`vs. Llama`) |
| ------------------------------------------------ | ------- | --------------------------- | ------------------------ | ---------------------- |
| **LoomLlama-162M**                               | 162.22M | `64 → 96`                   | **76.0%**                | 107.6%                 |
| **LoomLlama-162M*** (`n=16, d=48, r=16`)         | 162.20M | `64 → 64`                   | **77.8%**                | 106.1%                 |
| **LoomFormer-Medium**                            | 432.11M | `96 → 96`                   | **91.1%**                | 114.8%                 |
| **LoomFormer-Large**                             | 777.85M | `96 → 128`                  | **84.5%**                | 119.3%                 |
| **LoomFormer-Large*** (`D=1280, n=16, d=80, r=16`) | 777.89M | `96 → 96`                 | **86.7%**                | 114.5%                 |
| **LoomFormer-XL**                                | 1.432B  | `96 → 128`                  | **83.3%**                | 114.1%                 |
| **LoomFormer-XL*** (`D=1600, n=20, d=80, r=16`)  | 1.432B  | `96 → 96`                   | **85.2%**                | 110.1%                 |
| **LoomFormer-2.8B** (`r=20`)                     | 2.774B  | `96 → 128`                  | **84.8%**                | 108.7%                 |
| **LoomFormer-2.8B*** (`r=16`)                    | 2.757B  | `96 → 96`                   | **88.8%**                | 108.2%                 |


*Tip — FlashAttention-2 `HEADDIM` bucket alignment:* The default YAMLs in `configs/` preserve the exact paper architectures, and every row above is measured against the **unchanged** paper Llama baseline at that scale. When training LoomFormer on GPUs, you can adjust only LoomFormer's `(n, d, r)` (keeping `n` and `L` unchanged and re-matching total parameters via FFN width when `d` changes) so that `d + r` stays within the same native FlashAttention-2 `HEADDIM` bucket (`64 / 96 / 128`) as Llama—reducing attention-kernel overhead and peak memory with minimal expected impact on final loss (e.g., `162M` `76.0% → 77.8%`, `Large` `84.5% → 86.7%` and `-4.8 pp` peak mem, `XL` `83.3% → 85.2%` and `-4.0 pp` peak mem, `2.8B` `84.8% → 88.8%`).

**FineWeb 2B-token validation (**`160m`**,** `T=4096, B=128`**):** **LoomLlama-162M** achieves **3.2696** val loss / **26.30** PPL (−0.202 loss vs. Llama-162M **3.4711** / **32.17**; `KL = 1.34e-4` against `fast_inference` prefill + decode):

![162M Llama vs LoomLlama Training & Validation Loss on 2B FineWeb Tokens (T=4096, B=128)](../assets/pytorch_train_162m_2b.png)

---



## 2. Standard HuggingFace Generation

`LoomPythiaForCausalLM` and `LoomFormerForCausalLM` support standard `transformers` APIs (`forward`, `generate`, `DynamicCache` with per-layer `loom_m` state, left-padding, and chunked prefill):

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Caiyun-AI/LoomPythia-2.8B"  # or local checkpoint from `--save-ckpt` / converter
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, dtype=torch.bfloat16
).cuda().eval()

inputs = tokenizer("The capital of France is", return_tensors="pt").to("cuda")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=32, do_sample=False)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

---



## 3. Accelerated Inference with Triton Kernels & `torch.compile` CUDA Graphs

For maximum prefill and autoregressive decode throughput on CUDA GPUs (~88% of stock `Pythia-2.8B` compiled speed on A800), wrap any loaded model with `fast_inference.build`:

```python
import sys
sys.path.insert(0, "/path/to/LoomFormer/pytorch")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import fast_inference

model_id = "Caiyun-AI/LoomPythia-2.8B"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
hf_model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, dtype=torch.bfloat16
).cuda().eval()

fast = fast_inference.build(hf_model)  # also accepts GPTNeoXForCausalLM or LoomFormerForCausalLM

prompt_ids = tokenizer(["The capital of France is"], return_tensors="pt").input_ids.cuda()
max_new_tokens = 32
state = fast.new_state(batch=prompt_ids.shape[0], max_len=prompt_ids.shape[1] + max_new_tokens)

# Compile a single CUDA-graph-captured decode step (first call compiles & autotunes)
decode_step = torch.compile(
    lambda tok: fast.decode(tok, state), mode="max-autotune", fullgraph=True, dynamic=False
)

with torch.no_grad():
    nxt = fast.prefill(prompt_ids, state)[:, -1].argmax(dim=-1, keepdim=True)
    generated = [prompt_ids, nxt]
    for _ in range(max_new_tokens - 1):
        nxt = decode_step(nxt)[:, -1].argmax(dim=-1, keepdim=True)
        generated.append(nxt)

print(tokenizer.decode(torch.cat(generated, dim=1)[0]))
```

To reuse the compiled graph on a new prompt of the same batch size, call `state.reset()` and invoke `fast.prefill(...)` again.

---

