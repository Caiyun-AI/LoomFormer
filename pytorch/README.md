# LoomFormer — PyTorch Inference & Triton Kernels (`pytorch/`)

This directory provides HuggingFace-compatible PyTorch models for **LoomPythia** and **LoomFormer**, together with a **fused Triton + CUDA-graph fast inference engine** (`fast_inference/`) and MaxText checkpoint conversion utilities.

We release **[LoomPythia-2.8B](https://huggingface.co/Caiyun-AI/LoomPythia-2.8B)** (pretrained on 300B Pile tokens) on HuggingFace 🤗.

## Installation

```bash
cd pytorch
pip install -r requirements.txt
```

*(Tested with `python == 3.11`, `torch >= 2.5`, `triton >= 3.1`, `transformers >= 4.46` on NVIDIA A800 / RTX PRO 6000 GPUs.)*

## Directory Overview

- `loompythia/`: Standalone HuggingFace `LoomPythiaConfig` & `LoomPythiaForCausalLM` (Pythia serial pre-LN backbone with full-head `QKNorm`).
- `loomformer/`: Standalone HuggingFace `LoomFormerConfig` & `LoomFormerForCausalLM` (Llama RMSNorm + SwiGLU/SiTU backbone and standard MHA baselines).
- `fast_inference/`: Fused Triton kernels + static KV/$M$ cache wrappers (`fast_inference.build(hf_model)`):
  - `kernels_pythia.py` / `kernels_llama.py`: Fuses previous-layer `MWrite` + `MShift` + whole-matrix `MNorm` + four-arm `MRead` + `QKNorm` + `RoPE` in a single pass over $M$, plus grouped small-batch GEMVs/GEMMs.
  - `prefill_attention.py`: Strictly causal ($s < t$) Triton flash attention + `o_m` bypass for unequal Q/K (`100`) and V (`80`) head widths.
  - `decode_attention.py`: CUDA-graph-safe split-KV flash-decoding with transposed-block (`kt`) tail caching.
- `scripts/convert_maxtext_to_hf.py`: Converts an Orbax OCDBT MaxText checkpoint (`items/`) or flat `.npz` archive directly into a standalone HuggingFace `safetensors` directory (using `tensorstore`, without requiring JAX/Orbax).
- `scripts/benchmark_inference.py`: Benchmarks prefill and `torch.compile` CUDA-graph decode latency.
- `tests/test_loom_pytorch.py`: Unit tests for KV+$M$ cache decoding, chunked prefill, left-padding, `generate()`, and Triton `fast_inference` parity.

## Usage

### 1. Standard HuggingFace Generation

`LoomPythiaForCausalLM` and `LoomFormerForCausalLM` support standard `transformers` APIs (`forward`, `generate`, `DynamicCache` with per-layer `loom_m` state, left-padding, and chunked prefill):

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Caiyun-AI/LoomPythia-2.8B"  # or local converted directory
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, dtype=torch.bfloat16
).cuda().eval()

inputs = tokenizer("The capital of France is", return_tensors="pt").to("cuda")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=32, do_sample=False)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

### 2. Accelerated Inference with Triton Kernels & `torch.compile` CUDA Graphs

For maximum prefill and autoregressive decode throughput on CUDA GPUs (achieving ~88% of stock `Pythia-2.8B` compiled speed on A800), wrap the loaded model with `fast_inference.build`:

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

fast = fast_inference.build(hf_model)  # also accepts stock GPTNeoXForCausalLM (Pythia) or LoomFormerForCausalLM

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

### 3. Converting a MaxText Checkpoint to HuggingFace Format

```bash
python scripts/convert_maxtext_to_hf.py \
  --ckpt /path/to/checkpoints/143000/items \
  --out-dir /path/to/LoomPythia-2.8B \
  --dtype bfloat16 \
  --tokenizer /path/to/pythia-tokenizer
```

### 4. Running Tests & Benchmarks

```bash
# Run numerical parity & Triton kernel tests
python -m unittest -v tests.test_loom_pytorch

# Benchmark prefill (1x2048, 8x2048) and CUDA-graph decode (bs=1, 16, 64)
python scripts/benchmark_inference.py --model-dir /path/to/LoomPythia-2.8B --compile max-autotune
```
