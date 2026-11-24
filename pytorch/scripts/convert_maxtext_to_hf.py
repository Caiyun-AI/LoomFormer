"""Convert a MaxText LoomPythia checkpoint (Orbax `items/` directory or flat `.npz`) to HuggingFace `safetensors`.

Examples:
  # Convert directly from an Orbax MaxText checkpoint directory (`.../checkpoints/143000/items`):
  python pytorch/scripts/convert_maxtext_to_hf.py \
      --ckpt /path/to/checkpoints/143000/items \
      --out-dir /path/to/LoomPythia-2.8B \
      --dtype bfloat16 \
      --tokenizer /path/to/pythia-tokenizer

  # Convert from an exported flat `.npz` parameter archive:
  python pytorch/scripts/convert_maxtext_to_hf.py \
      --params /path/to/params.npz \
      --out-dir /path/to/LoomPythia-2.8B \
      --dtype bfloat16

Supports both the refactored `jax/` LoomPythia parameter tree (`decoder/embed_write/W_e`, `loom_mha/...`)
and the original training runtime parameter tree (`decoder/embedding_bam_write/W_emb_u`, `self_attention/...`),
asserting a strict 1-to-1 bijection over every parameter leaf and exact shape equality against `LoomPythiaForCausalLM`.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

import numpy as np
from safetensors.torch import save_file
import torch

PYTORCH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PYTORCH_ROOT not in sys.path:
  sys.path.insert(0, PYTORCH_ROOT)

from loompythia.configuration_loompythia import LoomPythiaConfig  # noqa: E402
from loompythia.modeling_loompythia import LoomPythiaForCausalLM  # noqa: E402

LOOMPYTHIA_DIR = os.path.join(PYTORCH_ROOT, "loompythia")
STANDALONE_CODE = ("configuration_loompythia.py", "modeling_loompythia.py")


def _linear(k: np.ndarray) -> np.ndarray:
  return k.reshape(k.shape[0], -1).T


def _out_proj(k: np.ndarray) -> np.ndarray:
  return k.reshape(-1, k.shape[-1]).T


def _flat(x: np.ndarray) -> np.ndarray:
  return x.reshape(-1)


def _plus_one(x: np.ndarray) -> np.ndarray:
  return x + 1.0


def _transpose_2d(x: np.ndarray) -> np.ndarray:
  return x.T


def load_orbax_params_only(ckpt_items_path: str) -> dict[str, np.ndarray]:
  """Restore ONLY the `params` subtree from an Orbax OCDBT/Zarr checkpoint via `tensorstore` (no JAX/Orbax required)."""
  from concurrent.futures import ThreadPoolExecutor
  import tensorstore as ts

  ckpt_items_path = os.path.abspath(ckpt_items_path)
  meta_path = os.path.join(ckpt_items_path, "_METADATA")
  with open(meta_path, "r") as f:
    meta = json.load(f)

  driver = "zarr3" if meta.get("use_zarr3") else "zarr"
  use_ocdbt = bool(meta.get("use_ocdbt", True))
  tree_meta = meta.get("tree_metadata", meta)

  # Verify checkpoint step if present
  if "('step',)" in tree_meta:
    kv = ({"driver": "ocdbt", "base": {"driver": "file", "path": ckpt_items_path}, "path": "step"}
          if use_ocdbt else {"driver": "file", "path": os.path.join(ckpt_items_path, "step")})
    step_val = int(ts.open({"driver": driver, "kvstore": kv}).result().read().result())
    print(f"Restoring Orbax checkpoint at {ckpt_items_path} (verified step={step_val})")

  tasks = []
  for entry in tree_meta.values():
    keys = [km["key"] for km in entry["key_metadata"]]
    if not keys or keys[0] != "params":
      continue
    ts_path = ".".join(keys)
    # Strip leading `params` (or `params/params`) prefix
    rel_keys = keys[2:] if (len(keys) >= 2 and keys[1] == "params") else keys[1:]
    flat_name = "/".join(rel_keys)
    tasks.append((flat_name, ts_path))

  def _read_one(item):
    flat_name, ts_path = item
    kv = ({"driver": "ocdbt", "base": {"driver": "file", "path": ckpt_items_path}, "path": ts_path}
          if use_ocdbt else {"driver": "file", "path": os.path.join(ckpt_items_path, ts_path)})
    arr = ts.open({"driver": driver, "kvstore": kv}).result().read().result()
    return flat_name, np.asarray(arr, dtype=np.float32)

  with ThreadPoolExecutor(max_workers=16) as pool:
    pairs = list(pool.map(_read_one, tasks))
  return dict(pairs)


def _is_refactored_tree(params: dict[str, np.ndarray]) -> bool:
  return "decoder/embed_write/W_e/kernel" in params


def _legacy_layer_map(writer: bool):
  """Mapping from original MaxText training parameter suffixes (`.../block/`) to `LoomPythiaDecoderLayer`."""
  a, s = "self_attention/", "loom_mha."
  m = [
      ("pre_self_attention_layer_norm/scale", "ln_1.weight", None),
      ("pre_self_attention_layer_norm/bias", "ln_1.bias", None),
      ("post_self_attention_layer_norm/scale", "ln_2.weight", None),
      ("post_self_attention_layer_norm/bias", "ln_2.bias", None),
      ("mlp/wi/kernel", "ffn_x.fc_in.weight", _linear),
      ("mlp/wi/bias", "ffn_x.fc_in.bias", None),
      ("mlp/wo/kernel", "ffn_x.fc_out.weight", _linear),
      ("mlp/wo/bias", "ffn_x.fc_out.bias", None),
      (a + "M_shift_gate/kernel", s + "W_sg.weight", _linear),
      (a + "M_shift_gate_b0", s + "W_sg.bias", None),
      (a + "query/kernel", s + "W_q.weight", _linear),
      (a + "query/bias", s + "W_q.bias", _flat),
      (a + "key/kernel", s + "W_k.weight", _linear),
      (a + "key/bias", s + "W_k.bias", _flat),
      (a + "full_m_read_down/kernel", s + "W_r1.weight", _linear),
      (a + "W_lq_c8_up/kernel", s + "W_r2_q.weight", _linear),
      (a + "general_q_pre_bias", s + "W_r2_q.bias", _flat),
      (a + "W_lk_c8_up/kernel", s + "W_r2_k.weight", _linear),
      (a + "general_k_pre_bias", s + "W_r2_k.bias", _flat),
      (a + "W_lv_c8_up/kernel", s + "W_r2_v.weight", _linear),
      (a + "general_v_pre_bias", s + "W_r2_v.bias", _flat),
      (a + "W_R_up/kernel", s + "W_r2_o.weight", _linear),
      (a + "general_vo_pre_bias", s + "W_r2_o.bias", _flat),
      (a + "W_lq_gate/kernel", s + "W_rg_q.weight", _linear),
      (a + "W_lq_gate_b0", s + "W_rg_q.bias", _flat),
      (a + "W_lk_gate/kernel", s + "W_rg_k.weight", _linear),
      (a + "W_lk_gate_b0", s + "W_rg_k.bias", _flat),
      (a + "W_lv_gate/kernel", s + "W_rg_v.weight", _linear),
      (a + "W_lv_gate_b0", s + "W_rg_v.bias", _flat),
      (a + "W_R_gate/kernel", s + "W_rg_o.weight", _linear),
      (a + "W_R_gate_b0", s + "W_rg_o.bias", _flat),
      (a + "common_qk_read_scale", s + "c_qk", None),
      (a + "common_vo_read_scale", s + "c_vo", None),
      (a + "qk_norm/q_norm/scale", s + "q_norm.weight", _plus_one),
      (a + "qk_norm/k_norm/scale", s + "k_norm.weight", _plus_one),
      (a + "out/kernel", s + "W_o.weight", _out_proj),
      (a + "out/bias", s + "W_o.bias", None),
  ]
  for arm in ("q", "k", "v", "o"):
    m += [
        (a + f"general_{arm}_static_gate/kernel", s + f"W_rg_bar_{arm}.weight", _linear),
        (a + f"general_{arm}_static_gate_bias", s + f"W_rg_bar_{arm}.bias", None),
        (a + f"static_{arm}_key", s + f"q_bar_{arm}", _transpose_2d),  # [d_k, n] -> [n, d_k]
    ]
  if writer:
    m += [
        (a + "P_loc_down/kernel", s + "W_w1.weight", _linear),
        (a + "P_loc_up/kernel", s + "W_w2.weight", _linear),
        (a + "P_loc_up/bias", s + "W_w2.bias", _flat),
        (a + "W_gw/kernel", s + "W_wg.weight", _linear),
        (a + "gw_b0", s + "W_wg.bias", None),
        ("memory_mlp/wi/kernel", "ffn_m.fc_in.weight", _linear),
        ("memory_mlp/wo/kernel", "ffn_m.fc_out.weight", _linear),
        ("mlp_address_down/kernel", "ffn_W_w1.weight", _linear),
        ("mlp_address_up/kernel", "ffn_W_w2.weight", _linear),
        ("mlp_address_up/bias", "ffn_W_w2.bias", _flat),
        ("mlp_write_gate/kernel", "ffn_W_wg.weight", _linear),
        ("mlp_write_gate_bias", "ffn_W_wg.bias", None),
    ]
  return m


def _refactored_layer_map(writer: bool):
  """Mapping from refactored `jax/MaxText/layers/loom.py` parameter suffixes to `LoomPythiaDecoderLayer`."""
  a, s = "loom_mha/", "loom_mha."
  m = [
      ("ln_1/scale", "ln_1.weight", None),
      ("ln_1/bias", "ln_1.bias", None),
      ("ln_2/scale", "ln_2.weight", None),
      ("ln_2/bias", "ln_2.bias", None),
      ("ffn_x/wi/kernel", "ffn_x.fc_in.weight", _linear),
      ("ffn_x/wi/bias", "ffn_x.fc_in.bias", None),
      ("ffn_x/wo/kernel", "ffn_x.fc_out.weight", _linear),
      ("ffn_x/wo/bias", "ffn_x.fc_out.bias", None),
      (a + "W_sg/kernel", s + "W_sg.weight", _linear),
      (a + "b_sg", s + "W_sg.bias", None),
      (a + "W_q/kernel", s + "W_q.weight", _linear),
      (a + "W_q/bias", s + "W_q.bias", _flat),
      (a + "W_k/kernel", s + "W_k.weight", _linear),
      (a + "W_k/bias", s + "W_k.bias", _flat),
      (a + "W_r1/kernel", s + "W_r1.weight", _linear),
      (a + "c_qk", s + "c_qk", None),
      (a + "c_vo", s + "c_vo", None),
      (a + "qk_norm/q_norm/scale", s + "q_norm.weight", _plus_one),
      (a + "qk_norm/k_norm/scale", s + "k_norm.weight", _plus_one),
      (a + "W_o/kernel", s + "W_o.weight", _out_proj),
      (a + "W_o/bias", s + "W_o.bias", None),
  ]
  for arm in ("q", "k", "v", "o"):
    m += [
        (a + f"W_r2_{arm}/kernel", s + f"W_r2_{arm}.weight", _linear),
        (a + f"W_r2_{arm}/bias", s + f"W_r2_{arm}.bias", _flat),
        (a + f"W_rg_{arm}/kernel", s + f"W_rg_{arm}.weight", _linear),
        (a + f"b_rg_{arm}", s + f"W_rg_{arm}.bias", _flat),
        (a + f"q_bar_{arm}", s + f"q_bar_{arm}", None),
        (a + f"W_rg_bar_{arm}/kernel", s + f"W_rg_bar_{arm}.weight", _linear),
        (a + f"b_rg_bar_{arm}", s + f"W_rg_bar_{arm}.bias", None),
    ]
  if writer:
    m += [
        (a + "W_w1/kernel", s + "W_w1.weight", _linear),
        (a + "W_w2/kernel", s + "W_w2.weight", _linear),
        (a + "W_w2/bias", s + "W_w2.bias", _flat),
        (a + "W_wg/kernel", s + "W_wg.weight", _linear),
        (a + "b_wg", s + "W_wg.bias", None),
        ("ffn_m/wi/kernel", "ffn_m.fc_in.weight", _linear),
        ("ffn_m/wo/kernel", "ffn_m.fc_out.weight", _linear),
        ("ffn_W_w1/kernel", "ffn_W_w1.weight", _linear),
        ("ffn_W_w2/kernel", "ffn_W_w2.weight", _linear),
        ("ffn_W_w2/bias", "ffn_W_w2.bias", _flat),
        ("ffn_W_wg/kernel", "ffn_W_wg.weight", _linear),
        ("ffn_b_wg", "ffn_W_wg.bias", None),
    ]
  return m


def infer_loompythia_config(params: dict[str, np.ndarray], **kwargs) -> LoomPythiaConfig:
  if _is_refactored_tree(params):
    scanned = params["decoder/layers/ffn_x/wi/kernel"]
    D, n_scan, width = scanned.shape
    n, _, d, _ = params["decoder/layers/loom_mha/W_o/kernel"].shape
    d_k = params["decoder/layers/loom_mha/b_sg"].shape[0]
    r_dim = params["decoder/layers/loom_mha/W_q/bias"].shape[-1]
    d_r = params["decoder/layers/loom_mha/W_r1/kernel"].shape[-1]
    last_width = params["decoder/last_layer/ffn_x/wi/kernel"].shape[-1]
    ffn_m_dim = params["decoder/layers/ffn_m/wi/kernel"].shape[-1]
  else:
    scanned = params["decoder/layers/block/mlp/wi/kernel"]
    D, n_scan, width = scanned.shape
    n, _, d, _ = params["decoder/layers/block/self_attention/out/kernel"].shape
    d_k = params["decoder/layers/block/self_attention/M_shift_gate_b0"].shape[0]
    r_dim = params["decoder/layers/block/self_attention/query/bias"].shape[-1]
    d_r = params["decoder/layers/block/self_attention/full_m_read_down/kernel"].shape[-1]
    last_width = params["decoder/final_local_layer/block/mlp/wi/kernel"].shape[-1]
    ffn_m_dim = params["decoder/layers/block/memory_mlp/wi/kernel"].shape[-1]

  return LoomPythiaConfig(
      vocab_size=params["token_embedder/embedding"].shape[0],
      hidden_size=D,
      num_hidden_layers=n_scan + 1,
      num_attention_heads=n,
      head_dim=d,
      d_k=d_k,
      d_r=d_r,
      rope_dim=r_dim,
      intermediate_size=width,
      last_ffn_x_dim=last_width,
      ffn_m_dim=ffn_m_dim,
      **kwargs,
  )


def convert_params_to_state_dict(params: dict[str, np.ndarray], config: LoomPythiaConfig) -> dict[str, torch.Tensor]:
  out, used = {}, set()
  refactored = _is_refactored_tree(params)

  def get(key: str) -> np.ndarray:
    assert key in params, f"Missing key in source params: {key}"
    used.add(key)
    return params[key]

  def put(name: str, value: np.ndarray):
    assert name not in out, f"Duplicate target key: {name}"
    out[name] = torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32))

  put("model.embed_tokens.weight", get("token_embedder/embedding"))
  if refactored:
    e = "decoder/embed_write/"
    put("model.embed_write.W_e.weight", _linear(get(e + "W_e/kernel")))
    put("model.embed_write.W_w1.weight", _linear(get(e + "W_w1/kernel")))
    put("model.embed_write.W_w2.weight", _linear(get(e + "W_w2/kernel")))
    put("model.embed_write.W_w2.bias", _flat(get(e + "W_w2/bias")))
    put("model.embed_write.W_wg.weight", _linear(get(e + "W_wg/kernel")))
    put("model.embed_write.W_wg.bias", get(e + "b_wg"))
  else:
    e = "decoder/embedding_bam_write/"
    put("model.embed_write.W_e.weight", _linear(get(e + "W_emb_u/kernel")))
    put("model.embed_write.W_w1.weight", _linear(get(e + "W_emb_v_down/kernel")))
    put("model.embed_write.W_w2.weight", _linear(get(e + "W_emb_v_up/kernel")))
    put("model.embed_write.W_w2.bias", _flat(get(e + "W_emb_v_up/bias")))
    put("model.embed_write.W_wg.weight", _linear(get(e + "W_emb_g/kernel")))
    put("model.embed_write.W_wg.bias", get(e + "emb_gw_b0"))

  put("model.norm.weight", get("decoder/lm_head/decoder_norm/scale"))
  put("model.norm.bias", get("decoder/lm_head/decoder_norm/bias"))
  put("lm_head.weight", get("decoder/lm_head/logits_dense").T)

  last = config.num_hidden_layers - 1
  for i in range(config.num_hidden_layers):
    if refactored:
      root, index = ("decoder/last_layer/", None) if i == last else ("decoder/layers/", i)
      lmap = _refactored_layer_map(config.has_write(i))
    else:
      root, index = ("decoder/final_local_layer/block/", None) if i == last else ("decoder/layers/block/", i)
      lmap = _legacy_layer_map(config.has_write(i))

    for src, dst, fn in lmap:
      value = get(root + src)
      if index is not None:
        value = np.take(value, index, axis=1)
      put(f"model.layers.{i}.{dst}", fn(value) if fn else value)

  unused = sorted(set(params) - used)
  assert not unused, f"Unused source leaves ({len(unused)}): {unused[:10]}"
  return out


def save_hf_directory(state: dict[str, torch.Tensor], config: LoomPythiaConfig, out_dir: str, dtype_str: str,
                      tokenizer_dir: str = None):
  config.auto_map = {
      "AutoConfig": "configuration_loompythia.LoomPythiaConfig",
      "AutoModelForCausalLM": "modeling_loompythia.LoomPythiaForCausalLM",
  }
  config.architectures = ["LoomPythiaForCausalLM"]

  with torch.device("meta"):
    model = LoomPythiaForCausalLM(config)
  expected = {k: tuple(v.shape) for k, v in model.state_dict().items()}
  assert set(expected) == set(state), f"State dict key mismatch: {sorted(set(expected) ^ set(state))[:10]}"
  for k, v in state.items():
    assert tuple(v.shape) == expected[k], f"Shape mismatch at {k}: got {tuple(v.shape)}, expected {expected[k]}"

  dtype = getattr(torch, dtype_str)
  os.makedirs(out_dir, exist_ok=True)
  st_path = os.path.join(out_dir, "model.safetensors")
  save_file({k: v.to(dtype).contiguous() for k, v in state.items()}, st_path, metadata={"format": "pt"})
  os.chmod(st_path, 0o644)
  config.save_pretrained(out_dir)

  for fname in STANDALONE_CODE:
    shutil.copy(os.path.join(LOOMPYTHIA_DIR, fname), out_dir)

  with open(os.path.join(out_dir, "generation_config.json"), "w") as f:
    json.dump(
        {"bos_token_id": config.bos_token_id, "eos_token_id": config.eos_token_id, "pad_token_id": config.eos_token_id},
        f, indent=2,
    )

  if tokenizer_dir:
    for fname in os.listdir(tokenizer_dir):
      if fname.startswith("tokenizer") or fname in ("special_tokens_map.json", "vocab.json", "merges.txt"):
        shutil.copy(os.path.join(tokenizer_dir, fname), out_dir)

  total_params = sum(v.numel() for v in state.values())
  h = hashlib.sha256()
  with open(st_path, "rb") as f:
    while chunk := f.read(16 * 1024 * 1024):
      h.update(chunk)
  print(f"CONVERTED {out_dir} | params={total_params:,} | dtype={dtype_str} | sha256={h.hexdigest()[:16]}...")


def main():
  parser = argparse.ArgumentParser(description="Convert MaxText LoomPythia checkpoint to HuggingFace format.")
  src = parser.add_mutually_exclusive_group(required=True)
  src.add_argument("--ckpt", help="Path to Orbax MaxText checkpoint `items` directory")
  src.add_argument("--params", help="Path to flat `.npz` parameter archive")
  parser.add_argument("--out-dir", required=True, help="Output HuggingFace model directory")
  parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
  parser.add_argument("--tokenizer", default=None, help="Directory containing GPT-NeoX / Pythia tokenizer files")
  parser.add_argument("--export-npz", default=None, help="Optional path to also save flat FP32 `.npz` parameters")
  args = parser.parse_args()

  if args.ckpt:
    params = load_orbax_params_only(args.ckpt)
    if args.export_npz:
      np.savez(args.export_npz, **params)
      print(f"EXPORTED_NPZ {args.export_npz}")
  else:
    data = np.load(args.params)
    params = {k: data[k] for k in data.files if k != "__config__"}

  config = infer_loompythia_config(params, dtype=args.dtype)
  state = convert_params_to_state_dict(params, config)
  save_hf_directory(state, config, args.out_dir, args.dtype, args.tokenizer)


if __name__ == "__main__":
  main()
