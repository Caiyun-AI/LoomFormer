"""Unit, TPU Pallas parity, `bf16_dot_fp32_output` + vocab-tiling, and `maxtext-v0.2.0` `train_step` tests."""
import os
import sys
import unittest

import jax
from jax.experimental.compilation_cache import compilation_cache
from jax.sharding import Mesh
import jax.numpy as jnp
import numpy as np

JAX_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if JAX_ROOT not in sys.path:
  sys.path.insert(0, JAX_ROOT)

os.makedirs("/tmp/jax_cache_loomtest", exist_ok=True)
compilation_cache.set_cache_dir("/tmp/jax_cache_loomtest")

from loomformer import LoomFormer, LoomPythia, build_loom_model, extract_loom_and_maxtext_argv  # noqa: E402
from maxtext.configs import pyconfig  # noqa: E402
from maxtext.optimizers import optimizers  # noqa: E402
from maxtext.trainers.pre_train import train as mt_train  # noqa: E402
from maxtext.utils import max_utils, maxtext_utils, sharding  # noqa: E402


def _init_configs(yml_name: str, extra_args: list[str]):
  yml_path = os.path.join(JAX_ROOT, "configs", yml_name)
  argv = ["train_loom.py", yml_path, "run_name=loom_test", "base_output_directory=/tmp/loom_out",
          "enable_checkpointing=False", "dataset_type=synthetic", "log_config=False"] + extra_args
  loom_cfg, mt_argv, tmp_yml = extract_loom_and_maxtext_argv(argv)
  try:
    mt_cfg = pyconfig.initialize(mt_argv)
  finally:
    if os.path.exists(tmp_yml):
      os.remove(tmp_yml)
  devices = maxtext_utils.create_device_mesh(mt_cfg)
  mesh = Mesh(devices, mt_cfg.mesh_axes)
  return mt_cfg, loom_cfg, mesh


class LoomFormerZeroPatchTest(unittest.TestCase):

  def test_exact_paper_param_counts_and_submodule_structure(self):
    """Verify exact integer parameter counts across all 5 YAML configs for `LoomFormer` & `LoomPythia`."""
    expected = {
        "loomformer_medium.yml": (LoomFormer, 432_112_228),
        "loomformer_large.yml": (LoomFormer, 777_849_500),
        "loomformer_xl.yml": (LoomFormer, 1_432_396_996),
        "loomformer_xl_situ.yml": (LoomFormer, 1_432_396_996),
        "loompythia_2p8b.yml": (LoomPythia, 2_775_190_944),
    }
    x = jnp.zeros((1, 256), dtype=jnp.int32)
    pos = jnp.broadcast_to(jnp.arange(256, dtype=jnp.int32)[None, :], (1, 256))
    rng = jax.random.PRNGKey(0)

    for yml_name, (exp_cls, exp_cnt) in expected.items():
      mt_cfg, loom_cfg, mesh = _init_configs(yml_name, ["per_device_batch_size=1", "max_target_length=256", "num_vocab_tiling=1"])
      model = build_loom_model(config=mt_cfg, mesh=mesh, loom_cfg=loom_cfg)
      self.assertIsInstance(model, exp_cls)
      av = jax.eval_shape(lambda k: model.init(k, x, pos, enable_dropout=False), rng)
      cnt = sum(int(np.prod(v.shape)) for v in jax.tree_util.tree_leaves(av["params"]))
      self.assertEqual(cnt, exp_cnt, f"{yml_name}: got {cnt:,}, expected {exp_cnt:,}")
      dec = av["params"]["decoder"]
      self.assertIn("m_write", dec["embed_write"])
      self.assertIn("read_q", dec["layers"]["loom_mha"])
      self.assertIn("m_shift", dec["layers"]["loom_mha"])
      self.assertIn("m_write", dec["layers"]["loom_mha"])
      self.assertIn("ffn_m_write", dec["layers"])
      self.assertNotIn("m_write", dec["last_layer"]["loom_mha"])
      self.assertNotIn("ffn_m_write", dec["last_layer"])

  def _check_pure_vs_pallas(self, yml_name: str, extra: list[str], dtype_str: str,
                            fwd_atol: float, grad_rtol: float, grad_atol: float):
    if jax.default_backend() != "tpu":
      self.skipTest("Pallas parity test requires TPU backend.")

    base_args = [
        f"dtype={dtype_str}",
        f"loom_logits_bf16_dot_fp32_out={dtype_str == 'bfloat16'}",
        "weight_dtype=float32",
        "matmul_precision=highest",
        "num_vocab_tiling=1",
        "per_device_batch_size=2",
        "max_target_length=256",
        "sa_block_q=128", "sa_block_kv=128", "sa_block_kv_compute=128",
        "sa_block_q_dkv=128", "sa_block_kv_dkv=128", "sa_block_kv_dkv_compute=128",
        "sa_block_q_dq=128", "sa_block_kv_dq=128",
    ] + extra

    mt_ref, lc_ref, mesh = _init_configs(yml_name, base_args + ["loom_pallas=False", "loom_splash=False"])
    mt_pal, lc_pal, _ = _init_configs(yml_name, base_args + ["loom_pallas=True", "loom_splash=True"])
    mdl_ref = build_loom_model(config=mt_ref, mesh=mesh, loom_cfg=lc_ref)
    mdl_pal = build_loom_model(config=mt_pal, mesh=mesh, loom_cfg=lc_pal)

    b, t = 2, 256
    k_init, k_tok, k_tgt = jax.random.split(jax.random.PRNGKey(42), 3)
    tokens = jax.random.randint(k_tok, (b, t), 1, mt_pal.vocab_size)
    targets = jax.random.randint(k_tgt, (b, t), 1, mt_pal.vocab_size)
    seg = jnp.concatenate([jnp.ones((b, 128), jnp.int32), 2 * jnp.ones((b, 128), jnp.int32)], axis=1)
    pos = jnp.broadcast_to(jnp.concatenate([jnp.arange(128), jnp.arange(128)])[None, :], (b, t)).astype(jnp.int32)

    vars0 = mdl_pal.init(k_init, tokens, pos, decoder_segment_ids=seg, enable_dropout=False)
    # Verify `loom_clone_vo_init` cloned `read_o/W_r2/kernel` into `read_v/W_r2/kernel`
    from flax import linen as nn
    mha0 = vars0["params"]["decoder"]["layers"]["loom_mha"]
    np.testing.assert_array_equal(
        np.asarray(nn.meta.unbox(mha0["read_v"]["W_r2"]["kernel"])),
        np.asarray(nn.meta.unbox(mha0["read_o"]["W_r2"]["kernel"])),
    )

    flat_p, treedef = jax.tree_util.tree_flatten_with_path(vars0["params"])
    pert = []
    for idx, (path, leaf) in enumerate(flat_p):
      name = "/".join(str(k) for k in path)
      noise = 0.02 * jax.random.normal(jax.random.PRNGKey(1000 + idx), leaf.shape, leaf.dtype)
      pert.append(leaf + noise if any(tag in name for tag in ("W_rg", "W_sg", "q_bar")) else leaf)
    params = jax.tree_util.tree_unflatten(treedef, pert)

    one_hot = jax.nn.one_hot(targets, mt_pal.vocab_size)

    def loss_fn(mdl, p):
      logits = mdl.apply({"params": p}, tokens, pos, decoder_segment_ids=seg, enable_dropout=False)
      xent, _ = max_utils.cross_entropy_with_logits(logits, one_hot, z_loss=0.0)
      return jnp.mean(xent), xent

    (l_ref, x_ref), g_ref = jax.jit(jax.value_and_grad(lambda p: loss_fn(mdl_ref, p), has_aux=True))(params)
    (l_pal, x_pal), g_pal = jax.jit(jax.value_and_grad(lambda p: loss_fn(mdl_pal, p), has_aux=True))(params)

    self.assertLess(float(jnp.abs(l_ref - l_pal)), fwd_atol)
    self.assertLess(float(jnp.max(jnp.abs(x_ref - x_pal))), fwd_atol * 30)

    min_cos = 0.999 if dtype_str == "float32" else 0.99
    for (path, gr), (_, gp) in zip(jax.tree_util.tree_leaves_with_path(g_ref), jax.tree_util.tree_leaves_with_path(g_pal)):
      name = "/".join(str(k) for k in path)
      a = np.asarray(gp, dtype=np.float64).ravel()
      b_arr = np.asarray(gr, dtype=np.float64).ravel()
      self.assertTrue(np.all(np.isfinite(a)), f"Non-finite grad in {name}")
      nr, np_ = float(np.linalg.norm(b_arr)), float(np.linalg.norm(a))
      err = float(np.linalg.norm(a - b_arr))
      self.assertLessEqual(err, grad_atol + grad_rtol * nr, f"Grad L2 mismatch in {name}")
      if nr > grad_atol and b_arr.size > 1:
        cos = float(np.dot(a, b_arr) / (nr * max(np_, 1e-12)))
        self.assertGreater(cos, min_cos, f"Grad cosine {cos:.6f} <= {min_cos} in {name}")

  def test_pure_vs_pallas_loomformer_odd_head_dim_and_situ_nondiv8_heads(self):
    # Medium: odd `head_dim=75`, `n=16`
    extra_med = ["base_num_decoder_layers=4", "vocab_size=1024"]
    self._check_pure_vs_pallas("loomformer_medium.yml", extra_med, "float32", 1e-3, 4.5e-2, 5e-3)
    self._check_pure_vs_pallas("loomformer_medium.yml", extra_med, "bfloat16", 2e-2, 8e-2, 3e-2)
    # XL-SiTU style: `n=4` (`n % 8 != 0`, exercising `_whole_keys` HBM slab path) + SiTU-GLU
    extra_xl = [
        "base_emb_dim=384", "base_num_query_heads=4", "base_num_kv_heads=4", "head_dim=96",
        "base_mlp_dim=768", "base_num_decoder_layers=3",
        "loom_d_k=8", "loom_d_r=32", "loom_rope_dim=24",
        "loom_ffn_m_dim=384", "loom_last_ffn_x_dim=1152", "vocab_size=1024",
    ]
    self._check_pure_vs_pallas("loomformer_xl_situ.yml", extra_xl, "bfloat16", 2e-2, 8e-2, 3e-2)

  def test_pure_vs_pallas_loompythia_qknorm_and_bf16_dot_fp32(self):
    extra = [
        "base_emb_dim=640", "base_num_query_heads=8", "base_num_kv_heads=8", "head_dim=80",
        "base_mlp_dim=1792", "base_num_decoder_layers=4",
        "loom_d_k=16", "loom_d_r=64", "loom_rope_dim=20",
        "loom_ffn_m_dim=640", "loom_last_ffn_x_dim=2432", "vocab_size=1024",
    ]
    self._check_pure_vs_pallas("loompythia_2p8b.yml", extra, "float32", 1e-3, 4.5e-2, 5e-3)
    self._check_pure_vs_pallas("loompythia_2p8b.yml", extra, "bfloat16", 2e-2, 8e-2, 3e-2)

  def test_official_maxtext_train_step_with_vocab_tiling(self):
    """Run 5 compiled official `maxtext-v0.2.0` `train_step` iterations (`num_vocab_tiling=2` + `adam_pax` + `adamw_mask`)."""
    mt_cfg, loom_cfg, mesh = _init_configs("loompythia_2p8b.yml", [
        "base_emb_dim=640", "base_num_query_heads=8", "base_num_kv_heads=8", "head_dim=80",
        "base_mlp_dim=1792", "base_num_decoder_layers=4",
        "loom_d_k=16", "loom_d_r=64", "loom_rope_dim=20",
        "loom_ffn_m_dim=640", "loom_last_ffn_x_dim=2432",
        "vocab_size=512", "num_vocab_tiling=2",
        "per_device_batch_size=2", "max_target_length=256",
        "learning_rate=1e-3", "steps=20", "learning_rate_schedule_steps=20",
        "sa_block_q=128", "sa_block_kv=128", "sa_block_kv_compute=128",
        "sa_block_q_dkv=128", "sa_block_kv_dkv=128", "sa_block_kv_dkv_compute=128",
        "sa_block_q_dq=128", "sa_block_kv_dq=128",
    ])
    model = build_loom_model(config=mt_cfg, mesh=mesh, loom_cfg=loom_cfg)
    lr_sched = maxtext_utils.create_learning_rate_schedule(mt_cfg)
    tx = optimizers.get_optimizer(mt_cfg, lr_sched, model)
    state, _, state_mesh_shardings, _ = maxtext_utils.setup_training_state(
        model, None, tx, mt_cfg, jax.random.PRNGKey(0), mesh, None
    )
    params_shardings, state_mesh_shardings = sharding.maybe_update_params_sharding_with_opt(mt_cfg, state_mesh_shardings)

    b, t = 2, 256
    tokens = jax.random.randint(jax.random.PRNGKey(7), (b, t), 1, mt_cfg.vocab_size)
    targets = jnp.roll(tokens, -1, axis=1)
    pos = jnp.broadcast_to(jnp.arange(t, dtype=jnp.int32)[None, :], (b, t))
    seg = jnp.ones((b, t), dtype=jnp.int32)
    batch = {
        "inputs": tokens, "inputs_position": pos, "inputs_segmentation": seg,
        "targets": targets, "targets_segmentation": seg,
    }

    p_step = jax.jit(lambda st, d, r: mt_train.train_step(model, mt_cfg, state_mesh_shardings, params_shardings, st, d, r))
    from flax.linen import partitioning as nn_partitioning
    losses = []
    with jax.set_mesh(mesh), nn_partitioning.axis_rules(mt_cfg.logical_axis_rules):
      for i in range(5):
        state, metrics = p_step(state, batch, jax.random.PRNGKey(i))
        loss_val = float(metrics["scalar"]["learning/loss"])
        self.assertTrue(np.isfinite(loss_val))
        losses.append(loss_val)

    self.assertLess(losses[-1], losses[0], f"Loss did not decrease over 5 steps: {losses}")


if __name__ == "__main__":
  unittest.main()
