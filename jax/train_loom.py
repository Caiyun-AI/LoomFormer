"""Training entrypoint for LoomFormer & LoomPythia on official `maxtext-v0.2.0` (zero patches).

Usage:
  python jax/train_loom.py jax/configs/loomformer_medium.yml \
      run_name=loomformer_medium \
      base_output_directory=gs://your-bucket/runs \
      dataset_type=hf hf_path=EleutherAI/pile
"""
import os
import sys

JAX_ROOT = os.path.dirname(os.path.abspath(__file__))
if JAX_ROOT not in sys.path:
  sys.path.insert(0, JAX_ROOT)

from loomformer import build_loom_model, extract_loom_and_maxtext_argv  # noqa: E402


def main(argv=None):
  argv = list(sys.argv if argv is None else argv)
  loom_cfg, mt_argv, tmp_yml = extract_loom_and_maxtext_argv(argv)

  try:
    from maxtext.trainers.pre_train import train as mt_train
    from maxtext.utils import model_creation_utils

    def _create_model(config, mesh, quant, model_mode="train", rngs=None):
      del rngs
      return build_loom_model(config=config, mesh=mesh, quant=quant, loom_cfg=loom_cfg, model_mode=model_mode)

    model_creation_utils.get_transformer_model = _create_model
    mt_train.main(mt_argv)
  finally:
    if os.path.exists(tmp_yml):
      os.remove(tmp_yml)


if __name__ == "__main__":
  main()
