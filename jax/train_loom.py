"""Training entrypoint for LoomFormer & LoomPythia on official `maxtext-v0.2.0` (zero patches).

Usage:
  python jax/train_loom.py jax/configs/loompythia_2p8b.yml \
      run_name=loompythia_2p8b \
      base_output_directory=gs://your-bucket/runs \
      dataset_type=hf hf_path=EleutherAI/pile
"""
import os
import sys

JAX_ROOT = os.path.dirname(os.path.abspath(__file__))
if JAX_ROOT not in sys.path:
  sys.path.insert(0, JAX_ROOT)

from loomformer.config import extract_loom_and_maxtext_argv  # noqa: E402
from loomformer.loom import LoomTransformer  # noqa: E402


def main(argv=None):
  argv = list(sys.argv if argv is None else argv)
  loom_cfg, mt_argv, tmp_yml = extract_loom_and_maxtext_argv(argv)

  try:
    from maxtext.trainers.pre_train import train as mt_train
    from maxtext.utils import model_creation_utils

    def _build_loom_model(config, mesh, quant, model_mode="train", rngs=None):
      del rngs
      return LoomTransformer(config=config, mesh=mesh, quant=quant, loom_cfg=loom_cfg, model_mode=model_mode)

    model_creation_utils.get_transformer_model = _build_loom_model
    mt_train.main(mt_argv)
  finally:
    if os.path.exists(tmp_yml):
      os.remove(tmp_yml)


if __name__ == "__main__":
  main()
