"""YAML-config training entry point for MolmoMotion.

Thin wrapper over the `molmo_motion.train.run_trainer` torchrun entry-point
for resuming or re-running from a full `TrainConfig` YAML — e.g. the
`config.yaml` that every training run writes into its `--save_folder`:

    torchrun --nproc_per_node=8 launch_scripts/train.py \\
        checkpoints/MolmoMotion-Stage1/config.yaml \\
        --save_folder=outputs/molmomotion_rerun

YAML config + dotted-path CLI overrides; per-key overrides are passed
through verbatim.

To launch the released training recipes from scratch, use
`launch_scripts/sft.py` instead (see the Training section of the README).
"""

from __future__ import annotations

import sys

from molmo_motion.exceptions import OLMoCliError
from molmo_motion.train.run_trainer import run_trainer
from molmo_motion.train.trainer_config import TrainConfig
from molmo_motion.util import clean_opt, prepare_torchrun_environment


def main():
    prepare_torchrun_environment()

    try:
        yaml_path, args_list = sys.argv[1], sys.argv[2:]
    except IndexError as e:
        raise OLMoCliError(
            f"Usage: {sys.argv[0]} <CONFIG_YAML> [DOTTED-PATH=VALUE ...]\n"
            f"Example: {sys.argv[0]} configs/pretrain_h3.yaml --save_folder=outputs/run1"
        ) from e

    cfg = TrainConfig.load(yaml_path, [clean_opt(s) for s in args_list])
    run_trainer(cfg)


if __name__ == "__main__":
    main()
