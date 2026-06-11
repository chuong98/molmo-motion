"""Evaluation entry point for MolmoMotion.

Thin wrapper over the internal `molmo_motion.eval.model_evaluator.EvalConfig`
torchrun entry-point. Run with:

    torchrun --nproc_per_node=8 launch_scripts/eval.py \\
        configs/eval_h3.yaml \\
        --load_path=outputs/molmomotion_pretrain_h3/step20000

YAML config + dotted-path CLI overrides — same convention the internal
evaluation loop uses.
"""

from __future__ import annotations

import sys

from molmo_motion.eval.model_evaluator import EvalConfig
from molmo_motion.exceptions import OLMoCliError
from molmo_motion.util import clean_opt, prepare_torchrun_environment


def main():
    prepare_torchrun_environment()

    try:
        yaml_path, args_list = sys.argv[1], sys.argv[2:]
    except IndexError as e:
        raise OLMoCliError(
            f"Usage: {sys.argv[0]} <CONFIG_YAML> [DOTTED-PATH=VALUE ...]\n"
            f"Example: {sys.argv[0]} configs/eval_h3.yaml --load_path=outputs/step20000"
        ) from e

    cfg = EvalConfig.load(yaml_path, [clean_opt(s) for s in args_list])
    cfg.build().run()


if __name__ == "__main__":
    main()
