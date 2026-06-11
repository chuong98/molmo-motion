#!/usr/bin/env bash
# Launch the per-checkpoint MolmoSpaces evaluation sidecar for a MolmoBot
# training run. Polls --save_folder for new step<N>/ checkpoints and runs
# a full benchmark rollout per checkpoint.
#
# Usage:
#   bash launch_scripts/eval.sh \\
#       --save_folder /path/to/train_run \\
#       --eval_out    /path/to/eval_output \\
#       --benchmark_dir /path/to/benchmark \\
#       [--eval_every_n_steps 10000] \\
#       [--stop_after_step 100000] \\
#       [--eval_mode {hybrid,standalone}] \\
#       [--pretrained_ckpt_path /path/to/MolmoBot-DROID]
#
# Prerequisites: MolmoBot + molmo_spaces installed (see project README).
# Required env vars:
#   MOLMOBOT_REPO   path to the MolmoBot clone (for olmo.* imports)
#
# Recommended env vars for MuJoCo + JAX on EGL nodes:
#   MUJOCO_GL=egl
#   PYOPENGL_PLATFORM=egl
#   JAX_PLATFORMS=cpu
set -euo pipefail
: "${MOLMOBOT_REPO:?set MOLMOBOT_REPO to the MolmoBot clone root}"

export PYTHONPATH="${MOLMOBOT_REPO}:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"

HERE="$(cd "$(dirname "$0")" && pwd)"
exec python "$HERE/../scripts/eval_sidecar.py" "$@"
