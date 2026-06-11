#!/usr/bin/env bash
# Minimal end-to-end smoke test. Edit examples/tasks_example.json so video_path
# points at two real .mp4 files, then run this from the repo root.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python run_pipeline.py \
    --tasks examples/tasks_example.json \
    --config configs/human_manipulation.yaml \
    --work_dir ./runs/example \
    --start_stage 1 --end_stage 6
