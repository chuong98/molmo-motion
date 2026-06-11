"""PointMotionBench all-point evaluation driver.

Runs `molmo_motion.eval.full_rollout` against each requested benchmark
subset (HOT3D, WorldTrack, DAVIS), then aggregates per-benchmark
predictions into ADE / FDE / PWT and writes a top-level `summary.json`.

Usage:

    torchrun --nproc-per-node=8 launch_scripts/eval_pointmotionbench.py \\
        checkpoints/MolmoMotion-H3-F30/step10000 \\
        --benchmarks hot3d,worldtrack,davis \\
        --all_points \\
        --fixed_t0 \\
        --device_batch_size=2 \\
        --output_dir eval_out/MolmoMotion-H3-F30

The script delegates the actual per-clip rollout to the existing
`full_rollout.main()` entry-point (the same engine used internally for
auto-eval). It then post-processes the per-example predictions JSONL into
the metric summary documented in the README.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


# Map the public benchmark names to the dataset-suffix the inner driver
# expects.  The benchmark loaders live in
# `molmo_motion/data/trajectory_3d_dataset.py` under
# `hot3d_bench`/`worldtrack_bench`/`davis_bench`.
_BENCH_TO_SUFFIX = {
    "hot3d":      "hot3d_bench",
    "worldtrack": "worldtrack_bench",
    "davis":      "davis_bench",
}

# Thresholds for PWT, in meters, per the paper.
_PWT_THRESHOLDS = (0.01, 0.02, 0.05, 0.10, 0.20)


def _compute_metrics(predictions_jsonl: Path) -> dict:
    """Read predictions.jsonl and compute ADE / FDE / PWT averaged across
    all visible (point, future-frame) entries."""
    ade_l2 = []                       # per-(point, frame) L2 for ADE
    fde_l2 = []                       # per-point final-frame L2 for FDE
    pwt_hits = {t: [] for t in _PWT_THRESHOLDS}
    n_clips = 0

    with open(predictions_jsonl) as f:
        for line in f:
            row = json.loads(line)
            gt = np.asarray(row["gt_future"], dtype=np.float32)        # (P, F, 3)
            pred = np.asarray(row["pred_future"], dtype=np.float32)    # (P, F, 3)
            vis = np.asarray(row["gt_future_vis"], dtype=bool)         # (P, F)
            if gt.shape != pred.shape or vis.shape != gt.shape[:2]:
                continue

            err = np.linalg.norm(pred - gt, axis=-1)                   # (P, F)
            err_vis = err[vis]
            if err_vis.size:
                ade_l2.extend(err_vis.tolist())
                for tau in _PWT_THRESHOLDS:
                    pwt_hits[tau].extend((err_vis <= tau).tolist())

            # FDE: last visible frame per point.
            for p_idx in range(gt.shape[0]):
                v = vis[p_idx]
                if v.any():
                    last_visible = np.where(v)[0].max()
                    fde_l2.append(float(err[p_idx, last_visible]))
            n_clips += 1

    pwt_per_tau = {tau: (float(np.mean(hits)) if hits else 0.0)
                   for tau, hits in pwt_hits.items()}
    return {
        "ADE": float(np.mean(ade_l2)) if ade_l2 else 0.0,
        "FDE": float(np.mean(fde_l2)) if fde_l2 else 0.0,
        "PWT": float(np.mean(list(pwt_per_tau.values()))) if pwt_per_tau else 0.0,
        "PWT_per_threshold": pwt_per_tau,
        "n_clips": n_clips,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoint_dir",
                    help="Trained MolmoMotion checkpoint directory.")
    ap.add_argument("--benchmarks", default="hot3d,worldtrack,davis",
                    help="Comma-separated list of subsets. Default = all 3.")
    ap.add_argument("--output_dir", required=True,
                    help="Where to write per-benchmark predictions + summary.")
    ap.add_argument("--num_points", type=int, default=8)
    ap.add_argument("--history", type=int, default=3)
    ap.add_argument("--future", type=int, default=30)
    ap.add_argument("--device_batch_size", type=int, default=2)
    ap.add_argument("--all_points", action="store_true",
                    help="Chunk every visible point into groups of P; "
                         "average metrics across chunks. Recommended.")
    ap.add_argument("--fixed_t0", action="store_true",
                    help="Pin t₀ = H − 1 (deterministic across runs).")
    ap.add_argument("--n_samples", type=int, default=1,
                    help="Best-of-N decoding (paper uses 5). Default 1.")
    args = ap.parse_args()

    benches = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    bad = [b for b in benches if b not in _BENCH_TO_SUFFIX]
    if bad:
        raise ValueError(
            f"Unknown benchmark(s) {bad}. Choose from {sorted(_BENCH_TO_SUFFIX)}."
        )

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # Lazy-import so `--help` works without torch.
    from molmo_motion.eval import full_rollout

    summary: dict = {}
    for bench in benches:
        bench_dir = output_root / bench
        bench_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_{_BENCH_TO_SUFFIX[bench]}_p{args.num_points}_h{args.history}_f{args.future}"

        # Build the argv that `full_rollout.main()` expects.
        inner_argv = [
            "full_rollout",
            "--checkpoint_dir", args.checkpoint_dir,
            "--output_dir", str(bench_dir),
            "--dataset_name", "trajectory_3d",
            "--dataset_suffix", suffix,
            "--split", "test",
            "--num_points", str(args.num_points),
            "--history", str(args.history),
            "--future", str(args.future),
            "--stride", str(args.future),                  # one-shot, no rolling
            "--n_rollouts", "1",
        ]
        if args.all_points:
            inner_argv.append("--all_points")
        if args.fixed_t0:
            inner_argv += ["--fixed_t0", str(args.history - 1)]

        saved_argv = sys.argv
        try:
            sys.argv = inner_argv
            full_rollout.main()
        finally:
            sys.argv = saved_argv

        pred_jsonl = bench_dir / "predictions.jsonl"
        if not pred_jsonl.exists():
            print(f"[warn] no predictions written for {bench}, skipping metrics")
            continue
        metrics = _compute_metrics(pred_jsonl)
        (bench_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        summary[bench] = {
            "ADE": metrics["ADE"],
            "FDE": metrics["FDE"],
            "PWT": metrics["PWT"],
            "n_clips": metrics["n_clips"],
        }

    (output_root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
