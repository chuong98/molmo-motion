"""Per-checkpoint MolmoSpaces evaluation sidecar.

This script polls a MolmoBot training run's ``save_folder`` for new
``step<N>/`` checkpoints and, for each new one, runs a full MolmoSpaces
rollout against a benchmark and appends the success rate to a
``summary.json`` log.

Two eval modes are supported:

* ``--eval_mode standalone`` — our policy drives every step from t=0.
  Uses ``FrankaState8ClampAbsPosConfig`` from MolmoBot.

* ``--eval_mode hybrid`` — the released MolmoBot-DROID policy drives
  the sim until the gripper closes on the pickup object (``policy_phase
  >= 5`` plus one execute-horizon tick). Our policy then takes over with
  real-history conditioning. Failures after handoff are attributable to
  our policy alone, isolating it from any sim-grasp domain gap.

Resumability: the script re-reads ``summary.json`` on startup. Steps
already present are skipped. Pass ``--eval_every_n_steps N`` to only
evaluate multiples of N (e.g. ``--eval_every_n_steps 10000``).

Usage:

    python scripts/eval_sidecar.py \\
        --save_folder /path/to/training_run \\
        --eval_out    /path/to/eval_output \\
        --benchmark_dir /path/to/benchmark_root \\
        --eval_every_n_steps 10000 \\
        --stop_after_step 100000 \\
        --eval_mode hybrid \\
        --pretrained_ckpt_path /path/to/MolmoBot-DROID

Dependencies: the MolmoBot codebase must be importable (``olmo.*``) and
``molmo_spaces`` must be installed (with its MuJoCo + JAX deps).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Set


log = logging.getLogger(__name__)


# Module-level mutable holder used by ``MonitorHybridEvalConfig``'s
# ``default_factory`` below. Each ``_run_eval(eval_mode="hybrid")`` call
# writes a freshly-built ``PretrainedToOursHybridPolicyConfig`` here, and
# the eval pipeline picks it up at instantiation time.
#
# This indirection exists because pydantic snapshots field defaults at
# class-creation time, so mutating a class attribute later does not
# propagate. The factory pattern works around that.
_HYBRID_PC_HOLDER: dict = {"current": None}


def _factory_hybrid_pc():
    pc = _HYBRID_PC_HOLDER.get("current")
    if pc is None:
        # Placeholder — instantiation will fail downstream with a clear
        # error if this is reached without first priming the holder.
        from olmo.eval.configure_molmo_spaces import (
            PretrainedToOursHybridPolicyConfig as _PC,
        )
        return _PC(gripper_representation_count=1, clamp_gripper=False)
    return pc


try:
    from pydantic import Field as _Field
    from olmo.eval.configure_molmo_spaces import (
        PretrainedToOursHybridEvalConfig as _HybridBase,
        PretrainedToOursHybridPolicyConfig as _HybridPC,
    )

    class MonitorHybridEvalConfig(_HybridBase):  # type: ignore[misc, valid-type]
        policy_config: _HybridPC = _Field(default_factory=_factory_hybrid_pc)
except ImportError:
    MonitorHybridEvalConfig = None  # type: ignore[assignment]


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _load_summary(path: Path) -> Set[str]:
    done: Set[str] = set()
    if not path.exists():
        return done
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if "step" in entry:
                    done.add(str(entry["step"]))
            except json.JSONDecodeError:
                continue
    return done


def _is_ckpt_ready(ckpt_dir: Path, settle_s: int) -> bool:
    """Heuristic: ``config.yaml`` is written last + nothing has changed
    in the dir for ``settle_s`` seconds → ckpt is safe to read."""
    cfg = ckpt_dir / "config.yaml"
    if not cfg.exists():
        return False
    latest_mtime = 0.0
    for p in ckpt_dir.rglob("*"):
        if p.is_file():
            try:
                latest_mtime = max(latest_mtime, p.stat().st_mtime)
            except OSError:
                continue
    return (time.time() - latest_mtime) >= settle_s


def _unshard_if_needed(ckpt_dir: Path, unshard_script: Path) -> Path:
    """If ``ckpt_dir`` is still in distcp shards, unshard into a sibling
    ``<step>-unsharded/`` dir and return that path. Idempotent."""
    mp = ckpt_dir / "model.pt"
    if mp.exists() and (ckpt_dir / "config.yaml").exists():
        return ckpt_dir  # already unsharded
    dst = ckpt_dir.parent / f"{ckpt_dir.name}-unsharded"
    if (dst / "model.pt").exists() and (dst / "config.yaml").exists():
        log.info(f"  reusing existing unshard at {dst}")
        return dst
    log.info(f"  unsharding {ckpt_dir} -> {dst}")
    subprocess.run(
        [sys.executable, str(unshard_script), str(ckpt_dir), str(dst)],
        check=True,
        cwd="/tmp",
    )
    return dst


def _run_eval(
    ckpt_unsharded: Path,
    benchmark_dir: Path,
    eval_out: Path,
    max_episodes: Optional[int],
    task_horizon_steps: int,
    wandb_run_name: Optional[str],
    eval_mode: str,
    pretrained_ckpt_path: Optional[str],
    action_type: str,
) -> float:
    """Run a single MolmoSpaces evaluation; return the success rate."""
    from molmo_spaces.evaluation.eval_main import run_evaluation
    eval_out.mkdir(parents=True, exist_ok=True)

    if eval_mode == "standalone":
        from olmo.eval.configure_molmo_spaces import FrankaState8ClampAbsPosConfig
        kwargs = dict(
            eval_config_cls=FrankaState8ClampAbsPosConfig,
            benchmark_dir=benchmark_dir,
            checkpoint_path=str(ckpt_unsharded),
            task_horizon_steps=task_horizon_steps,
            output_dir=str(eval_out),
            num_workers=1,
            use_wandb=bool(wandb_run_name),
        )
        if max_episodes is not None:
            kwargs["max_episodes"] = max_episodes
        return float(run_evaluation(**kwargs).success_rate)

    if eval_mode == "hybrid":
        if not pretrained_ckpt_path:
            raise ValueError(
                "eval_mode='hybrid' requires --pretrained_ckpt_path "
                "(the MolmoBot-DROID checkpoint that bootstraps the grasp)."
            )
        from olmo.eval.configure_molmo_spaces import PretrainedToOursHybridPolicyConfig

        pc = PretrainedToOursHybridPolicyConfig(
            gripper_representation_count=1,
            clamp_gripper=False,
            checkpoint_path=str(ckpt_unsharded),
            pretrained_checkpoint_path=str(pretrained_ckpt_path),
            handoff_ticks_after_grasp=1,
            action_type=action_type,
        )
        pc.action_keys["arm"] = action_type
        _HYBRID_PC_HOLDER["current"] = pc

        kwargs = dict(
            eval_config_cls=MonitorHybridEvalConfig,
            benchmark_dir=benchmark_dir,
            checkpoint_path=str(ckpt_unsharded),
            task_horizon_steps=task_horizon_steps,
            output_dir=str(eval_out),
            num_workers=1,
            use_wandb=bool(wandb_run_name),
        )
        if max_episodes is not None:
            kwargs["max_episodes"] = max_episodes
        return float(run_evaluation(**kwargs).success_rate)

    raise ValueError(f"Unknown eval_mode={eval_mode!r}")


def main() -> int:
    _setup_logging()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--save_folder", type=Path, required=True,
        help="Training run's save_folder; scanned for ``step*/`` subdirs.",
    )
    p.add_argument(
        "--eval_out", type=Path, required=True,
        help="Root dir for per-step eval outputs + summary.json.",
    )
    p.add_argument(
        "--benchmark_dir", type=Path, required=True,
        help="Directory containing ``benchmark.json`` + ``benchmark_metadata.json``.",
    )
    p.add_argument(
        "--unshard_script", type=Path,
        default=Path(__file__).parent / "unshard_pretrained.py",
        help="Path to the unshard helper. Defaults to the sibling script.",
    )
    p.add_argument(
        "--poll_interval_s", type=int, default=120,
        help="Seconds between save_folder scans. Default 120.",
    )
    p.add_argument(
        "--settle_s", type=int, default=60,
        help="Seconds a ckpt dir must be idle before it is accepted.",
    )
    p.add_argument(
        "--max_episodes", type=int, default=None,
        help="Override to run fewer episodes per ckpt.",
    )
    p.add_argument(
        "--task_horizon_steps", type=int, default=600,
        help="Per-episode step budget at policy_dt=66ms (600 steps ≈ 40s).",
    )
    p.add_argument(
        "--wandb_name", type=str, default=None,
        help="wandb run name prefix. If set, use_wandb=True.",
    )
    p.add_argument(
        "--stop_after_step", type=int, default=None,
        help="Exit after this step is evaluated. None = poll forever.",
    )
    p.add_argument(
        "--eval_every_n_steps", type=int, default=None,
        help="Only evaluate checkpoints whose step is a multiple of "
        "this value (e.g. 10000 → step10000, step20000, …).",
    )
    p.add_argument(
        "--eval_mode", type=str, default="hybrid",
        choices=["standalone", "hybrid"],
        help="standalone: our policy drives every step. "
        "hybrid: MolmoBot-DROID bootstraps grasp, then our ckpt takes over.",
    )
    p.add_argument(
        "--pretrained_ckpt_path", type=str, default=None,
        help="MolmoBot-DROID checkpoint path. Required when --eval_mode=hybrid.",
    )
    p.add_argument(
        "--action_type", type=str, default="joint_pos",
        help="Action representation. ``joint_pos`` (absolute) matches our "
        "default training recipe.",
    )
    args = p.parse_args()

    save_folder = args.save_folder.resolve()
    eval_out = args.eval_out.resolve()
    eval_out.mkdir(parents=True, exist_ok=True)
    summary_path = eval_out / "summary.json"
    log.info(f"monitoring {save_folder}")
    log.info(f"eval_out   {eval_out}")
    log.info(f"benchmark  {args.benchmark_dir}")

    done = _load_summary(summary_path)
    if done:
        log.info(f"resuming — {len(done)} steps already evaluated: {sorted(done)}")

    while True:
        candidates = sorted(
            (d for d in save_folder.glob("step*") if d.is_dir()),
            key=lambda p: int(p.name[len("step"):]) if p.name[len("step"):].isdigit() else -1,
        )
        for ckpt_dir in candidates:
            if ckpt_dir.name in done:
                continue
            if not _is_ckpt_ready(ckpt_dir, args.settle_s):
                continue
            if args.eval_every_n_steps:
                try:
                    step_num = int(ckpt_dir.name[len("step"):])
                except ValueError:
                    step_num = -1
                if step_num <= 0 or step_num % args.eval_every_n_steps != 0:
                    done.add(ckpt_dir.name)
                    continue

            log.info(f"=== evaluating {ckpt_dir.name} ===")
            t0 = time.time()
            try:
                unsharded = _unshard_if_needed(ckpt_dir, args.unshard_script)
                step_eval_out = eval_out / ckpt_dir.name
                step_wandb_name = (
                    f"{args.wandb_name}_{ckpt_dir.name}" if args.wandb_name else None
                )
                success_rate = _run_eval(
                    ckpt_unsharded=unsharded,
                    benchmark_dir=args.benchmark_dir,
                    eval_out=step_eval_out,
                    max_episodes=args.max_episodes,
                    task_horizon_steps=args.task_horizon_steps,
                    wandb_run_name=step_wandb_name,
                    eval_mode=args.eval_mode,
                    pretrained_ckpt_path=args.pretrained_ckpt_path,
                    action_type=args.action_type,
                )
            except Exception as e:
                log.exception(f"  eval of {ckpt_dir.name} FAILED: {e}")
                continue
            dur = time.time() - t0
            entry = {
                "step": ckpt_dir.name,
                "success_rate": success_rate,
                "duration_s": dur,
                "max_episodes": args.max_episodes,
            }
            with summary_path.open("a") as f:
                f.write(json.dumps(entry) + "\n")
            log.info(
                f"  {ckpt_dir.name}: success_rate={success_rate:.1%} "
                f"duration={dur/60:.1f}m"
            )
            done.add(ckpt_dir.name)

            if args.stop_after_step is not None:
                cur = int(ckpt_dir.name[len("step"):])
                if cur >= args.stop_after_step:
                    log.info(f"reached stop_after_step={args.stop_after_step}, exiting")
                    return 0

        time.sleep(args.poll_interval_s)


if __name__ == "__main__":
    raise SystemExit(main())
