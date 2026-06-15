"""Prepare MolmoSpaces pick-and-place data for MolmoBot training.

Input: the raw output of MolmoSpaces' data-generation pipeline, one task
per directory. Each task dir contains ``house_*/`` subdirectories with:

  trajectories_batch_*_of_*.h5        — per-house trajectories (h5)
  episode_<N>_<camera>_batch_*.mp4    — one mp4 per (episode, camera)

Output: a frozen training view under ``--dst_root/<task>/{train, val}/``
with one ``house_*/`` per kept house. We:

1. Deterministic 95/5 train/val split by house (--val_frac, --seed).
2. Symlink every mp4 into the new location (no copy).
3. Copy each ``.h5`` and inject ``traj_<i>/obs/sensor_data/<camera>``
   datasets holding the mp4 filename in UTF-8 bytes so the MolmoBot data
   loader can resolve videos.
4. Cross-check the mp4 frame count against ``traj_<i>/success``'s length;
   clamp the recorded trajectory length so training never reads past the
   end of a video.
5. Write ``valid_trajectory_index.json`` at each split root, mapping
   ``{house_name: {h5_name: {traj_i: valid_length}}}``.

Output layout (for the default 2-camera task):

    <DST_ROOT>/
    └── pick_place_2cam_randomized/
        ├── train/
        │   ├── valid_trajectory_index.json
        │   └── house_0/
        │       ├── trajectories_batch_1_of_1.h5
        │       ├── episode_00000000_exo_camera_1_batch_1_of_1.mp4   (symlink)
        │       └── episode_00000000_wrist_camera_batch_1_of_1.mp4   (symlink)
        └── val/   (same shape)

Usage:

    python scripts/prepare_training_data.py \\
        --src_root /path/to/raw_data \\
        --dst_root /path/to/train_view \\
        --task_dirs pick_place_2cam_randomized pick_place_color_2cam_randomized \\
        --val_frac 0.05 --seed 42

Runtime: ~90 s for ~260 houses. Output footprint: ~1.6 GB copied h5s +
~5800 mp4 symlinks per task.

Dependencies: ``h5py``, ``decord``, ``numpy``.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import shutil
from pathlib import Path

import h5py
import numpy as np
from decord import VideoReader, cpu as decord_cpu


log = logging.getLogger(__name__)

# Default 2-camera setup. MolmoBot's `franka_droid_exo_then_wrist` preset
# expects these two names in this order.
DEFAULT_CAMERAS = ["exo_camera_1", "wrist_camera"]

MP4_RE = re.compile(r"^episode_(\d{8})_(.+)_batch_(\d+)_of_(\d+)\.mp4$")

# Optional: per-task camera lists. Add an entry here if your task uses a
# different camera setup than the default 2-camera. The keys must match
# the task directory names under ``--src_root``.
SCENARIO_CAMERAS: dict[str, list[str]] = {
    "pick_place_2cam_randomized": ["exo_camera_1", "wrist_camera"],
    "pick_place_color_2cam_randomized": ["exo_camera_1", "wrist_camera"],
}


# ── Flat HF-release layout adapter ──────────────────────────────────────────
# The molmo-motion-1m release ships robot trajectories flat:
#   <release>/robot_trajectories/{scenario}__{house}.h5
#   <release>/videos/{scenario}__{house}__{episode}__{cam}.mp4
# `--release_root` rebuilds the nested <scenario>/house_*/ layout (via symlinks)
# that the rest of this script expects, so it works straight off an unpacked
# HF download without re-downloading the raw data-gen tree.
RELEASE_H5_RE = re.compile(r"^(?P<scenario>.+)__(?P<house>house_\d+)\.h5$")
RELEASE_MP4_RE = re.compile(
    r"^(?P<scenario>.+?)__(?P<house>house_\d+)__(?P<ep>\d{8})__(?P<cam>.+)\.mp4$")


def materialize_from_release(
    release_root: Path, work_dir: Path, task_dirs: list[str]
) -> Path:
    """Build a nested <scenario>/house_*/ symlink tree from the flat release."""
    release_root = Path(release_root)
    h5_dir = release_root / "robot_trajectories"
    vid_dir = release_root / "videos"
    if not h5_dir.is_dir():
        raise FileNotFoundError(
            f"{h5_dir} not found — point --release_root at an unpacked "
            f"molmospaces/ release dir (with robot_trajectories/ + videos/)."
        )
    if not vid_dir.is_dir():
        raise FileNotFoundError(f"{vid_dir} not found under {release_root}.")
    keep = set(task_dirs) if task_dirs else None

    n_h5 = 0
    for h5 in sorted(h5_dir.glob("*.h5")):
        m = RELEASE_H5_RE.match(h5.name)
        if not m:
            continue
        scen, house = m.group("scenario"), m.group("house")
        if keep is not None and scen not in keep:
            continue
        hd = work_dir / scen / house
        hd.mkdir(parents=True, exist_ok=True)
        link = hd / "trajectories_batch_1_of_1.h5"
        if not link.is_symlink() and not link.exists():
            link.symlink_to(h5.resolve())
        n_h5 += 1

    n_mp4 = 0
    for mp4 in sorted(vid_dir.glob("*.mp4")):
        m = RELEASE_MP4_RE.match(mp4.name)
        if not m:
            continue
        scen, house = m.group("scenario"), m.group("house")
        if keep is not None and scen not in keep:
            continue
        hd = work_dir / scen / house
        if not hd.is_dir():  # only houses that have a shipped h5
            continue
        link = hd / f"episode_{m.group('ep')}_{m.group('cam')}_batch_1_of_1.mp4"
        if not link.is_symlink() and not link.exists():
            link.symlink_to(mp4.resolve())
        n_mp4 += 1

    log.info(
        f"[release] materialized {n_h5} h5 + {n_mp4} mp4 symlinks under {work_dir}"
    )
    return work_dir


def discover_houses(src_dir: Path) -> list[Path]:
    houses = sorted(
        p for p in src_dir.iterdir() if p.is_dir() and p.name.startswith("house_")
    )
    return [h for h in houses if any(h.glob("trajectories_batch_*.h5"))]


def split_houses(
    houses: list[Path], val_frac: float, seed: int
) -> tuple[list[Path], list[Path]]:
    rng = random.Random(seed)
    shuffled = list(houses)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_frac))
    val = sorted(shuffled[:n_val], key=lambda p: p.name)
    train = sorted(shuffled[n_val:], key=lambda p: p.name)
    return train, val


def build_mp4_lookup(house_src: Path) -> dict[tuple[int, str], str]:
    """Map ``(episode_idx, camera_name) -> mp4_filename`` for one house."""
    lookup: dict[tuple[int, str], str] = {}
    for mp4 in house_src.glob("*.mp4"):
        m = MP4_RE.match(mp4.name)
        if not m:
            continue
        lookup[(int(m.group(1)), m.group(2))] = mp4.name
    return lookup


def _mp4_frame_count(mp4_path: Path) -> int:
    vr = VideoReader(str(mp4_path), ctx=decord_cpu(0))
    try:
        return len(vr)
    finally:
        del vr


def inject_sensor_data_and_index(
    h5_dst: Path,
    mp4_lookup: dict[tuple[int, str], str],
    house_dir: Path,
    cameras: list[str],
) -> dict[str, int]:
    """Write mp4 filenames into ``traj_<i>/obs/sensor_data/<camera>``.

    Returns ``{traj_key: valid_length}`` where
    ``valid_length = min(success_length, min_over_cameras(mp4_frames))``
    so we never read past the end of any referenced video.
    """
    index: dict[str, int] = {}
    with h5py.File(h5_dst, "a") as f:
        traj_keys = sorted(
            (k for k in f.keys() if k.startswith("traj_")),
            key=lambda k: int(k.split("_")[1]),
        )
        for tk in traj_keys:
            traj_idx = int(tk.split("_")[1])
            traj = f[tk]
            if "success" not in traj:
                log.warning(f"{h5_dst}:{tk} has no 'success' dataset; skipping")
                continue
            T = int(traj["success"].shape[0])
            sd = traj["obs"].require_group("sensor_data")
            mp4_frames: list[int] = []
            for cam in cameras:
                mp4 = mp4_lookup.get((traj_idx, cam))
                if mp4 is None:
                    raise FileNotFoundError(
                        f"No mp4 found for {h5_dst}:{tk} camera={cam} "
                        f"(traj_idx={traj_idx})"
                    )
                buf = np.frombuffer(mp4.encode("utf-8"), dtype=np.uint8)
                if cam in sd:
                    del sd[cam]
                sd.create_dataset(cam, data=buf)
                mp4_frames.append(_mp4_frame_count(house_dir / mp4))

            valid_length = min(T, min(mp4_frames)) if mp4_frames else T
            if valid_length < T:
                log.info(
                    f"  {h5_dst.parent.name}/{tk}: clamped T={T} -> "
                    f"{valid_length} (min mp4 frames={min(mp4_frames)})"
                )
            index[tk] = valid_length
    return index


def prep_split(
    src_task_dir: Path,
    dst_split_dir: Path,
    houses: list[Path],
    cameras: list[str],
    dry_run: bool,
    skip_existing_h5: bool = True,
) -> tuple[int, int]:
    index_root: dict[str, dict[str, dict[str, int]]] = {}
    n_traj_total = 0
    dst_split_dir.mkdir(parents=True, exist_ok=True)

    for i, h_src in enumerate(houses):
        h_dst = dst_split_dir / h_src.name
        h_dst.mkdir(exist_ok=True)

        # 1. symlink every mp4
        for mp4 in h_src.glob("*.mp4"):
            link = h_dst / mp4.name
            if link.is_symlink() or link.exists():
                link.unlink()
            if not dry_run:
                os.symlink(mp4.resolve(), link)

        # 2. copy each h5 and inject sensor_data
        h5_srcs = sorted(h_src.glob("trajectories_batch_*.h5"))
        if not h5_srcs:
            log.warning(f"{h_src} has no h5, skipping")
            continue

        mp4_lookup = build_mp4_lookup(h_src)

        per_house_index: dict[str, dict[str, int]] = {}
        for h5_src in h5_srcs:
            h5_dst = h_dst / h5_src.name
            if h5_dst.exists() and skip_existing_h5:
                # Already prepped; just re-index from disk.
                traj_index: dict[str, int] = {}
                try:
                    with h5py.File(h5_dst, "r") as f:
                        for tk in sorted(
                            (k for k in f.keys() if k.startswith("traj_")),
                            key=lambda k: int(k.split("_")[1]),
                        ):
                            if "success" in f[tk]:
                                traj_index[tk] = int(f[tk]["success"].shape[0])
                except OSError as e:
                    log.warning(f"  {h5_dst} unreadable: {e}; re-copying")
                    h5_dst.unlink()
                    shutil.copy2(h5_src, h5_dst)
                    traj_index = inject_sensor_data_and_index(
                        h5_dst, mp4_lookup, h_dst, cameras
                    )
            elif not dry_run:
                if h5_dst.exists():
                    h5_dst.unlink()
                shutil.copy2(h5_src, h5_dst)
                traj_index = inject_sensor_data_and_index(
                    h5_dst, mp4_lookup, h_dst, cameras
                )
            else:
                with h5py.File(h5_src, "r") as f:
                    traj_index = {
                        tk: int(f[tk]["success"].shape[0])
                        for tk in f.keys()
                        if tk.startswith("traj_") and "success" in f[tk]
                    }

            rel = f"{h_dst.name}/{h5_dst.name}"
            per_house_index[rel] = traj_index
            n_traj_total += len(traj_index)

        index_root[h_dst.name] = per_house_index

        if (i + 1) % 10 == 0 or (i + 1) == len(houses):
            log.info(f"  [{dst_split_dir.name}] {i+1}/{len(houses)} houses prepared")

    index_path = dst_split_dir / "valid_trajectory_index.json"
    if not dry_run:
        with open(index_path, "w") as f:
            json.dump(index_root, f, sort_keys=True, indent=2)
    log.info(
        f"  wrote {index_path} ({len(index_root)} houses, {n_traj_total} trajs)"
    )
    return len(index_root), n_traj_total


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--src_root", type=Path, default=None,
        help="Raw data-gen root (one <task>/house_*/ directory per task).",
    )
    src.add_argument(
        "--release_root", type=Path, default=None,
        help="Unpacked molmo-motion-1m molmospaces/ release dir (flat "
        "robot_trajectories/*.h5 + videos/*.mp4). Rebuilt into the nested "
        "layout via symlinks before preparing.",
    )
    p.add_argument(
        "--dst_root", required=True, type=Path,
        help="Where to write the prepared training view.",
    )
    p.add_argument(
        "--task_dirs", nargs="+",
        default=["pick_place_2cam_randomized", "pick_place_color_2cam_randomized"],
        help="Task subdirectories under --src_root to prepare. "
        "Add entries to SCENARIO_CAMERAS at the top of this script if "
        "your task uses non-default camera names.",
    )
    p.add_argument("--val_frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument(
        "--no_skip_existing_h5", action="store_true",
        help="Default: already-prepped h5 files are left untouched and "
        "only re-indexed. Pass to force-overwrite (slow).",
    )
    args = p.parse_args()

    args.dst_root.mkdir(parents=True, exist_ok=True)

    if args.release_root is not None:
        work = args.dst_root / "_release_raw_view"
        log.info(
            f"release_root = {args.release_root} -> materializing nested "
            f"layout at {work}"
        )
        materialize_from_release(args.release_root, work, args.task_dirs)
        args.src_root = work

    log.info(f"src_root  = {args.src_root}")
    log.info(f"dst_root  = {args.dst_root}")
    log.info(f"task_dirs = {args.task_dirs}")
    log.info(f"val_frac={args.val_frac}, seed={args.seed}, dry_run={args.dry_run}")

    summary: dict[str, dict[str, int]] = {}
    for task in args.task_dirs:
        src_task = args.src_root / task
        dst_task = args.dst_root / task
        if not src_task.is_dir():
            log.warning(f"[{task}] missing under --src_root; skipping")
            continue

        houses = discover_houses(src_task)
        train_houses, val_houses = split_houses(houses, args.val_frac, args.seed)
        cameras = SCENARIO_CAMERAS.get(task, DEFAULT_CAMERAS)
        log.info(
            f"[{task}] {len(houses)} houses -> train={len(train_houses)}, "
            f"val={len(val_houses)}; cameras={cameras}"
        )

        skip = not args.no_skip_existing_h5
        t_n, t_traj = prep_split(
            src_task, dst_task / "train", train_houses, cameras, args.dry_run, skip
        )
        v_n, v_traj = prep_split(
            src_task, dst_task / "val", val_houses, cameras, args.dry_run, skip
        )
        summary[task] = dict(
            train_houses=t_n,
            train_trajs=t_traj,
            val_houses=v_n,
            val_trajs=v_traj,
        )

    log.info("--- SUMMARY ---")
    for k, v in summary.items():
        log.info(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
