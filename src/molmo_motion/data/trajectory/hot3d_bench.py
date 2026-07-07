"""HOT3D benchmark trajectory dataset — extracted verbatim from the
monolithic `trajectory_3d_dataset.py`.

Eval-only PointMotionBench benchmark: filtered HOT3D-Clips ships exactly
2000 surface points per sub-clip, with tracks under `<data_root>/tracks/`
and RGB under `<data_root>/rgbs/`. No train/test split — every clip is a
test entry (enumerated by walking the directory once).
"""

import json

import numpy as np

from molmo_motion.data.trajectory.base import BaseTrajectoryDataset
from molmo_motion.data.trajectory.constants import POINTMOTIONBENCH_ROOT


class Hot3DBenchDataset(BaseTrajectoryDataset):
    """HOT3D PointMotionBench eval-only trajectory dataset (Molmo2 SFT)."""

    TOKEN = "hot3d_bench"
    DATA_ROOT_ENV = "HOT3D_BENCH_ROOT"
    DATA_ROOT_DEFAULT = f"{POINTMOTIONBENCH_ROOT}/hot3d"
    SPLIT_FILE = None
    IS_DICT_FORMAT = False
    DEPTH_TOKEN_ELIGIBLE = False
    TIME_STRIDE = 1

    # ── Benchmark entry enumeration (eval-only, no split file) ───────────

    def _build_bench_entries(self):
        root = self.data_root
        captions_path = root / "hot3d_annotations.json"
        if not captions_path.exists():
            raise FileNotFoundError(
                f"HOT3D annotations not found at {captions_path}. Download the "
                f"PointMotionBench dataset (ships `hot3d/hot3d_annotations.json`) "
                f"from allenai/PointMotionBench; see pointmotionbench/README.md."
            )
        captions = json.load(open(captions_path))
        entries = []
        for npz_path in sorted((root / "tracks").glob("*_3d.npz")):
            stem = npz_path.stem[:-3]   # strip "_3d"
            # Peek shape without loading full data
            with np.load(npz_path, allow_pickle=True) as f:
                T = f["points_3d"].shape[1]
            cap = captions.get(stem, {}).get("caption", "")
            entries.append({
                "file": stem,
                "num_frames": int(T),
                "fps": 30,
                "caption": cap,
                "clips_by_object": {"obj0": [[0, int(T) - 1]]},
                "num_clips_total": 1,
            })
        return entries

    # ── Per-dataset 3D + visibility loading ─────────────────────────────

    def _load_3d_and_vis(self, entry, obj_name):
        path = self.data_root / "tracks" / f"{entry['file']}_3d.npz"
        with np.load(str(path), allow_pickle=True) as f:
            pts = f["points_3d"].astype(np.float32)       # (2000, T, 3)
            vis = f["visibility"].astype(bool).squeeze(-1)  # (2000, T, 1) → (2000, T)
        return pts, vis

    # ── Per-dataset 2D loading ────────────────────────────────────────────

    def _load_2d_coords(self, entry, obj_name, chosen_indices, t):
        path = self.data_root / "tracks" / f"{entry['file']}_2d.npz"
        with np.load(str(path), allow_pickle=True) as f:
            tracks = f["tracks"]              # (T, 2000, 2)
            dim = f["dim"].astype(int)        # [H, W]
        # tracks: (T, N, 2) — same convention _sample_2d expects.
        return self._sample_2d(tracks, chosen_indices, t, int(dim[0]), int(dim[1]))

    # ── Per-dataset video helpers ───────────────────────────────────────

    def _get_video_path(self, entry):
        return str(self.data_root / "rgbs" / f"{entry['file']}.mp4")

    # ── Per-dataset object helpers ──────────────────────────────────────

    def _get_point_count(self, entry, obj_name):
        return 2000   # filtered HOT3D-Clips ships exactly 2000 surface points per sub-clip
