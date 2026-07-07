"""DAVIS benchmark trajectory dataset — extracted verbatim from the
monolithic `trajectory_3d_dataset.py`.

Eval-only PointMotionBench benchmark: DAVIS clips ship dict-format 3D tracks
(one entry per object, keyed by object name) under `<data_root>/tracks/`,
with captions in `davis_captions.json` and RGB video under
`<data_root>/videos/input_480p/`. No train/test split — every clip is a
test entry, enumerated by walking the tracks directory once and picking the
first object per clip.
"""

import json

import numpy as np

from molmo_motion.data.trajectory.base import BaseTrajectoryDataset
from molmo_motion.data.trajectory.constants import POINTMOTIONBENCH_ROOT


class DavisBenchDataset(BaseTrajectoryDataset):
    """DAVIS PointMotionBench eval-only trajectory dataset (Molmo2 SFT)."""

    TOKEN = "davis_bench"
    DATA_ROOT_ENV = "DAVIS_BENCH_ROOT"
    DATA_ROOT_DEFAULT = f"{POINTMOTIONBENCH_ROOT}/davis"
    SPLIT_FILE = None
    IS_DICT_FORMAT = False
    DEPTH_TOKEN_ELIGIBLE = False
    TIME_STRIDE = 1

    # ── Benchmark entry enumeration (eval-only, no split file) ───────────

    def _build_bench_entries(self):
        root = self.data_root
        captions_path = root / "davis_captions.json"
        if not captions_path.exists():
            raise FileNotFoundError(
                f"DAVIS captions not found at {captions_path}. Download the "
                f"PointMotionBench dataset (ships `davis/davis_captions.json`) "
                f"from allenai/PointMotionBench."
            )
        captions = json.load(open(captions_path))
        entries = []
        for npz_path in sorted((root / "tracks").glob("*_3d.npz")):
            seq = npz_path.stem[:-3]
            # davis_bench 3D is dict {obj_name: (N, T, 3)}; pick the FIRST obj
            # and use it for the entry (full benchmark could iterate all obj
            # but for first-pass eval one-per-clip is sufficient).
            with np.load(npz_path, allow_pickle=True) as f:
                d = f["points_3d"].item()
            obj0 = sorted(d.keys())[0]
            T = int(d[obj0].shape[1])
            entries.append({
                "file": seq,
                "num_frames": T,
                "fps": 30,
                "caption": captions[seq]["description"],
                "clips_by_object": {obj0: [[0, T - 1]]},
                "num_clips_total": 1,
            })
        return entries

    # ── Per-dataset 3D + visibility loading ─────────────────────────────

    def _load_3d_and_vis(self, entry, obj_name):
        path = self.data_root / "tracks" / f"{entry['file']}_3d.npz"
        with np.load(str(path), allow_pickle=True) as f:
            d = f["points_3d"].item()
        pts = d[obj_name].astype(np.float32)              # (N, T, 3)
        vis = np.isfinite(pts).all(axis=-1)                # (N, T)
        return pts, vis

    # ── Per-dataset 2D loading ────────────────────────────────────────────

    def _load_2d_coords(self, entry, obj_name, chosen_indices, t):
        path = self.data_root / "tracks" / f"{entry['file']}_2d.npz"
        with np.load(str(path), allow_pickle=True) as f:
            tracks_raw = f["tracks"]
            dim = f["dim"].astype(int)
        if tracks_raw.dtype == object:
            d = tracks_raw.item()
            tracks = d[obj_name]            # (T, N, 2)
        else:
            tracks = tracks_raw
        return self._sample_2d(tracks, chosen_indices, t, int(dim[0]), int(dim[1]))

    # ── Per-dataset video helpers ───────────────────────────────────────

    def _get_video_path(self, entry):
        return str(self.data_root / "videos" / "input_480p" / f"{entry['file']}.mp4")

    # ── Per-dataset object helpers ──────────────────────────────────────

    def _get_point_count(self, entry, obj_name):
        return 100    # AllTracker samples ~100 query points per object via k-means
