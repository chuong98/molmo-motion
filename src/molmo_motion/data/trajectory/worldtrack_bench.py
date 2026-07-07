"""WorldTrack benchmark trajectory dataset — extracted verbatim from the
monolithic `trajectory_3d_dataset.py`.

Eval-only PointMotionBench benchmark: sub-clips are enumerated by walking
the `adt_mini` / `ds_mini` / `po_mini` / `pstudio_mini` split directories
under `<data_root>/`, each clip dir holding a single `<clip>.npz` (camera-
frame `tracks_XYZ`, `visibility`, `fx_fy_cx_cy`, inline `images_jpeg_bytes`)
and an optional `caption.json`. No train/test split — every clip is a test
entry.
"""

import json

import numpy as np

from molmo_motion.data.trajectory.base import BaseTrajectoryDataset
from molmo_motion.data.trajectory.constants import POINTMOTIONBENCH_ROOT


class WorldTrackBenchDataset(BaseTrajectoryDataset):
    """WorldTrack PointMotionBench eval-only trajectory dataset (Molmo2 SFT)."""

    TOKEN = "worldtrack_bench"
    DATA_ROOT_ENV = "WT_BENCH_ROOT"
    DATA_ROOT_DEFAULT = f"{POINTMOTIONBENCH_ROOT}/worldtrack"
    SPLIT_FILE = None
    IS_DICT_FORMAT = False
    DEPTH_TOKEN_ELIGIBLE = False
    TIME_STRIDE = 1

    # ── Benchmark entry enumeration (eval-only, no split file) ───────────

    def _build_bench_entries(self):
        root = self.data_root
        entries = []
        for split_name in ("adt_mini", "ds_mini", "po_mini", "pstudio_mini"):
            split_root = root / split_name
            if not split_root.exists():
                continue
            for clip_dir in sorted(split_root.iterdir()):
                if not clip_dir.is_dir():
                    continue
                npz_path = clip_dir / f"{clip_dir.name}.npz"
                if not npz_path.exists():
                    continue
                caption = ""
                cap_path = clip_dir / "caption.json"
                if cap_path.exists():
                    try:
                        caption = json.load(open(cap_path)).get("caption", "")
                    except Exception:
                        pass
                with np.load(npz_path, allow_pickle=True) as f:
                    T = int(f["tracks_XYZ"].shape[0])
                entries.append({
                    "file": f"{split_name}/{clip_dir.name}",
                    "num_frames": T,
                    "fps": 30,
                    "caption": caption,
                    "clips_by_object": {"obj0": [[0, T - 1]]},
                    "num_clips_total": 1,
                })
        return entries

    # ── Per-dataset 3D + visibility loading ─────────────────────────────

    def _load_3d_and_vis(self, entry, obj_name):
        path = self.data_root / entry["file"]
        npz_path = path / f"{path.name}.npz"
        with np.load(str(npz_path), allow_pickle=True) as f:
            tracks_xyz = f["tracks_XYZ"].astype(np.float32)   # (T, N, 3) camera frame
            vis = f["visibility"].astype(bool)                 # (T, N)
        # Convert (T, N, 3) → (N, T, 3) to match our convention.
        pts = np.transpose(tracks_xyz, (1, 0, 2))
        vis = vis.T
        return pts, vis

    # ── Per-dataset 2D loading ────────────────────────────────────────────

    def _load_2d_coords(self, entry, obj_name, chosen_indices, t):
        path = self.data_root / entry["file"]
        npz_path = path / f"{path.name}.npz"
        with np.load(str(npz_path), allow_pickle=True) as f:
            tracks_xyz = f["tracks_XYZ"]     # (T, N, 3) camera frame
            fx, fy, cx, cy = f["fx_fy_cx_cy"].astype(np.float64)
            jpeg = f["images_jpeg_bytes"][0]
        # Decode just the first JPEG to find image size.
        import io

        from PIL import Image
        img = Image.open(io.BytesIO(jpeg))
        W, H = img.size
        # Project per chosen point at frame t.
        ti = min(int(t), tracks_xyz.shape[0] - 1)
        xyz = tracks_xyz[ti, chosen_indices, :]                # (P, 3)
        z = np.clip(xyz[:, 2], 1e-6, None)
        u = xyz[:, 0] / z * fx + cx
        v = xyz[:, 1] / z * fy + cy
        coords = np.stack([u / W, v / H], axis=-1).astype(np.float32)
        return np.clip(coords, 0.0, 1.0)

    # ── Per-dataset video helpers ───────────────────────────────────────

    def _get_video_path(self, entry):
        clip_dir = self.data_root / entry["file"]
        npz_path = clip_dir / f"{clip_dir.name}.npz"
        with np.load(str(npz_path), allow_pickle=True) as f:
            return f["images_jpeg_bytes"]

    # ── Per-dataset object helpers ──────────────────────────────────────

    def _get_point_count(self, entry, obj_name):
        return 100    # per-object after object_id mask (similar to hotworld)
