"""Xperience trajectory dataset — extracted verbatim from the monolithic
`trajectory_3d_dataset.py`.

Per-object NPZ layout: `<data_root>/tracks/{stem}_{object,left_hand,right_hand}.npz`,
flat (N, T, 3) points + (N, T) visibility, with 2D tracks/pixel_coords shipped
inline in the same NPZ. Camera poses ship as c2w (inverted here to w2c);
depth is stored per-episode in hdf5 files keyed by a frame_indices mapping.
"""

import numpy as np

from molmo_motion.data.trajectory.base import BaseTrajectoryDataset
from molmo_motion.data.trajectory.constants import MOLMO_MOTION_1M_ROOT


class XperienceDataset(BaseTrajectoryDataset):
    """Xperience 3D trajectory dataset ("xperience" token)."""

    TOKEN = "xperience"
    DATA_ROOT_ENV = "MOLMO_MOTION_1M_ROOT"
    DATA_ROOT_DEFAULT = f"{MOLMO_MOTION_1M_ROOT}/xperience"
    SPLIT_FILE = "xperience/annotations/xperience_split.json"
    SPLIT_IS_ABSOLUTE = True
    IS_DICT_FORMAT = False
    DEPTH_TOKEN_ELIGIBLE = True
    TIME_STRIDE = 1

    # ── Per-dataset 3D + visibility loading ─────────────────────────────

    def _load_3d_and_vis(self, entry, obj_name):
        # Per-object NPZ: `{stem}_{object,left_hand,right_hand}.npz`.
        stem = entry["file"]
        track_key = obj_name if obj_name in ("object", "left_hand", "right_hand") else "object"
        path = self.data_root / "tracks" / f"{stem}_{track_key}.npz"
        with np.load(str(path), allow_pickle=True) as f:
            pts_3d = f["points_3d"].astype(np.float32)   # (N, T, 3)
            vis    = f["visibility"].astype(bool)         # (N, T)
        return pts_3d, vis

    # ─── molmo-motion-1m unified 2D loader ─────────────────────────────

    def _load_2d_coords(self, entry, obj_name, chosen_indices, t):
        # 2D coords live INSIDE the 3D NPZ as tracks_2d (object) or pixel_coords (hand).
        stem = entry["file"]
        track_key = obj_name if obj_name in ("object", "left_hand", "right_hand") else "object"
        path = self.data_root / "tracks" / f"{stem}_{track_key}.npz"
        if not path.exists():
            return None
        with np.load(str(path), allow_pickle=True) as f:
            if "tracks_2d" in f.files:
                tracks_2d = f["tracks_2d"].astype(np.float32)        # (N, T, 2)
            elif "pixel_coords" in f.files:
                tracks_2d = f["pixel_coords"].astype(np.float32)     # (N, T, 2)
            else:
                return None
        img_h, img_w = 512, 512
        return self._sample_2d(tracks_2d, chosen_indices, t, img_h, img_w)

    # ── Per-dataset camera-pose loading for camera-frame-at-t0 ──────────

    def _load_w2c_for_frame(self, entry, t):
        stem = entry["file"]
        path = self.data_root / "camera" / "poses" / f"{stem}.npz"
        if not path.exists():
            return None
        with np.load(str(path), allow_pickle=True) as f:
            c2w_all = f["c2w"]
            idx = min(max(0, int(t)), c2w_all.shape[0] - 1)
            c2w = c2w_all[idx].astype(np.float32)
        # invert c2w to get w2c
        R = c2w[:3, :3]
        tvec = c2w[:3, 3]
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = R.T
        w2c[:3, 3]  = -R.T @ tvec
        return w2c

    # ── Per-dataset depth loading for the depth-token branch ─────────────

    def _load_depth_at_t(self, entry, t):
        # depth/{uuid}__{ep}.hdf5 at key "depth/depth": (N, 256, 256) fp32.
        # frame_indices/{slug}.npy gives the (T,) → episode-frame mapping.
        import h5py
        stem = entry["file"]
        size = self.depth_target_size
        # slug = entry["file"] is "<uuid>__ep<int>__clip_<...>"
        stem_parts = stem.split("__")
        if len(stem_parts) < 2:
            return None
        ep_key = "__".join(stem_parts[:2])    # "<uuid>__ep<int>"
        hdf5_p = self.data_root / "depth" / f"{ep_key}.hdf5"
        idx_p  = self.data_root / "camera" / "frame_indices" / f"{stem}.npy"
        if not hdf5_p.exists() or not idx_p.exists():
            return None
        indices = np.load(str(idx_p))
        if int(t) >= len(indices):
            return None
        ep_frame = int(indices[int(t)])
        # `locking=False` is required on weka/NFS — default locking
        # raises OSError(errno=37, 'No locks available').
        h5 = self._get_handle(("h5", str(hdf5_p)),
                              lambda: h5py.File(str(hdf5_p), "r", locking=False))
        depth = h5["depth/depth"][ep_frame].astype(np.float32)
        return self._resize_depth(depth, size)

    # ── Per-dataset video helpers ───────────────────────────────────────

    def _get_video_path(self, entry):
        # The seven molmo-motion-1m datasets all use the uniform layout
        # `<root>/videos/<file>.mp4`.
        return str(self.data_root / "videos" / f"{entry['file']}.mp4")

    # ── Per-dataset object helpers ──────────────────────────────────────

    def _get_point_count(self, entry, obj_name):
        return 100  # Single object, no weighting needed
