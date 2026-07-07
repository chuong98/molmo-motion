"""DROID trajectory dataset — extracted verbatim from the monolithic
`trajectory_3d_dataset.py` (the `ds == "droid"` branches).
"""

import numpy as np

from molmo_motion.data.trajectory.base import BaseTrajectoryDataset
from molmo_motion.data.trajectory.constants import MOLMO_MOTION_1M_ROOT


class DroidDataset(BaseTrajectoryDataset):
    """DROID: flat-format NPZ tracks, single object, already in camera frame."""

    TOKEN = "droid"
    DATA_ROOT_ENV = "MOLMO_MOTION_1M_ROOT"
    DATA_ROOT_DEFAULT = f"{MOLMO_MOTION_1M_ROOT}/droid"
    SPLIT_FILE = "droid/annotations/droid_split.json"
    SPLIT_IS_ABSOLUTE = True
    IS_DICT_FORMAT = False
    DEPTH_TOKEN_ELIGIBLE = True
    TIME_STRIDE = 1

    def _load_3d_and_vis(self, entry, obj_name):
        # In molmo-motion-1m, `file` already includes the `__{cam}` suffix.
        stem = entry["file"]
        path = self.data_root / "tracks" / f"{stem}_3d.npz"
        with np.load(str(path), allow_pickle=True) as f:
            pts_3d = f["points_3d"].astype(np.float32)  # (T, N, 3)
            valid_3d = f["valid_3d"]                     # (T, N) bool
        return np.transpose(pts_3d, (1, 0, 2)), valid_3d.T.astype(bool)

    def _load_2d_coords(self, entry, obj_name, chosen_indices, t):
        stem = entry["file"]
        path = self.data_root / "tracks" / f"{stem}_2d.npz"
        with np.load(str(path), allow_pickle=True) as f:
            tracks_2d = np.transpose(f["tracks_2d"].astype(np.float32), (1, 0, 2))  # (T,N,2)→(N,T,2)
            ds_dim = f["ds_dim"]
        return self._sample_2d(tracks_2d, chosen_indices, t, int(ds_dim[0]), int(ds_dim[1]))

    def _load_w2c_for_frame(self, entry, t):
        # `points_3d` is already in the camera frame, so the world frame
        # IS the camera frame: w2c[t] = identity for all t.
        return np.eye(4, dtype=np.float32)

    def _load_depth_at_t(self, entry, t):
        # depth/{ep}.h5 at "{serial}+ext/depth": (T, 360, 640) uint16.
        # Slug format `{episode}__{serial}`.
        import h5py
        stem = entry["file"]
        size = self.depth_target_size
        if "__" not in stem:
            return None
        ep, serial = stem.rsplit("__", 1)
        h5p = self.data_root / "depth" / f"{ep}.h5"
        if not h5p.exists():
            return None
        h5 = self._get_handle(("h5", str(h5p)),
                              lambda: h5py.File(str(h5p), "r", locking=False))
        key = f"{serial}+ext/depth"
        if key not in h5:
            return None
        d = h5[key]
        if int(t) >= d.shape[0]:
            return None
        depth = d[int(t)].astype(np.float32)
        # droid depth is uint16 mm → convert to meters.
        depth = depth / 1000.0
        return self._resize_depth(depth, size)

    def _get_video_path(self, entry):
        # DROID videos already include the `__{cam}` suffix in `file`.
        return str(self.data_root / "videos" / f"{entry['file']}.mp4")

    def _get_point_count(self, entry, obj_name):
        return 100  # DROID: single object, ~60-95 points. Exact count not needed.

    def _example_file_id(self, entry):
        return f"{entry['file']}_{entry['cam']}"
