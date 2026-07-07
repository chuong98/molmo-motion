"""Stereo4D dataset — extracted verbatim from the monolithic
`trajectory_3d_dataset.py`.
"""

import numpy as np

from molmo_motion.data.trajectory.base import BaseTrajectoryDataset
from molmo_motion.data.trajectory.constants import MOLMO_MOTION_1M_ROOT


class Stereo4DDataset(BaseTrajectoryDataset):
    """Stereo4D: dict-format NPZ tracks keyed by 'obj0', 'obj1', ...."""

    TOKEN = "stereo4d"
    DATA_ROOT_ENV = "MOLMO_MOTION_1M_ROOT"
    DATA_ROOT_DEFAULT = f"{MOLMO_MOTION_1M_ROOT}/stereo4d"
    SPLIT_FILE = "stereo4d/annotations/stereo4d_split.json"
    SPLIT_IS_ABSOLUTE = True
    IS_DICT_FORMAT = True
    DEPTH_TOKEN_ELIGIBLE = True
    TIME_STRIDE = 1

    # ── 3D + visibility loading ─────────────────────────────────────────

    def _load_3d_and_vis(self, entry, obj_name):
        # Dict NPZ keyed by 'obj0', 'obj1', ...
        stem = entry["file"]
        path = self.data_root / "tracks" / f"{stem}_3d.npz"
        with np.load(str(path), allow_pickle=True) as f:
            pts_raw = f["points_3d"]
            vis_raw = f["visibility"] if "visibility" in f.files else None
        return self._unpack_3d_from_dict(pts_raw, vis_raw, obj_name)

    # ── 2D coords loading ────────────────────────────────────────────────

    def _load_2d_coords(self, entry, obj_name, chosen_indices, t):
        stem = entry["file"]
        path = self.data_root / "tracks" / f"{stem}_2d.npz"
        if not path.exists():
            return None
        with np.load(str(path), allow_pickle=True) as f:
            tracks_raw = f["tracks"]
            dim = f["dim"]
        tdict = tracks_raw.item() if tracks_raw.dtype == object else None
        if tdict is None or obj_name not in tdict:
            return None
        tracks_2d = np.transpose(tdict[obj_name].astype(np.float32), (1, 0, 2))
        return self._sample_2d(tracks_2d, chosen_indices, t, int(dim[0]), int(dim[1]))

    # ── Camera-pose loading for camera-frame-at-t0 ──────────────────────

    def _load_w2c_for_frame(self, entry, t):
        # camera/{slug}.npz ships `cam_pos` (T, 3) + `cam_fwd` (T, 3).
        # World convention is Y-up (verified by reprojection at ~1.3 px
        # median error with constant intrinsics fx=fy=443.405, cx=cy=256).
        # Construct w2c via look-at: camera-forward = cam_fwd, world-up =
        # +Y, camera-right = up × fwd, camera-down = fwd × right.
        stem = entry["file"]
        path = self.data_root / "camera" / f"{stem}.npz"
        if not path.exists():
            return None
        with np.load(str(path), allow_pickle=True) as f:
            pos = f["cam_pos"]
            fwd = f["cam_fwd"]
        idx = min(max(0, int(t)), pos.shape[0] - 1)
        p = pos[idx].astype(np.float32)
        fd = fwd[idx].astype(np.float32)
        z = fd / (np.linalg.norm(fd) + 1e-9)
        up_world = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        x = np.cross(up_world, z)
        nx = np.linalg.norm(x)
        if nx < 1e-6:
            # cam_fwd is parallel to world Y — degenerate; fall back to no
            # transform rather than emit a garbage R.
            return None
        x = x / nx
        y = np.cross(z, x)
        R = np.stack([x, y, z], axis=0)        # rows are cam axes in world
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = R
        w2c[:3, 3] = -R @ p
        return w2c

    # ── Video helpers ────────────────────────────────────────────────────

    def _get_video_path(self, entry):
        return str(self.data_root / "videos" / f"{entry['file']}.mp4")

    # ── Point count ──────────────────────────────────────────────────────

    def _get_point_count(self, entry, obj_name):
        return 100  # Stereo4D: ~100-300 points per object cluster
