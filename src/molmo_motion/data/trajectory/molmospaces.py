"""MolmoSpaces trajectory dataset — extracted verbatim from the monolithic
`trajectory_3d_dataset.py` (the `ds == "molmospaces"` branches).
"""

import numpy as np

from molmo_motion.data.trajectory.base import BaseTrajectoryDataset
from molmo_motion.data.trajectory.constants import MOLMO_MOTION_1M_ROOT, MOLMOSPACES_TIME_STRIDE


class MolmoSpacesDataset(BaseTrajectoryDataset):
    """MolmoSpaces: dict-format NPZ tracks, 4x temporal stride, unified mp4 video."""

    TOKEN = "molmospaces"
    DATA_ROOT_ENV = "MOLMO_MOTION_1M_ROOT"
    DATA_ROOT_DEFAULT = f"{MOLMO_MOTION_1M_ROOT}/molmospaces"
    SPLIT_FILE = "molmospaces/annotations/molmospaces_split.json"
    SPLIT_IS_ABSOLUTE = True
    IS_DICT_FORMAT = True
    DEPTH_TOKEN_ELIGIBLE = True
    TIME_STRIDE = MOLMOSPACES_TIME_STRIDE

    def _load_3d_and_vis(self, entry, obj_name):
        # obj_name is 'body_{int}'.
        stem = entry["file"]
        path = self.data_root / "tracks" / f"{stem}_3d.npz"
        s = MOLMOSPACES_TIME_STRIDE
        with np.load(str(path), allow_pickle=True) as f:
            pts_dict = f["points_3d"].item()
            vis_dict = f["visibility"].item()
        if obj_name not in pts_dict:
            raise KeyError(f"obj_name='{obj_name}' not in {list(pts_dict.keys())}")
        pts_3d = pts_dict[obj_name].astype(np.float32)   # (N, T, 3)
        vis    = vis_dict[obj_name].astype(bool)          # (N, T)
        # Apply 4× temporal stride to compress slow-moving sim clips.
        pts_3d = pts_3d[:, ::s, :]
        vis    = vis[:, ::s]
        return pts_3d, vis

    def _load_2d_coords(self, entry, obj_name, chosen_indices, t):
        stem = entry["file"]
        path = self.data_root / "tracks" / f"{stem}_2d.npz"
        s = MOLMOSPACES_TIME_STRIDE
        with np.load(str(path), allow_pickle=True) as f:
            tracks_raw = f["tracks"]
            dim = f["dim"]
        tdict = tracks_raw.item()
        if obj_name not in tdict:
            return None
        # Per-object (T, N, 2) → strided (T', N, 2) → transposed (N, T', 2)
        tr = tdict[obj_name].astype(np.float32)[::s, :, :]
        tracks_2d = np.transpose(tr, (1, 0, 2))
        return self._sample_2d(tracks_2d, chosen_indices, t, int(dim[0]), int(dim[1]))

    def _load_w2c_for_frame(self, entry, t):
        stem = entry["file"]
        path = self.data_root / "camera" / f"{stem}.npz"
        if not path.exists():
            return None
        with np.load(str(path), allow_pickle=True) as f:
            c2w_all = f["cam_poses"]  # (T_raw, 4, 4) at raw fps
        # The track loader strides molmospaces by MOLMOSPACES_TIME_STRIDE;
        # `t` is in stride-frame space, so map back to raw-frame index.
        t_raw = int(t) * MOLMOSPACES_TIME_STRIDE
        idx = min(max(0, t_raw), c2w_all.shape[0] - 1)
        c2w = c2w_all[idx].astype(np.float32)
        R = c2w[:3, :3]
        tvec = c2w[:3, 3]
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = R.T
        w2c[:3, 3]  = -R.T @ tvec
        return w2c

    def _load_depth_at_t(self, entry, t):
        # camera/{slug}.npz `depth_frames` (T_raw, H, W). t is in
        # stride-frame space, so map back to raw.
        stem = entry["file"]
        size = self.depth_target_size
        npz_p = self.data_root / "camera" / f"{stem}.npz"
        if not npz_p.exists():
            return None
        mm = self._get_handle(
            ("npz", str(npz_p)),
            lambda: np.load(str(npz_p), allow_pickle=True, mmap_mode="r"))
        if "depth_frames" not in mm:
            return None
        t_raw = int(t) * MOLMOSPACES_TIME_STRIDE
        arr = mm["depth_frames"]
        if t_raw >= arr.shape[0]:
            return None
        depth = arr[t_raw].astype(np.float32)
        return self._resize_depth(depth, size)

    def _get_video_path(self, entry):
        return str(self.data_root / "videos" / f"{entry['file']}.mp4")

    def _map_frame_to_video(self, entry, frame_indices):
        return [idx * MOLMOSPACES_TIME_STRIDE for idx in frame_indices]

    def _get_point_count(self, entry, obj_name):
        return 100  # Approximate; exact count only matters for multi-object weighting
