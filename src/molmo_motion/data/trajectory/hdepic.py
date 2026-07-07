"""HD-EPIC (token "hepic") 3D trajectory dataset.

Extracted verbatim from the monolithic `trajectory_3d_dataset.py`. NOTE:
the internal dataset token is "hepic" but the released corpus stores
HD-EPIC under "hdepic/" with "hdepic_*" annotation filenames.
"""

import os

import numpy as np

from molmo_motion.data.trajectory.base import BaseTrajectoryDataset
from molmo_motion.data.trajectory.constants import MOLMO_MOTION_1M_ROOT


class HdEpicDataset(BaseTrajectoryDataset):
    """HD-EPIC 3D trajectory dataset ("hepic" token)."""

    TOKEN = "hepic"
    DATA_ROOT_ENV = "MOLMO_MOTION_1M_ROOT"
    DATA_ROOT_DEFAULT = f"{MOLMO_MOTION_1M_ROOT}/hdepic"
    SPLIT_FILE = "hdepic/annotations/hdepic_split.json"
    SPLIT_IS_ABSOLUTE = True
    IS_DICT_FORMAT = True
    DEPTH_TOKEN_ELIGIBLE = True
    TIME_STRIDE = 1

    def _load_3d_and_vis(self, entry, obj_name):
        # Multi-object dict with keys 'left_hand', 'right_hand', 'object'.
        stem = entry["file"]
        path = self.data_root / "tracks" / f"{stem}_3d.npz"
        with np.load(str(path), allow_pickle=True) as f:
            pts_raw = f["points_3d"]
            vis_raw = f["visibility"] if "visibility" in f.files else None
        return self._unpack_3d_from_dict(pts_raw, vis_raw, obj_name)

    def _load_2d_coords(self, entry, obj_name, chosen_indices, t):
        # AllTracker pixel-space tracks. The 2D npz is a flat (T, N_total, 2)
        # tensor where N_total is the concatenation of per-object blocks in
        # the dict-iteration order of the matching _3d.npz. There's no
        # explicit per-object slicing key — re-derive the offset from 3D
        # shapes. (Per hepic README: 2D is NOT a reprojection of 3D; they
        # share seeds but are produced by independent pipelines.)
        stem = entry["file"]
        p2 = self.data_root / "tracks" / f"{stem}_2d.npz"
        p3 = self.data_root / "tracks" / f"{stem}_3d.npz"
        if not p2.exists():
            return None
        with np.load(str(p3), allow_pickle=True) as f3:
            pts_dict = f3["points_3d"].item()
        if obj_name not in pts_dict:
            return None
        offset = 0
        for k, v in pts_dict.items():
            if k == obj_name:
                break
            offset += v.shape[0]
        n_obj = pts_dict[obj_name].shape[0]
        with np.load(str(p2), allow_pickle=True) as f2:
            tracks_raw = f2["tracks"].astype(np.float32)  # (T, N_total, 2)
            dim = f2["dim"]
        # Slice this object's block then transpose (T,N,2) → (N,T,2).
        tracks_block = np.transpose(tracks_raw[:, offset:offset + n_obj, :], (1, 0, 2))
        return self._sample_2d(tracks_block, chosen_indices, t, int(dim[0]), int(dim[1]))

    def _load_w2c_for_frame(self, entry, t):
        # vipe pipeline: README labels `data` as w2c but the matrices are
        # actually c2w (verified by reprojection on ytvis — direct project
        # of `data` lands ~12 px off, inverted projection matches the
        # recorded 2D to sub-pixel). So invert here.
        stem = entry["file"]
        path = self.data_root / "camera" / "pose" / f"{stem}.npz"
        if not path.exists():
            return None
        with np.load(str(path), allow_pickle=True) as f:
            c2w = self._w2c_from_pose_dict(f, t)
        R = c2w[:3, :3]
        tvec = c2w[:3, 3]
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = R.T
        w2c[:3, 3]  = -R.T @ tvec
        return w2c

    def _load_depth_at_t(self, entry, t):
        # EXR-in-zip: one .exr per frame, named like `{idx:05d}.exr`.
        import tempfile
        import zipfile

        import Imath
        import OpenEXR
        stem = entry["file"]
        size = self.depth_target_size
        zp = self.data_root / "depth" / f"{stem}.zip"
        if not zp.exists():
            return None
        z = self._get_handle(("zip", str(zp)), lambda: zipfile.ZipFile(str(zp)))
        target_name = f"{int(t):05d}.exr"
        if target_name not in z.namelist():
            return None
        # OpenEXR needs a real path. Write to a tmp once per call.
        with tempfile.NamedTemporaryFile(suffix=".exr", delete=False) as tf:
            tf.write(z.read(target_name))
            tmp = tf.name
        try:
            exr = OpenEXR.InputFile(tmp)
            hdr = exr.header()
            dw = hdr["dataWindow"]
            W = dw.max.x - dw.min.x + 1
            H = dw.max.y - dw.min.y + 1
            ch = "Z" if "Z" in hdr["channels"] else next(iter(hdr["channels"]))
            buf = exr.channel(ch, Imath.PixelType(Imath.PixelType.FLOAT))
            depth = np.frombuffer(buf, dtype=np.float32).reshape(H, W)
        finally:
            os.unlink(tmp)
        return self._resize_depth(depth, size)

    def _get_video_path(self, entry):
        # The seven molmo-motion-1m datasets all use the uniform layout
        # `<root>/videos/<file>.mp4`.
        return str(self.data_root / "videos" / f"{entry['file']}.mp4")

    def _get_point_count(self, entry, obj_name):
        return 100  # Fixed 100 points per object block
