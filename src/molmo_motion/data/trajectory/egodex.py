"""EgoDex trajectory dataset — extracted verbatim from the monolithic
`trajectory_3d_dataset.py`.

Object-mode EgoDex clips: one object-track cluster per clip, tracks/depth
shipped in `<data_root>/tracks/object/` and `<data_root>/depth/` respectively.
"""

import numpy as np

from molmo_motion.data.trajectory.base import BaseTrajectoryDataset
from molmo_motion.data.trajectory.constants import MOLMO_MOTION_1M_ROOT


class EgoDexDataset(BaseTrajectoryDataset):
    """EgoDex object-mode trajectory dataset (Molmo2 SFT)."""

    TOKEN = "egodex"
    DATA_ROOT_ENV = "MOLMO_MOTION_1M_ROOT"
    DATA_ROOT_DEFAULT = f"{MOLMO_MOTION_1M_ROOT}/egodex"
    SPLIT_FILE = "egodex/annotations/egodex_split.json"
    SPLIT_IS_ABSOLUTE = True
    IS_DICT_FORMAT = True
    DEPTH_TOKEN_ELIGIBLE = True
    TIME_STRIDE = 1

    # ── Per-dataset 3D + visibility loading ─────────────────────────────

    def _load_3d_and_vis(self, entry, obj_name):
        stem = entry["file"]
        path = self.data_root / "tracks" / "object" / f"{stem}_3d.npz"
        with np.load(str(path), allow_pickle=True) as f:
            pts_raw = f["points_3d"]
            vis_raw = f.get("visibility", None) if hasattr(f, "get") else None
            # numpy NpzFile doesn't have .get; use 'in' instead
            if vis_raw is None and "visibility" in f.files:
                vis_raw = f["visibility"]
        if pts_raw.dtype == object:
            return self._unpack_3d_from_dict(pts_raw, vis_raw, obj_name)
        # FLAT: (N, T, 3) — only one object per egodex object-mode clip
        pts_3d = pts_raw.astype(np.float32)
        if vis_raw is not None:
            vis = vis_raw
            if vis.ndim == 3:
                vis = vis.squeeze(-1)
            vis = vis.astype(bool)
        else:
            vis = np.isfinite(pts_3d).all(axis=-1)
        return pts_3d, vis

    # ─── molmo-motion-1m unified 2D loader ─────────────────────────────

    def _load_2d_coords(self, entry, obj_name, chosen_indices, t):
        stem = entry["file"]
        path = self.data_root / "tracks" / "object" / f"{stem}_2d.npz"
        if not path.exists():
            return None
        with np.load(str(path), allow_pickle=True) as f:
            tracks_raw = f["tracks"]
            dim = f["dim"]
        if tracks_raw.dtype == object:
            tdict = tracks_raw.item()
            if obj_name not in tdict:
                return None
            tracks_2d = np.transpose(tdict[obj_name].astype(np.float32), (1, 0, 2))  # (T,N,2)→(N,T,2)
        else:
            # Flat (T, N, 2) → (N, T, 2)
            tracks_2d = np.transpose(tracks_raw.astype(np.float32), (1, 0, 2))
        return self._sample_2d(tracks_2d, chosen_indices, t, int(dim[0]), int(dim[1]))

    # ── Per-dataset camera-pose loading for camera-frame-at-t0 ──────────

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

    # ── Per-dataset depth loading for the depth-token branch ─────────────

    def _load_depth_at_t(self, entry, t):
        # EXR-in-zip: one .exr per frame, named like `{idx:05d}.exr`.
        import os
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

    # ── Per-dataset video helpers ───────────────────────────────────────

    def _get_video_path(self, entry):
        return str(self.data_root / "videos" / f"{entry['file']}.mp4")

    # ── Per-dataset object helpers ──────────────────────────────────────

    def _get_point_count(self, entry, obj_name):
        # molmo-motion-1m egodex has one object cluster per clip; the legacy
        # multi-cluster `_egodex_filter_meta` path is no longer used.
        return 100

    # ── Caption ─────────────────────────────────────────────────────────

    def _get_caption(self, entry):
        caption = entry.get("caption", "")
        if caption and caption.strip():
            return caption.strip()
        # EgoDex fallback: extract task name from stem
        stem = entry.get("file", "")
        # stem like "part1_add_remove_lid_0" → "add remove lid"
        parts = stem.split("_")
        if len(parts) >= 3:
            return " ".join(parts[1:-1])
        return ""
