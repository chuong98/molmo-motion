"""Dataset-agnostic base class for 3D trajectory prediction datasets.

`BaseTrajectoryDataset` holds ALL shared logic extracted verbatim from the
monolithic `trajectory_3d_dataset.py`: config-flag handling, single-dataset
entry assembly, clip-based training sampling, deterministic eval-config
expansion, the `_build_example` text/answer builder (frame + B-spline modes),
and the shared video / depth / 2D helpers. Per-dataset behavior lives behind
overridable hooks (see the bottom of this file for the default implementations).

This class is a SINGLE-dataset base — one instance is one dataset. The old
cross-dataset weighted mixing (`_ds_probs`/`_ds_to_indices`) moves to a
future `TrajectoryMixtureDataset`.
"""

import json
import logging
import os
from pathlib import Path

import cv2
import numpy as np

from molmo_motion.data.dataset import Dataset
from molmo_motion.data.trajectory.constants import (
    LABEL_TEXT_2D_COORD,
    LABEL_TEXT_CONTROL_POINTS,
    LABEL_TEXT_ENDPOINT,
    LABEL_TEXT_FUTURE,
    LABEL_TEXT_HISTORY,
    LABEL_TEXT_V1,
    METADATA_ROOT,
    MOLMO_MOTION_1M_ROOT,
)
from molmo_motion.data.video_loader import VideoFrames
from molmo_motion.tokenizer import POINT_FEATURE_TOKEN, TWO_D_FEAT_END_TOKEN, TWO_D_FEAT_START_TOKEN

log = logging.getLogger(__name__)


class BaseTrajectoryDataset(Dataset):
    """Single-dataset base for 3D trajectory prediction (Molmo2 SFT)."""

    # ── Per-dataset class attrs (subclasses override) ────────────────────
    TOKEN = ""
    DATA_ROOT_ENV = ""
    DATA_ROOT_DEFAULT = ""
    SPLIT_FILE = None
    SPLIT_IS_ABSOLUTE = True
    IS_DICT_FORMAT = False
    DEPTH_TOKEN_ELIGIBLE = False
    TIME_STRIDE = 1

    # Class-level fallbacks for newer flags so that an already-pickled
    # dataset instance (e.g. one created before these attrs existed and
    # then unpickled into a worker running newer code) doesn't crash with
    # AttributeError when newer methods inspect `self.<flag>`.
    use_camera_frame = False
    use_depth_token = False
    depth_target_size = 378
    eval_first_h_frames = False

    def __init__(self, split="train", num_points=16, num_future_frames=8,
                 history_size=3,
                 use_2d_point_features=False, max_eval_per_dataset=None,
                 pred_end_point_first=False,
                 use_2d_coordinate=False, predict_history_3d=False,
                 v1_match_format=False, mixed_history=True,
                 use_camera_frame=False, use_depth_token=False,
                 depth_target_size=378,
                 eval_first_h_frames=False,
                 merge_train_test=False,
                 bspline_n_ctrl=0, bspline_reg_lambda=0.0,
                 bspline_reg_order=1, bspline_ctrl_clip=None):
        self.split = split
        self.num_points = num_points
        self.num_future_frames = num_future_frames
        # `history_size` is the default / nominal value used when
        # `mixed_history=False`. When True we ignore it and sample H per
        # example from {1, 3} with 50/50 probability; the value here only
        # affects the eval-mode deterministic configs (we fix H=3 there).
        self.history_size = history_size
        # If True, sample H ∈ {1, 3} uniformly per training example. This
        # is the regime described in the paper (one shared model that has
        # seen both single-frame and 3-frame histories).
        self.mixed_history = mixed_history
        # "train_test" = load train entries but use deterministic test-style eval
        self.is_eval = split in ("validation", "test", "train_test")
        self.use_2d_point_features = use_2d_point_features
        # Endpoint-first answer mode: prepend a single-frame <tracks>... endpoint block
        # (last future frame, labelled "object endpoints") before the full trajectory.
        self.pred_end_point_first = pred_end_point_first
        # If True, prepend a "2d point coordinates: <points coords="(H-1).0 1 x y 2 x y .../>"
        # field to the prompt, using normalized [0,1] 2D coords × 1000 rounded to ints.
        self.use_2d_coordinate = use_2d_coordinate
        # If True, drop the "history 3d point coordinates" field from the prompt and
        # emit it as part of the model answer (as a <tracks>...</tracks> block with
        # label "3d object history"). The answer is then supervised on history too.
        self.predict_history_3d = predict_history_3d
        # If True, emit the *exact* v1-era prompt + answer format used by
        # traj3d_droid_v1 (label "object points", hardcoded F in prompt text,
        # bare caption, "and {H} history frames:\n<tracks>…" connector, no
        # quotes, no Oxford comma, no trailing period). Overrides and ignores
        # the other prompt-augmenting flags (_pointfeat, _2dcoord, _endfirst,
        # _predhist) so the prompt is byte-identical to v1.
        self.v1_match_format = v1_match_format
        # If True, transform every 3D point in `traj_clean` into the
        # camera-of-frame-`t` coordinate frame before the existing
        # delta-from-anchor step. This makes the model's coordinate axes
        # track the camera at the query time t (paper t_0 convention).
        self.use_camera_frame = use_camera_frame
        # If True, load a (H, W) metric-depth map at the query frame t and
        # include it in the example metadata as `depth_t`. The model side
        # (Molmo2TrajectoryDepth) then runs it through the LingBot-Depth
        # encoder and injects the projected features into the SigLIP2 image
        # feature map at the query-frame patches.
        # Stochastic schedule per example: 30% monocular (depth zeroed out),
        # 30% patch-masked (per-pixel Bernoulli p=0.5), 40% full depth.
        # stereo4d is deterministically pinned to the monocular bucket
        # because no depth is shipped for stereo4d in molmo-motion-1m.
        self.use_depth_token = use_depth_token
        self.depth_target_size = depth_target_size
        # Eval-only override: emit exactly ONE config per entry with the
        # query frame fixed at t = H − 1 (history = clip's first H frames)
        # and future = [H, …, H + num_future_frames − 1] (clipped to clip
        # length). Activated by the `_t0first` token in the dataset name.
        self.eval_first_h_frames = eval_first_h_frames
        # `merge_train_test` is accepted for backwards compatibility with
        # the source repo's call sites; the release always uses every
        # annotated clip in `split == "train"` mode regardless of this flag,
        # so it has no effect here.
        self.merge_train_test = merge_train_test
        # B-spline control-point answer mode (opt-in, see
        # docs/bspline_control_points_plan.md). When `bspline_n_ctrl > 0`, the
        # future trajectory is emitted as D cubic control points per point
        # (fit online per example) instead of F frame rows; the reg/clip knobs
        # tune the least-squares fit. 0 disables (default frame-based format).
        self.bspline_n_ctrl = int(bspline_n_ctrl)
        self.bspline_reg_lambda = float(bspline_reg_lambda)
        self.bspline_reg_order = int(bspline_reg_order)
        self.bspline_ctrl_clip = bspline_ctrl_clip
        if self.bspline_n_ctrl > 0 and self.bspline_n_ctrl not in (4, 7, 10):
            raise ValueError(
                f"bspline_n_ctrl must be in {{4, 7, 10}} (or 0 to disable); "
                f"got {self.bspline_n_ctrl}")
        # Per-instance LRU cache of opened zip/h5/hdf5 file handles. Datasets
        # that store depth in archives (egodex/ytvis/hepic zips,
        # xperience hdf5, droid h5) benefit a lot from keeping handles open
        # across consecutive __getitem__ calls inside the same worker.
        self._depth_handle_cache = {}

        # Resolve THIS dataset's data root from the class attrs.
        self.token = self.TOKEN
        if self.DATA_ROOT_ENV == "MOLMO_MOTION_1M_ROOT":
            # DATA_ROOT_DEFAULT already folds in the env-resolved corpus
            # root with the per-dataset subdirectory appended; consulting
            # the env var again would drop the subdirectory.
            self.data_root = Path(self.DATA_ROOT_DEFAULT)
        else:
            self.data_root = Path(
                os.environ.get(self.DATA_ROOT_ENV, self.DATA_ROOT_DEFAULT))

        # Load train/test splits.
        # Release behavior: train mode always uses ALL annotated data
        # (train + test concatenated) — the held-out split metadata still
        # exists in the JSONs but is not exposed as a separate mode in the
        # public API. Evaluation in the release path is done against
        # PointMotionBench, not a held-out slice of the human corpus.
        # Benchmark datasets (SPLIT_FILE is None) enumerate sub-clips from disk.
        if self.SPLIT_FILE is None:
            self.entries = self._build_bench_entries()
        else:
            # Some configs point under MOLMO_MOTION_1M_ROOT, others under
            # the legacy METADATA_ROOT (davis, hotworld).
            if self.SPLIT_IS_ABSOLUTE:
                split_path = os.path.join(MOLMO_MOTION_1M_ROOT, self.SPLIT_FILE)
            else:
                split_path = os.path.join(METADATA_ROOT, self.SPLIT_FILE)
            with open(split_path) as f:
                split_data = json.load(f)
            if self.is_eval:
                self.entries = split_data["test"]
            else:
                # Train mode: use every annotated clip in the split JSON.
                self.entries = list(split_data["train"]) + list(split_data["test"])

        for entry in self.entries:
            entry["_dataset"] = self.TOKEN

        self.entries.sort(key=lambda e: (e["_dataset"], e.get("file", "")))

        # Filter entries against the per-NPZ track-keys cache. Some upstream
        # split JSONs reference obj keys (e.g. `obj1`) that the corresponding
        # `_3d.npz` doesn't actually expose (upstream re-clustered tracks but
        # didn't update the split). Build the cache with
        # `python scripts/build_track_keys_cache.py` if it's missing.
        self.entries = self._filter_entries_by_npz_keys(self.entries)

        # Molmospaces: compress num_frames + clip ranges to strided units so the
        # rest of the pipeline (clip sampling, eval expansion, video seeking)
        # treats frame index k as original raw frame k * STRIDE.
        s = self.TIME_STRIDE
        if s > 1:
            for entry in self.entries:
                entry["num_frames"] = (entry["num_frames"] + s - 1) // s
                new_clips = {}
                for obj, ranges in entry["clips_by_object"].items():
                    kept = [[st // s, en // s] for st, en in ranges if en // s >= st // s]
                    if kept:
                        new_clips[obj] = kept
                entry["clips_by_object"] = new_clips

        # NOTE: short motion clips are NOT hard-dropped here. The training
        # sampler / eval-config builder pick the longest extended clip and
        # `_build_example` pads short futures by repeating the last value;
        # `valid_mask` keeps loss masked on the padded positions. We previously
        # hard-dropped clips with ext_len < H+F, but that disproportionately
        # culled molmospaces under long-horizon (F=30/F=32) configs, so we
        # reverted to the soft pad+mask policy.

        if self.is_eval:
            # Expand entries into all (entry, obj, clip, t) eval configs
            # Per spec: all objects, all clips, 3 t values per clip
            all_configs = self._build_eval_configs()
            if max_eval_per_dataset is not None:
                all_configs = self._stratified_subsample(all_configs, max_eval_per_dataset)
            self.eval_configs = all_configs
        else:
            self.eval_configs = None

        # Print stats
        ds_counts = {}
        for e in self.entries:
            ds_counts[e["_dataset"]] = ds_counts.get(e["_dataset"], 0) + 1
        stats = ", ".join(f"{k}={v}" for k, v in sorted(ds_counts.items()))
        n_eval = f" -> {len(self.eval_configs)} eval configs" if self.eval_configs else ""
        print(f"[Traj3D] {split}: {len(self.entries)} entries ({stats}), "
              f"H={history_size}, F={num_future_frames}, P={num_points}{n_eval}")

    # ── NPZ key filtering (against upstream JSON / NPZ key mismatch) ──

    def _filter_entries_by_npz_keys(self, entries):
        """Drop entries / `clips_by_object` keys not present in the NPZ.

        Loads the per-dataset `track_keys` cache once. For each entry (only
        dict-format datasets), intersects `clips_by_object` keys with the
        NPZ's actual keys; drops entries whose intersection is empty. Entries
        whose NPZ is flat ("__flat__" sentinel) or missing from the cache file
        are passed through unchanged.
        """
        if not self.IS_DICT_FORMAT:
            return entries

        ds = self.TOKEN
        cache_dir = Path(os.environ.get(
            "MOLMO_MOTION_1M_TRACK_KEYS_CACHE",
            str(Path.home() / ".cache" / "molmo_motion_1m_track_keys")))

        cp = cache_dir / f"{ds}.json"
        if not cp.exists():
            raise FileNotFoundError(
                f"track-keys cache missing for {ds}: {cp}. "
                f"Build it with: python scripts/build_track_keys_cache.py "
                f"--datasets {ds}"
            )
        with open(cp) as f:
            key_map = json.load(f)

        kept = []
        n_drop_entry = 0
        n_drop_obj = 0
        n_missing_fid = 0
        for e in entries:
            fid = e["file"]
            available = key_map.get(fid)
            if available is None:
                # Missing/corrupted NPZ — drop entry entirely.
                n_drop_entry += 1
                n_missing_fid += 1
                continue
            if available == ["__flat__"]:
                # Loader ignores obj_name for flat NPZs — keep entry as-is.
                kept.append(e)
                continue
            avail_set = set(available)
            new_clips = {o: r for o, r in e["clips_by_object"].items() if o in avail_set}
            dropped_objs = len(e["clips_by_object"]) - len(new_clips)
            if dropped_objs:
                n_drop_obj += dropped_objs
            if not new_clips:
                n_drop_entry += 1
                continue
            if dropped_objs:
                # Shallow copy so we don't mutate the upstream split JSON dict.
                e = {**e, "clips_by_object": new_clips,
                     "num_clips_total": sum(len(r) for r in new_clips.values())}
            kept.append(e)

        # Log only if there were any drops or corrupt files.
        if n_drop_entry or n_drop_obj:
            msg = (
                f"[Traj3D] {ds}: filter dropped {n_drop_entry} entries"
                f" and pruned {n_drop_obj} obj keys missing in NPZs"
            )
            if n_missing_fid:
                msg += f" ({n_missing_fid} corrupted/missing NPZs)"
            log.info(msg)
        return kept

    # ── molmo-motion-1m unified helpers ───────────────────────────────

    @staticmethod
    def _unpack_3d_from_dict(npz_data, vis_raw, obj_name):
        """Pull (pts_3d (N,T,3), vis (N,T) bool) for one object out of dict-format NPZ data."""
        pts_dict = npz_data.item() if hasattr(npz_data, "item") and npz_data.dtype == object else npz_data
        if obj_name not in pts_dict:
            raise KeyError(f"obj_name='{obj_name}' not in pts dict keys: {list(pts_dict.keys())}")
        pts_3d = pts_dict[obj_name].astype(np.float32)  # (N, T, 3)
        if vis_raw is not None and getattr(vis_raw, "dtype", None) == object:  # noqa: E721
            vis_dict = vis_raw.item()
            vis = vis_dict.get(obj_name, None)
            if vis is None:
                vis = np.isfinite(pts_3d).all(axis=-1)
            else:
                if vis.ndim == 3:
                    vis = vis.squeeze(-1)
                vis = vis.astype(bool)
        else:
            vis = np.isfinite(pts_3d).all(axis=-1)
        return pts_3d, vis

    # ── Per-dataset camera-pose helper for camera-frame-at-t0 ──────────

    @staticmethod
    def _w2c_from_pose_dict(pose, t):
        """Helper: look up w2c[t] from a {data:(T,4,4), inds:(T,)} NPZ-like
        dict, using `inds` to map the requested track frame `t` to the
        corresponding row of `data`. Falls back to direct indexing when
        `inds` is missing or just a 0..T-1 range."""
        data = pose["data"]
        if "inds" in pose:
            inds = pose["inds"]
            # Vast majority of files have `inds == arange(T)`; only do the
            # search when the frame numbering is non-trivial.
            if len(inds) and (inds[0] != 0 or inds[-1] != len(inds) - 1):
                hits = np.where(inds == t)[0]
                if hits.size:
                    return data[hits[0]].astype(np.float32)
        idx = min(max(0, int(t)), data.shape[0] - 1)
        return data[idx].astype(np.float32)

    # ── Depth dropout ─────────────────────────────────────────────────

    @staticmethod
    def _apply_depth_dropout(depth, dataset_name, rng):
        """30 / 30 / 40 dropout schedule per example. Stereo4d clamps to
        monocular regardless of the random draw. Returns (depth, bucket)
        where bucket ∈ {'monocular', 'masked', 'full'}.

        - monocular: depth replaced with all-zeros (no signal)
        - masked: per-pixel Bernoulli with p_keep=0.5
        - full: depth passed through unchanged
        """
        if dataset_name == "stereo4d" or depth is None:
            H, W = (None, None) if depth is None else depth.shape[-2:]
            return None, "monocular"
        r = rng.random()
        if r < 0.30:
            return np.zeros_like(depth), "monocular"
        elif r < 0.60:
            keep_mask = (rng.random(depth.shape) < 0.5).astype(depth.dtype)
            return depth * keep_mask, "masked"
        else:
            return depth, "full"

    # ── Shared depth I/O helpers (used by depth-supporting subclasses) ──

    def _get_handle(self, key, opener):
        """LRU-style: open a handle on first miss and cache for reuse within
        the same dataloader worker."""
        h = self._depth_handle_cache.get(key)
        if h is None:
            h = opener()
            self._depth_handle_cache[key] = h
        return h

    @staticmethod
    def _resize_depth(depth, size):
        """Bilinear-resize a (H, W) float32 depth map to (size, size). Done in
        torch for parity with the model side (and to avoid any cv2 dependency
        at training time)."""
        import torch
        # Copy out of any mmap'd / non-writable buffer first so torch doesn't
        # warn (the npz mmap path returns read-only arrays).
        depth_np = np.ascontiguousarray(np.asarray(depth, dtype=np.float32))
        t = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0)
        t = torch.nn.functional.interpolate(t, size=(size, size), mode="bilinear",
                                             align_corners=False)
        return t.squeeze(0).squeeze(0).numpy()

    # ── 2D track sampling ──────────────────────────────────────────────

    @staticmethod
    def _sample_2d(tracks_NT2, chosen_indices, t, img_h, img_w):
        """Common tail: sample (P, 2) tracks at frame t, normalize to [0,1], clip."""
        coords = tracks_NT2[chosen_indices, t, :].copy()
        coords = np.nan_to_num(coords, nan=0.5 * img_w)
        coords[:, 0] /= img_w
        coords[:, 1] /= img_h
        return np.clip(coords, 0.0, 1.0).astype(np.float32)

    # ── Per-dataset object helpers ──────────────────────────────────────

    def _get_object_names(self, entry):
        return sorted(entry["clips_by_object"].keys())

    # ── Video reading ───────────────────────────────────────────────────

    def _read_video_frames(self, mp4_path, frame_indices):
        """Read specific frames from video as (N, H, W, 3) uint8 RGB.

        Optimized: if frames are consecutive, seek once to first frame
        and read sequentially (2× faster than per-frame seeking).

        Special-case: when `mp4_path` is a numpy array of JPEG bytes (the
        format used by WorldTrack inside HotWorld — RGB frames are stored
        inline in the tracks NPZ, no mp4 file), decode each requested frame
        directly with PIL and skip the OpenCV path.
        """
        if isinstance(mp4_path, np.ndarray):
            import io

            from PIL import Image
            frames = []
            for fi in frame_indices:
                fi_clamped = max(0, min(int(fi), len(mp4_path) - 1))
                img = np.array(Image.open(io.BytesIO(bytes(mp4_path[fi_clamped]))))
                if img.ndim == 2:
                    img = np.stack([img, img, img], axis=-1)
                if img.shape[-1] == 4:
                    img = img[..., :3]
                frames.append(img)
            return np.stack(frames, axis=0).astype(np.uint8)
        if not os.path.exists(mp4_path):
            raise FileNotFoundError(
                f"Video not found: {mp4_path}. The molmo-motion-1m / PointMotionBench "
                f"downloads do not include raw videos for every dataset — run the "
                f"per-dataset reconstruction script first (e.g. "
                f"`<dataset>/reconstruct_videos.py` for molmo-motion-1m, or the "
                f"`pointmotionbench/hot3d/` rgbs rebuild for HOT3D eval). See the "
                f"per-dataset README."
            )
        cap = cv2.VideoCapture(mp4_path)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(
                f"Could not open video (corrupt file or unsupported codec): {mp4_path}"
            )
        frames = []

        # Check if indices are consecutive → use fast sequential read
        sorted_indices = sorted(frame_indices)
        is_consecutive = all(
            sorted_indices[i + 1] - sorted_indices[i] == 1
            for i in range(len(sorted_indices) - 1)
        ) if len(sorted_indices) > 1 else True

        if is_consecutive and len(sorted_indices) > 0:
            # Seek once to first frame, then read consecutively
            cap.set(cv2.CAP_PROP_POS_FRAMES, sorted_indices[0])
            for _ in sorted_indices:
                ret, frame = cap.read()
                if ret:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                elif frames:
                    frames.append(frames[-1].copy())
                else:
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 854
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
                    frames.append(np.zeros((h, w, 3), dtype=np.uint8))
        else:
            # Non-consecutive: seek per frame
            for fi in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ret, frame = cap.read()
                if ret:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                elif frames:
                    frames.append(frames[-1].copy())
                else:
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 854
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
                    frames.append(np.zeros((h, w, 3), dtype=np.uint8))

        cap.release()
        return np.stack(frames, axis=0)  # (N, H, W, 3)

    # ── Eval config expansion ──────────────────────────────────────────

    def _build_eval_configs(self):
        """Expand all entries into deterministic eval configs.

        Per entry: pick ONE (object, clip) — the object with the most points
        (ties broken alphabetically) and its longest extended clip — then emit
        up to three timestamps for that clip: t_min (earliest valid), t_mid
        (midpoint of [t_min, t_max]), t_max (latest valid). If the clip is
        too short to fit H + F, falls back to a single clamped t with
        need_padding=True.

        When `self.eval_first_h_frames` is True (set by the `_t0first` token),
        the eval emits exactly ONE config per entry with the query frame
        pinned at t = H − 1 — i.e. history is the clip's first H frames and
        future is [H, ..., H + F − 1] (clipped to clip length). This is the
        protocol used for the PointMotionBench benchmark evals.
        """
        H = self.history_size
        F = self.num_future_frames
        if self.eval_first_h_frames:
            return self._build_eval_configs_first_h(H, F)
        configs = []
        for entry_idx, entry in enumerate(self.entries):
            num_frames = entry["num_frames"]
            obj_names = sorted(entry["clips_by_object"].keys())
            if not obj_names:
                continue

            # Object with the most points (alphabetical tie-break).
            try:
                pc = [(self._get_point_count(entry, n), n) for n in obj_names]
            except Exception:
                continue
            max_count = max(c for c, _ in pc)
            best_obj = next(n for c, n in pc if c == max_count)

            # Longest extended clip for that object.
            clips = entry["clips_by_object"][best_obj]
            ext_clips = [(max(0, s - 1), min(num_frames - 1, e + 2)) for s, e in clips]
            clip_lens = [ee - es + 1 for es, ee in ext_clips]
            best_clip_idx = clip_lens.index(max(clip_lens))
            ext_start, ext_end = ext_clips[best_clip_idx]
            anchor_s = clips[best_clip_idx][0]

            # Up to 3 timestamps: earliest, middle, latest.
            t_min = ext_start + H - 1
            t_max = ext_end - F
            if t_max < t_min:
                t_clamp = max(ext_start, min(ext_start + H - 1, ext_end))
                t_list = [t_clamp]
                need_padding_list = [True]
            else:
                t_mid = (t_min + t_max) // 2
                # Dedup preserving order min → mid → max.
                seen = set()
                t_list = []
                for t in (t_min, t_mid, t_max):
                    if t not in seen:
                        t_list.append(t)
                        seen.add(t)
                need_padding_list = [False] * len(t_list)

            for t, need_padding in zip(t_list, need_padding_list):
                hist_start = t - H + 1
                hist_frames = list(range(max(ext_start, hist_start), t + 1))
                while len(hist_frames) < H:
                    hist_frames.insert(0, hist_frames[0])
                future_end = min(ext_end, t + F)
                future_frames = list(range(t + 1, future_end + 1))

                configs.append({
                    "entry_idx": entry_idx,
                    "obj_name": best_obj,
                    "clip_idx": best_clip_idx,
                    "anchor_s": anchor_s,
                    "t": t,
                    "hist_frames": hist_frames,
                    "future_frames": future_frames,
                    "need_padding": need_padding,
                })
        return configs

    def _build_eval_configs_first_h(self, H, F):
        """First-H-frames eval mode: one config per entry, t = H − 1,
        future = [H, …, H + F − 1] clipped to clip length.

        For each entry: pick the object with the most points (alphabetical
        tie-break) and the first clip range. Build a single config with
        deterministic frame indices anchored at the clip start.
        """
        configs = []
        for entry_idx, entry in enumerate(self.entries):
            num_frames = entry["num_frames"]
            obj_names = sorted(entry["clips_by_object"].keys())
            if not obj_names:
                continue
            try:
                pc = [(self._get_point_count(entry, n), n) for n in obj_names]
            except Exception:
                continue
            best_obj = max(pc)[1]
            clips = entry["clips_by_object"][best_obj]
            if not clips:
                continue
            anchor_s = clips[0][0]

            # Need at least H frames of history. If the clip is too short
            # even for H, skip it (no useful eval for this entry).
            if num_frames < H:
                continue
            # Query frame t = H − 1 within the clip; the clip uses
            # absolute frame indices into the track tensor, so for benchmark
            # datasets that start at 0 this is exactly frame H-1.
            t = H - 1
            hist_frames = list(range(0, H))
            # Future ends at min(num_frames - 1, t + F).
            future_end = min(num_frames - 1, t + F)
            future_frames = list(range(t + 1, future_end + 1))
            # If the clip is too short to even reach one future frame, skip.
            if not future_frames:
                continue

            configs.append({
                "entry_idx": entry_idx,
                "obj_name": best_obj,
                "clip_idx": 0,
                "anchor_s": anchor_s,
                "t": t,
                "hist_frames": hist_frames,
                "future_frames": future_frames,
                "need_padding": len(future_frames) < F,
            })
        return configs

    def _stratified_subsample(self, configs, max_per_dataset):
        """Deterministic per-video subsampling.

        max_per_dataset is a cap on the number of VIDEOS per dataset. All
        (obj, clip) configs for each selected video are kept. Videos are
        picked in sorted (entry_idx) order so every eval job sees the same
        videos.
        """
        # Group configs by dataset, then by entry_idx (video)
        ds_by_video = {}
        for cfg in configs:
            entry = self.entries[cfg["entry_idx"]]
            ds = entry["_dataset"]
            ds_by_video.setdefault(ds, {}).setdefault(cfg["entry_idx"], []).append(cfg)

        result = []
        for ds in sorted(ds_by_video.keys()):
            by_video = ds_by_video[ds]
            video_indices = sorted(by_video.keys())[:max_per_dataset]

            selected_configs = []
            for vidx in video_indices:
                selected_configs.extend(by_video[vidx])

            result.extend(selected_configs)
            log.info(f"[Traj3D] {ds}: {len(selected_configs)} configs from "
                     f"{len(video_indices)} videos (target ≤{max_per_dataset} videos)")

        return result

    # ── Text formatting ─────────────────────────────────────────────────

    @staticmethod
    def _quantize(val):
        return int(round(val * 1000))

    @staticmethod
    def _format_tracks_text(timestamps, points_delta, visibility, label):
        """Format delta coords into <tracks> text.

        Args:
            timestamps: list of float leading numbers (frame timestamps in the
                default format; control-point indices 0..D-1 in B-spline mode)
            points_delta: (P, F, 3) delta values (or (P, D, 3) control points)
            visibility: (P, F) bool (or (P, D))
            label: trailing label text (e.g. "3d object trajectories" /
                "3d object control points")
        """
        P = points_delta.shape[0]
        frame_strings = []
        for fi, ts in enumerate(timestamps):
            parts = [f"{ts:.1f}"]
            for pi in range(P):
                if not visibility[pi, fi]:
                    continue
                obj_id = pi + 1
                x = BaseTrajectoryDataset._quantize(points_delta[pi, fi, 0])
                y = BaseTrajectoryDataset._quantize(points_delta[pi, fi, 1])
                z = BaseTrajectoryDataset._quantize(points_delta[pi, fi, 2])
                parts.append(f"{obj_id} {x} {y} {z}")
            if len(parts) > 1:
                frame_strings.append(" ".join(parts))
        coords_str = ";".join(frame_strings)
        return f'<tracks coords="{coords_str}">{label}</tracks>'

    # ── Example building ────────────────────────────────────────────────

    def _build_example(self, entry, obj_name, pts_3d, visibility, chosen_indices,
                       hist_frames, future_frames, t, need_padding, anchor_s=None):
        """Build Molmo2-compatible example dict.

        Args:
            entry: metadata dict with _dataset field
            obj_name: chosen object name
            pts_3d: (N_obj, T_total, 3) all points for this object
            visibility: (N_obj, T_total) bool
            chosen_indices: (P,) indices into N_obj
            hist_frames: list of H frame indices (track space)
            future_frames: list of <=F frame indices (track space)
            t: current timestep (last history frame)
            need_padding: whether future frames need padding
            anchor_s: deprecated (still accepted for backwards compatibility,
                but ignored). The 2D point-feature anchor is now the last
                history frame `t`; the anchor is no longer the segment start.
        """
        # Per-example H is determined by the length of `hist_frames` itself
        # (the sampler decides whether to use H=1 or H=3 per example).
        H = len(hist_frames)
        P = self.num_points

        all_frames = hist_frames + future_frames
        n_actual_future = len(future_frames)
        # Effective F: at least num_future_frames (for padding short clips),
        # but expand when _get_train samples a longer future horizon.
        F = max(self.num_future_frames, n_actual_future)

        # Extract trajectory for chosen points at all frames
        traj = pts_3d[chosen_indices][:, all_frames, :]  # (P, H+n_actual, 3)

        # Clean NaN → 0 (non-interpolated tracks like DAVIS, hand datasets may have NaN)
        traj_clean = np.nan_to_num(traj, nan=0.0)

        # Optional: world → camera-frame-at-t (paper t_0 convention).
        # Applied uniformly across all history+future frames so every point's
        # coordinates are expressed in the camera frame of the QUERY time t.
        # Datasets without extrinsics on disk return None from
        # `_load_w2c_for_frame` and fall back to the world-frame delta.
        if self.use_camera_frame:
            w2c = self._load_w2c_for_frame(entry, t)
            if w2c is not None:
                R = w2c[:3, :3]
                tvec = w2c[:3, 3]
                # traj_clean shape (P, T_total, 3) → X_cam = X_world @ R.T + t
                traj_clean = traj_clean @ R.T + tvec

        # Anchor: first point at frame t (last history frame, index H-1)
        anchor_idx = min(H - 1, traj_clean.shape[1] - 1)
        anchor = traj_clean[0, anchor_idx, :].copy()  # (3,)

        # Delta encoding
        traj_delta = traj_clean - anchor[np.newaxis, np.newaxis, :]

        # Pad future if clip too short
        if traj_delta.shape[1] < H + F:
            pad_count = (H + F) - traj_delta.shape[1]
            last_val = traj_delta[:, -1:, :]
            traj_delta = np.concatenate(
                [traj_delta, np.repeat(last_val, pad_count, axis=1)], axis=1)

        # Visibility for chosen points
        vis_chosen = visibility[chosen_indices]  # (P, T_total)

        # History visibility
        hist_vis = np.ones((P, H), dtype=bool)
        for hi, fi in enumerate(hist_frames):
            hist_vis[:, hi] = vis_chosen[:, fi]

        # Future valid mask: visible AND within actual clip bounds (not padded)
        valid_mask = np.zeros((P, F), dtype=bool)
        for fi_idx in range(n_actual_future):
            frame_idx = future_frames[fi_idx]
            valid_mask[:, fi_idx] = vis_chosen[:, frame_idx]
        # Padded future frames remain False

        # Split delta into history and future
        hist_delta = traj_delta[:, :H, :]    # (P, H, 3)
        future_delta = traj_delta[:, H:, :]  # (P, F, 3)

        # Zero out invisible positions in both deltas so text formatting never
        # picks up junk values (raw tracks may carry NaN there; nan_to_num
        # above turned those into 0 relative to the raw anchor, which is not
        # the same as a 0-delta). Sparse emission then skips them.
        hist_delta[~hist_vis] = 0.0
        future_delta[~valid_mask] = 0.0

        # ── Text format ─────────────────────────────────────────────
        hist_timestamps = [float(i) for i in range(H)]
        future_timestamps = [float(H + i) for i in range(F)]

        # v1-era sparse emission: only visible entries (raw tracks carry NaN at
        # invisible positions, so deltas there are junk — skip them in text).
        # `valid_mask` already excludes padded-future positions; `hist_vis`
        # reflects real per-frame visibility within the (always full-length) H
        # history window.
        # When _v1match is on, use the single-label "object points" for BOTH
        # input and output <tracks> blocks (matches traj3d_droid_v1 exactly).
        lbl_hist = LABEL_TEXT_V1 if self.v1_match_format else LABEL_TEXT_HISTORY
        lbl_fut = LABEL_TEXT_V1 if self.v1_match_format else LABEL_TEXT_FUTURE
        input_tracks = self._format_tracks_text(hist_timestamps, hist_delta, hist_vis, lbl_hist)

        # ── B-spline control-point answer (opt-in) ──────────────────
        # Fit D cubic control points to each point's future (in the same
        # shared-anchor delta space), pinning t=0 to the point's own last-
        # history position, and emit them as D rows (index = 0..D-1) instead
        # of F frame rows. History (prompt) stays frame-based. `ctrl_valid_kp`
        # marks points with >= D valid future frames; others are dropped.
        ctrl_np = None
        ctrl_valid_kp = None
        if self.bspline_n_ctrl > 0:
            from molmo_motion.data.bspline import fit_control_points
            anchor_pos = hist_delta[:, H - 1, :]  # (P, 3) each point's t0 (shared-anchor space)
            ctrl_t, valid_kp_t = fit_control_points(
                future_delta, valid=valid_mask, n_ctrl=self.bspline_n_ctrl,
                anchor=anchor_pos, reg_lambda=self.bspline_reg_lambda,
                reg_order=self.bspline_reg_order, clip=self.bspline_ctrl_clip)
            ctrl_np = ctrl_t.numpy()                       # (P, D, 3)
            ctrl_valid_kp = valid_kp_t.numpy()             # (P,)
            ctrl_vis = np.broadcast_to(
                ctrl_valid_kp[:, None], (P, self.bspline_n_ctrl))
            ctrl_indices = [float(i) for i in range(self.bspline_n_ctrl)]
            output_tracks = self._format_tracks_text(
                ctrl_indices, ctrl_np, ctrl_vis, LABEL_TEXT_CONTROL_POINTS)
        else:
            output_tracks = self._format_tracks_text(
                future_timestamps, future_delta, valid_mask, lbl_fut)

        # Endpoint-first mode: emit a single-frame <tracks>...<LABEL_TEXT_ENDPOINT> block
        # covering only the last valid future frame, prepended to the full trajectory
        # answer. Supervised by the normal answer CE loss (appears first in the answer,
        # so gets gradient before the full trajectory tokens).
        endpoint_tracks = ""
        if (not self.v1_match_format) and self.bspline_n_ctrl == 0 \
                and self.pred_end_point_first and n_actual_future > 0:
            last_idx = n_actual_future - 1
            endpoint_tracks = self._format_tracks_text(
                [future_timestamps[last_idx]],
                future_delta[:, last_idx:last_idx + 1, :],
                valid_mask[:, last_idx:last_idx + 1],
                LABEL_TEXT_ENDPOINT)

        caption = self._get_caption(entry)

        # ── 2D point features (optional) ────────────────────────────
        # Two independent flags:
        # 2D anchor frame for both point-feature injection and coord text
        # is now the LAST HISTORY FRAME `t` (= paper's t_0). Both
        # `use_2d_point_features` and `use_2d_coordinate` consume the same
        # per-point coords loaded at frame t.
        coords_2d_t = None
        points_inline = ""
        coords_2d_inline = ""
        if self.use_2d_point_features or self.use_2d_coordinate:
            coords_2d_t = self._load_2d_coords(entry, obj_name, chosen_indices, int(t))

        if self.use_2d_point_features and coords_2d_t is not None:
            # One per-point feature per example, sampled from frame t's ViT
            # patches (no extra anchor frame in the video stack). Format:
            # <points anchor 1 <2d_feat_start><|pf|><2d_feat_end> ... P .../>
            wrapped_pf = f"{TWO_D_FEAT_START_TOKEN}{POINT_FEATURE_TOKEN}{TWO_D_FEAT_END_TOKEN}"
            pts = " ".join(f"{pi + 1} {wrapped_pf}" for pi in range(P))
            points_inline = f"<points anchor {pts}/>"

        if self.use_2d_coordinate and coords_2d_t is not None:
            # One-frame <points coords="..."> with integer [0, 1000] x/y per point
            # at the last history timestamp (H-1).0.
            ts = f"{float(H - 1):.1f}"
            xy = (np.clip(coords_2d_t, 0.0, 1.0) * 1000.0).round().astype(int)
            parts = [ts]
            for pi in range(P):
                parts.append(f"{pi + 1} {int(xy[pi, 0])} {int(xy[pi, 1])}")
            coords_str = " ".join(parts)
            coords_2d_inline = f'<points coords="{coords_str}">{LABEL_TEXT_2D_COORD}</points>'

        if self.v1_match_format:
            # Byte-exact v1 prompt + answer. Ignores all other format flags
            # (_pointfeat / _2dcoord / _endfirst / _predhist) so the result
            # matches traj3d_droid_v1 verbatim: bare caption, hardcoded nominal
            # F timestamps, "and {H} history frames:\n<tracks>" connector,
            # single "object points" label. Answer is the single future block.
            F_nominal = self.num_future_frames
            question = (
                f"Predict the future 3D trajectories of {P} points over {F_nominal} timestamps, "
                f"given action: {caption}, and {H} history frames:\n"
                f"{input_tracks}")
            # output_tracks stays as the single future block (no endpoint / history prepend)
        else:
            # ── Build prompt: "given action", optional 2D fields, and (unless
            #    predict_history_3d) the history-3D field, joined with Oxford-comma "and".
            fields = [f'given action: "{caption}"']
            if coords_2d_inline:
                fields.append(f'2d point coordinates: "{coords_2d_inline}"')
            if points_inline:
                fields.append(f'2d history point features: "{points_inline}"')
            if not self.predict_history_3d:
                fields.append(f'history 3d point coordinates: "{input_tracks}"')
            if len(fields) >= 2:
                fields[-1] = f"and {fields[-1]}"
            if self.bspline_n_ctrl > 0:
                # B-spline mode: ask for D control points over an F-frame horizon.
                question = (
                    f"Predict the {self.bspline_n_ctrl} B-spline control points of "
                    f"{P} points over a {F}-frame horizon, " + ", ".join(fields) + ".")
            else:
                question = (f"Predict the future 3D point coordinates of {P} points over "
                            f"{n_actual_future} timestamps, " + ", ".join(fields) + ".")
            if self.bspline_n_ctrl == 0 and self.pred_end_point_first:
                question += " predict the endpoint 3d coordinate first."

            # ── Build answer: optional endpoint → optional history → full trajectory ──
            answer_blocks = []
            if endpoint_tracks:
                answer_blocks.append(endpoint_tracks)
            # In B-spline mode the answer is the single control-point block only
            # (history stays frame-based in the prompt; mixing formats in the
            # answer would confuse the label-keyed decoder).
            if self.predict_history_3d and self.bspline_n_ctrl == 0:
                answer_blocks.append(input_tracks)
            answer_blocks.append(output_tracks)
            output_tracks = " ".join(answer_blocks)

        # ── Video frames ────────────────────────────────────────────
        # Feed the H history frames `[t-H+1, …, t]` directly. We no longer
        # prepend a separate anchor frame: the 2D-feature anchor *is* the
        # last history frame `t`, which is already the last frame here.
        video_path = self._get_video_path(entry)
        video_frame_indices = self._map_frame_to_video(entry, list(hist_frames))
        frames_rgb = self._read_video_frames(video_path, video_frame_indices)

        n_video_frames = len(hist_frames)
        timestamps_arr = np.arange(n_video_frames, dtype=np.float64) * 1.0
        video_frames = VideoFrames(
            frames=frames_rgb,
            timestamps=timestamps_arr,
            target_fps=1.0,
        )

        # ── Metadata ────────────────────────────────────────────────
        ds = entry["_dataset"]
        file_id = self._example_file_id(entry)

        example_id = f"traj3d_{ds}_{file_id}_{obj_name}_t{t}"

        metadata = {
            "example_id": example_id,
            "task": "3d_trajectory",
            "expression": caption,
            "video": file_id,
            "task_name": ds,
            "obj_label": obj_name,
            "dataset_name": ds,
            "t": int(t),
            "hist_frames": [int(x) for x in hist_frames],
            "future_frames": [int(x) for x in future_frames],
            # GT for metric computation (raw 3D space)
            "gt_answer": output_tracks,
            "gt_anchor": anchor.tolist(),
            # IMPORTANT: gt_future_raw MUST be in the SAME coordinate frame as
            # gt_anchor. `traj_clean` has the optional `use_camera_frame` transform
            # already applied, so it's the right tensor to slice from. Reading
            # from raw `pts_3d` here was a bug that made v3/v4 (camera-frame)
            # eval L2 incomparable to v2 — the metric was computing
            # ||pred_in_cam - gt_in_world|| instead of ||pred - gt|| in one frame.
            "gt_future_raw": (traj_clean[:, H:H + n_actual_future, :].astype(np.float32)
                              if n_actual_future > 0 else np.zeros((P, 0, 3), dtype=np.float32)).tolist(),
            "gt_future_vis": valid_mask.tolist(),
            "pass_idx": None,
            "point_indices": chosen_indices.tolist(),
        }

        # B-spline mode: record D and the render horizon F so the decoder knows
        # how to parse the control-point answer and render it back to F frames.
        # gt_future_raw / gt_future_vis stay frame-based (F frames) for metrics.
        if self.bspline_n_ctrl > 0:
            metadata["bspline_n_ctrl"] = int(self.bspline_n_ctrl)
            metadata["future_horizon"] = int(F)
            if ctrl_valid_kp is not None:
                metadata["bspline_valid_kp"] = ctrl_valid_kp.tolist()

        # Pad gt_future_raw if needed (for padded frames, use last actual value)
        if n_actual_future < F:
            gt_raw = np.array(metadata["gt_future_raw"], dtype=np.float32)  # (P, n_actual, 3)
            if gt_raw.shape[1] > 0:
                pad_count = F - gt_raw.shape[1]
                last_val = gt_raw[:, -1:, :]
                gt_raw = np.concatenate([gt_raw, np.repeat(last_val, pad_count, axis=1)], axis=1)
            else:
                gt_raw = np.zeros((P, F, 3), dtype=np.float32)
            metadata["gt_future_raw"] = gt_raw.tolist()

        # Pass coords_2d through metadata for the model hook to extract.
        # These are the coords at frame t (= the last history frame); the
        # model side reads the corresponding ViT patches from `raw_vit[:, -1]`.
        if self.use_2d_point_features and coords_2d_t is not None:
            metadata["coords_2d"] = coords_2d_t  # (P, 2) float32, normalized [0,1]
            metadata["coords_2d_frame"] = int(t)

        # Depth token branch: load depth at frame t, apply 30/30/40 dropout,
        # ship through metadata as (H, W) float32 (or None if monocular bucket).
        if self.use_depth_token and self.DEPTH_TOKEN_ELIGIBLE:
            raw_depth = self._load_depth_at_t(entry, int(t))
            # Use a per-example RNG seeded by entry idx + t for reproducible
            # dropout across runs (cheap and self-contained).
            d_rng = np.random.RandomState(abs(hash((entry["_dataset"], entry.get("file",""), int(t)))) & 0xffffffff)
            dropped, bucket = self._apply_depth_dropout(raw_depth, entry["_dataset"], d_rng)
            metadata["depth_t"] = dropped     # (H, W) float32 metric meters OR None
            metadata["depth_bucket"] = bucket

        return {
            "video": video_frames,  # VideoFrames object, preprocessor accepts this directly
            "message_list": [{"style": "video_qa", "question": question,
                              "answer": output_tracks}],
            "metadata": metadata,
        }

    # ── Dataset interface ───────────────────────────────────────────────

    @classmethod
    def download(cls, n_procs=1):
        pass

    def __len__(self):
        if self.is_eval:
            return len(self.eval_configs)
        return len(self.entries)

    def get(self, item, rng):
        if self.is_eval:
            return self._get_eval(item)
        return self._get_train(item, rng)

    # ── Training sampling ───────────────────────────────────────────────

    def _get_train(self, item, rng):
        entry = self.entries[rng.randint(0, len(self.entries))]
        # Per-example history length. When `mixed_history` is on, draw H from
        # {1, 3} with 50/50 probability (paper §4 setting). Otherwise use
        # the constructor-fixed `self.history_size`.
        if self.mixed_history:
            H = 1 if rng.rand() < 0.5 else 3
        else:
            H = self.history_size
        F = self.num_future_frames
        P = self.num_points
        num_frames = entry["num_frames"]

        # Step 1: Sample object (weighted by point count)
        obj_names = self._get_object_names(entry)
        if len(obj_names) == 0:
            log.warning(f"No objects for {entry.get('file', '?')}, skipping")
            return self._get_train(rng.randint(0, len(self.entries)), rng)

        try:
            weights = np.array([self._get_point_count(entry, n) for n in obj_names],
                               dtype=np.float64)
        except Exception as e:
            log.warning(f"Failed to get point counts for {entry.get('file', '?')}: {e}")
            return self._get_train(rng.randint(0, len(self.entries)), rng)

        if weights.sum() == 0:
            log.warning(f"All point counts zero for {entry.get('file', '?')}, skipping")
            return self._get_train(rng.randint(0, len(self.entries)), rng)

        weights = weights / weights.sum()
        chosen_obj = obj_names[rng.choice(len(obj_names), p=weights)]

        # Step 2: Sample clip (weighted by extended frame count)
        clips = entry["clips_by_object"][chosen_obj]
        ext_clips = [(max(0, s - 1), min(num_frames - 1, e + 2)) for s, e in clips]
        clip_weights = np.array([ee - es + 1 for es, ee in ext_clips], dtype=np.float64)
        clip_weights = clip_weights / clip_weights.sum()
        clip_idx = rng.choice(len(clips), p=clip_weights)
        ext_start, ext_end = ext_clips[clip_idx]

        # Step 3: Load 3D + visibility first (needed to find valid t)
        try:
            pts_3d, visibility = self._load_3d_and_vis(entry, chosen_obj)
        except Exception as e:
            log.warning(f"Failed to load 3D for {entry.get('file', '?')}: {e}")
            return self._get_train(rng.randint(0, len(self.entries)), rng)

        # Step 4: Sample timestep t, retry within clip if no visible points
        t_min = ext_start + H - 1
        t_max = ext_end - F
        need_padding = t_max < t_min
        if need_padding:
            t = min(ext_start + H - 1, ext_end)
            t = max(ext_start, t)
        else:
            t = rng.randint(t_min, t_max + 1)

        # Clamp t to valid range for this track. Visibility is filtered at
        # frame t (= last history frame = 2D-feature anchor in the new
        # paper-aligned setup).
        t_clamped = min(t, pts_3d.shape[1] - 1)
        visible_idx = np.where(visibility[:, t_clamped])[0]

        # If no visible points at t, try other t values within the clip.
        if len(visible_idx) == 0 and not need_padding:
            all_t = list(range(t_min, t_max + 1))
            rng.shuffle(all_t)
            for alt_t in all_t:
                alt_t_clamped = min(alt_t, pts_3d.shape[1] - 1)
                alt_idx = np.where(visibility[:, alt_t_clamped])[0]
                if len(alt_idx) > 0:
                    t = alt_t
                    t_clamped = alt_t_clamped
                    visible_idx = alt_idx
                    break

        if len(visible_idx) == 0:
            # Truly no visible points at the filter frame — skip to another entry
            return self._get_train(rng.randint(0, len(self.entries)), rng)

        # Step 5: Build frame indices (incremental, not uniform)
        hist_start = t - H + 1
        hist_frames = list(range(max(ext_start, hist_start), t + 1))
        while len(hist_frames) < H:
            hist_frames.insert(0, hist_frames[0])

        future_end = min(ext_end, t + F)
        future_frames = list(range(t + 1, future_end + 1))

        if len(visible_idx) >= P:
            chosen = rng.choice(visible_idx, P, replace=False)
        else:
            extra = rng.choice(visible_idx, P - len(visible_idx), replace=True)
            chosen = np.concatenate([visible_idx, extra])
            rng.shuffle(chosen)

        return self._build_example(entry, chosen_obj, pts_3d, visibility,
                                   chosen, hist_frames, future_frames, t, need_padding)

    # ── Eval (deterministic, all objects × clips × 3 t-values) ────────

    def _get_eval(self, item):
        """Deterministic eval: uses pre-expanded configs from _build_eval_configs."""
        P = self.num_points
        cfg = self.eval_configs[item % len(self.eval_configs)]
        entry = self.entries[cfg["entry_idx"]]
        obj_name = cfg["obj_name"]
        t = cfg["t"]
        hist_frames = cfg["hist_frames"]
        future_frames = cfg["future_frames"]
        need_padding = cfg["need_padding"]

        pts_3d, visibility = self._load_3d_and_vis(entry, obj_name)
        # Visibility filter at frame t (= the 2D-feature anchor).
        filter_idx = min(t, pts_3d.shape[1] - 1)
        visible_idx = np.where(visibility[:, filter_idx])[0]

        if len(visible_idx) == 0:
            visible_idx = np.arange(pts_3d.shape[0])

        # Deterministic uniform spacing (same as modeling/ eval_sampler)
        if len(visible_idx) >= P:
            step = len(visible_idx) / P
            chosen = visible_idx[[int(i * step) for i in range(P)]]
        else:
            reps = P // len(visible_idx) + 1
            chosen = np.tile(visible_idx, reps)[:P]

        return self._build_example(entry, obj_name, pts_3d, visibility,
                                   chosen, hist_frames, future_frames, t, need_padding)

    # ── Hooks (overridden per subclass) ─────────────────────────────────

    def _load_3d_and_vis(self, entry, obj_name):
        """Load (N_obj, T, 3) points and (N_obj, T) bool visibility. Abstract."""
        raise NotImplementedError

    def _get_video_path(self, entry):
        """Return an mp4 path (str) or a numpy JPEG-bytes array. Abstract."""
        raise NotImplementedError

    def _load_2d_coords(self, entry, obj_name, chosen_indices, t):
        """Return normalized [0,1] (P, 2) 2D coords at frame t, or None."""
        return None

    def _load_w2c_for_frame(self, entry, t):
        """Return a (4, 4) world-to-camera matrix at track-space frame `t`,
        or None if the dataset has no extrinsics on disk."""
        return None

    def _load_depth_at_t(self, entry, t):
        """Return a (size, size) float32 metric-depth map at frame t, or None."""
        return None

    def _map_frame_to_video(self, entry, frame_indices):
        """Map track-space frame indices to video-space frame indices."""
        return list(frame_indices)

    def _get_point_count(self, entry, obj_name):
        """Return the (approximate) point count for an object."""
        return 100

    def _get_caption(self, entry):
        """Return the language action caption for this entry."""
        return entry.get("caption", "")

    def _build_bench_entries(self):
        """Enumerate benchmark sub-clip entries (SPLIT_FILE is None)."""
        return []

    def _example_file_id(self, entry):
        """Return the file-id string used in `example_id` and metadata."""
        return entry.get("file", "")
