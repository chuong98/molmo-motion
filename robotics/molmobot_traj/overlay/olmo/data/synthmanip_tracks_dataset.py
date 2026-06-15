"""Merged SynthManip + Trajectory-3D dataset for Phase-2 MolmoBot training.

Ports the prompt/answer construction from
``MotionPlanner/molmo2/olmo/data/trajectory_3d_dataset.py`` onto MolmoBot's
SynthmanipDataset so the LLM gets CE supervision on a ``<tracks>`` text
answer while the ActionExpert still flow-matches joint_pos actions.

Data source per episode in ``pick_place_{,color_}2cam_randomized/house_*/``:
    - ``trajectories_batch_*.h5``                      (MolmoBot: actions, state, obs_scene JSON)
    - ``episode_<id>_<cam>_batch_*.mp4``               (per-camera video)
    - ``episode_<id>_<cam>_point_tracks.npz``          (per-camera 3D point tracks)

Per-NPZ schema (MolmoSpaces 2cam export, verified empirically):
    trajs_2d             (T, N=100, 2)   float32
    visibility           (T, N)          float32 (0..1, threshold 0.5)
    points_3d_initial    (N, 3)          float32
    points_3d            (T, N, 3)       float32, in the NPZ's camera frame
    body_ids             (N,)            int32, usually a single pickup-object body
    intrinsics           (3, 3)          float32
    cam_poses            (T, 4, 4)       float32  (identity in the exported NPZs)
    num_sampled_from     ()              int32
    depth_frames         (T, H, W)       float32  (2cam only)

We restrict per-step sampling to ``sample_phases`` (default {5,6,7,8}:
lift, preplace, place, retreat) — the window where the pickup object is
either in the gripper or just released onto the receptacle.
"""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from olmo.data.synthmanip_dataset import (
    SynthmanipDataset,
    _open_h5_with_retry,
)
from olmo.data.synthmanip_config import SynthmanipDatasetConfig, DEFAULT_PROMPT_TEMPLATES

log = logging.getLogger(__name__)


MOLMOSPACES_VIS_THRESHOLD = 0.5
_QUANTIZE_SCALE = 1000  # 1 mm per integer, meter-scale inputs

# ── <tracks> label strings ─────────────────────────────────────────────────
# Must match molmo2/olmo/data/trajectory_3d_dataset.py byte-for-byte so the
# init checkpoint's learned response distribution is preserved.
LABEL_TEXT_HISTORY = "3d object history"
LABEL_TEXT_FUTURE = "3d object trajectories"
LABEL_TEXT_V1 = "object points"


def _quantize(val: float) -> int:
    return int(round(val * _QUANTIZE_SCALE))


class SynthmanipTracksDataset(SynthmanipDataset):
    """SynthmanipDataset subclass that additionally emits a ``<tracks>`` answer.

    Behaviour preserved vs. base class:
        - action / state / image / action_is_pad / normalization / weighted sampling
        - h5 trajectory bookkeeping

    Behaviour changed:
        - ``_sample_step_weighted`` restricts to ``config.sample_phases`` first,
          then applies base-class grasp-aware weights within the eligible set.
        - ``get`` reads an NPZ ``points_3d`` + ``visibility`` in parallel and
          replaces ``question`` / ``answers`` with trajectory_3d-style text.
    """

    def __init__(self, config: SynthmanipDatasetConfig):
        if not config.load_3d_tracks:
            raise ValueError("SynthmanipTracksDataset requires load_3d_tracks=True")
        super().__init__(config)
        self.num_points = config.num_points
        self.history_size = config.history_size
        self.num_future_frames = config.num_future_frames
        self.predict_camera = config.predict_camera
        self.sample_phases = tuple(config.sample_phases)
        self.prompt_style = config.prompt_style
        if self.prompt_style not in ("default", "v1_match"):
            raise ValueError(
                f"prompt_style must be 'default' or 'v1_match', got {self.prompt_style!r}"
            )
        self.prompt_encoder_mode = config.prompt_encoder_mode

        # Lazy per-traj caches.
        self._phase_valid_steps: Dict[int, np.ndarray] = {}
        self._npz_cache: Dict[Tuple[int, int, str], Dict[str, np.ndarray]] = {}

    # ── Phase-restricted step sampling ────────────────────────────────────────

    def _phase_eligible_steps(self, global_traj_idx: int) -> np.ndarray:
        """Return the set of step indices whose policy_phase is in sample_phases."""
        if global_traj_idx in self._phase_valid_steps:
            return self._phase_valid_steps[global_traj_idx]
        file_idx, traj_idx = self._get_file_and_traj_idx(global_traj_idx)
        traj_length = self.traj_idx_to_length[global_traj_idx]
        with _open_h5_with_retry(self._files[file_idx]) as f:
            phase = np.array(f[f"traj_{traj_idx}"]["obs/extra/policy_phase"][:traj_length])
        eligible = np.where(np.isin(phase, np.array(self.sample_phases, dtype=phase.dtype)))[0]
        self._phase_valid_steps[global_traj_idx] = eligible
        return eligible

    def _sample_step_weighted(self, global_traj_idx: int, rng: np.random.Generator) -> int:
        # Phase 2: always restrict to sample_phases for both prompt-encoder mode
        # and joint mode. User requirement: both training runs must only see the
        # {5, 6, 7, 8} window (lift / preplace / place / retreat) — matches the
        # eval subset and keeps the two init checkpoints' training data identical.
        eligible = self._phase_eligible_steps(global_traj_idx)
        if eligible.size == 0:
            # Trajectory has zero steps in the allowed phases. Fall back to base
            # sampler so training doesn't crash — the example will be emitted
            # without track supervision (caller guards on empty answers later).
            return super()._sample_step_weighted(global_traj_idx, rng)
        # Compose base weights restricted to eligible steps. Cheap: O(|eligible|).
        self._ensure_weights_cached(global_traj_idx)
        w = self._step_weights[global_traj_idx][eligible]
        w = np.maximum(w, 1e-8)
        cdf = np.cumsum(w)
        cdf /= cdf[-1]
        i = int(np.searchsorted(cdf, rng.random()))
        return int(eligible[min(i, eligible.size - 1)])

    # ── NPZ loader ────────────────────────────────────────────────────────────

    def _npz_path_for(self, file_path: Path, traj_idx: int, camera: Optional[str] = None) -> Path:
        """Map the h5 traj entry to the NPZ path for a given camera.

        Data-gen convention: per-trajectory NPZ sits next to the h5 as
        ``episode_<traj_idx:08d>_<camera>_point_tracks.npz``. When ``camera``
        is None, falls back to ``self.predict_camera``.
        """
        cam = camera if camera is not None else self.predict_camera
        house_dir = file_path.parent
        return house_dir / f"episode_{traj_idx:08d}_{cam}_point_tracks.npz"

    def _load_npz(self, file_idx: int, traj_idx: int, file_path: Path,
                  camera: Optional[str] = None) -> Dict[str, np.ndarray]:
        cam = camera if camera is not None else self.predict_camera
        key = (file_idx, traj_idx, cam)
        if key in self._npz_cache:
            return self._npz_cache[key]
        p = self._npz_path_for(file_path, traj_idx, camera=cam)
        if not p.exists():
            raise FileNotFoundError(
                f"Phase-2 expects a point_tracks.npz next to each h5/traj, but "
                f"{p} was not found. Re-run prepare_training_data.py after enabling "
                f"NPZ symlinking."
            )
        with np.load(p, allow_pickle=False) as z:
            data = {
                "points_3d": z["points_3d"].astype(np.float32),        # (T, N, 3)
                "visibility": z["visibility"].astype(np.float32),      # (T, N)
                "body_ids": z["body_ids"].astype(np.int32),            # (N,)
            }
        # Keep the most recent ~64 episodes cached to bound RAM.
        if len(self._npz_cache) > 64:
            self._npz_cache.pop(next(iter(self._npz_cache)))
        self._npz_cache[key] = data
        return data

    # ── Point + frame sampling ────────────────────────────────────────────────

    def _select_point_indices(
        self,
        visibility: np.ndarray,   # (T_npz, N)
        hist_frames: List[int],
        future_frames: List[int],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Choose P point indices with best visibility across hist+future."""
        N = visibility.shape[1]
        vis_hist = (visibility[hist_frames] >= MOLMOSPACES_VIS_THRESHOLD).all(axis=0)   # (N,)
        vis_future = (visibility[future_frames] >= MOLMOSPACES_VIS_THRESHOLD).mean(axis=0)  # (N,)
        # Prefer all-history-visible points first; break ties by future visibility.
        score = vis_hist.astype(np.float32) * 10.0 + vis_future
        # Randomize order among equal scores.
        perm = rng.permutation(N)
        order = np.argsort(-score[perm], kind="stable")
        indices = perm[order][: self.num_points]
        return indices

    def _build_frames(self, step: int, traj_length: int) -> Tuple[List[int], List[int], int]:
        """Return (history_frames, future_frames, n_actual_future).

        history_frames = [step-(H-1)*HS, ..., step-HS, step]   len = H   (stride HS)
        future_frames  = [step+FS, step+2*FS, ..., step+F*FS]  len = n_actual  (stride FS)

        The stride arguments let the track window span the same physical time
        window as the ActionExpert's action_horizon without bumping the number
        of text-emitted frames. Example: ``history_stride=1, future_stride=2,
        H=3, F=8`` → history is dense (3 frames @ 15fps = 0.2s back) while
        future spans 16 sim steps (≈1.07s) — matching ``action_horizon=16``.
        """
        H = self.history_size
        F = self.num_future_frames
        HS = max(1, int(getattr(self.config, "history_stride", 1)))
        FS = max(1, int(getattr(self.config, "future_stride", 1)))

        hist_start = step - (H - 1) * HS
        hist = [max(0, hist_start + i * HS) for i in range(H)]

        # Future: step + FS, step + 2*FS, ..., step + F*FS.
        # n_actual = count of frames that fit within traj_length.
        all_future = [step + (i + 1) * FS for i in range(F)]
        future = [f for f in all_future if f <= traj_length - 1]
        n_actual = len(future)
        return hist, future, n_actual

    # ── <tracks> text formatting ──────────────────────────────────────────────

    def _format_tracks_text(
        self,
        timestamps: List[float],
        points_delta: np.ndarray,      # (P, F, 3)
        visibility: np.ndarray,        # (P, F) bool
        label: str = "object points",
    ) -> str:
        P = points_delta.shape[0]
        frame_strings = []
        for fi, ts in enumerate(timestamps):
            parts = [f"{ts:.1f}"]
            for pi in range(P):
                if not visibility[pi, fi]:
                    continue
                x = _quantize(points_delta[pi, fi, 0])
                y = _quantize(points_delta[pi, fi, 1])
                z = _quantize(points_delta[pi, fi, 2])
                parts.append(f"{pi + 1} {x} {y} {z}")
            if len(parts) > 1:
                frame_strings.append(" ".join(parts))
        coords_str = ";".join(frame_strings)
        return f'<tracks coords="{coords_str}">{label}</tracks>'

    # ── get() override ────────────────────────────────────────────────────────

    def get(self, idx: int, rng: np.random.Generator) -> Dict[str, Any]:
        """Same core data path as SynthmanipDataset.get, but overrides the final
        example dict's ``question`` and ``answers`` with merged trajectory-3D
        prompt + <tracks> GT text. When ``prompt_encoder_mode=True`` the history
        tracks are stripped and the answer is left empty (MolmoBot action-only
        recipe with a trajectory-3D-shaped prompt prefix)."""
        result = super().get(idx, rng)

        step = result["metadata"]["step"]
        global_traj_idx = result["metadata"]["traj_index"]
        file_idx, traj_idx = self._get_file_and_traj_idx(global_traj_idx)
        file_path = self._files[file_idx]
        traj_length = self.traj_idx_to_length[global_traj_idx]

        if self.prompt_encoder_mode:
            # Bypass NPZ. Rebuild just the trajectory-3D prompt PREFIX (task-
            # description-shaped, same as the init ckpt learned to consume)
            # and emit an empty answer so no CE loss fires.
            caption = self._goal_for_prompt(result)
            # n_actual mirrors the real get() path for the default style so the
            # "{F_actual} timestamps" slot stays consistent with what the LLM
            # saw during its trajectory_3d pretraining.
            _, _, n_actual = self._build_frames(step, traj_length)
            if self.prompt_style == "v1_match":
                question = (
                    f"Predict the future 3D trajectories of {self.num_points} points over "
                    f"{self.num_future_frames} timestamps, given action: {caption},"
                )
            else:
                question = (
                    f"Predict the future 3D point coordinates of {self.num_points} points over "
                    f"{n_actual} timestamps, given action: \"{caption}\","
                )
            result["question"] = question
            result["answers"] = ""
            result["metadata"].update({
                "prompt_style": self.prompt_style,
                "prompt_encoder_mode": True,
                "repo_id": "synthmanip_tracks",
            })
            return result

        # 1. Load NPZ for the exo camera that super().get() actually picked for
        # this example. When predict_camera_follows_exo is set (5-cam random-pick
        # mode), we want the NPZ to match the exo camera the VLM just saw —
        # otherwise the prompt's 3D coords would be in a different camera frame
        # than the input images.
        eff_cams = result["metadata"].get("effective_cameras", [])
        if self.config.predict_camera_follows_exo and eff_cams:
            per_example_cam = eff_cams[0]
        else:
            per_example_cam = self.predict_camera
        npz = self._load_npz(file_idx, traj_idx, file_path, camera=per_example_cam)
        pts_3d = npz["points_3d"]           # (T_npz, N, 3)
        vis = npz["visibility"]             # (T_npz, N)
        T_npz = pts_3d.shape[0]
        # Clamp to min(h5_traj_length, npz_length) — prep already aligns these,
        # but guard defensively in case of stale symlinks.
        if T_npz < traj_length:
            traj_length = T_npz
        # step must fit inside the NPZ
        step = min(step, T_npz - 1)

        # 2. Build history/future windows
        hist_frames, future_frames, n_actual = self._build_frames(step, traj_length)
        if n_actual <= 0:
            # Edge case (phase 8 at the very end of clip): emit empty answer so
            # the LM loss falls back to 0 for this example; still return full
            # MolmoBot side-channel for ActionExpert supervision.
            result["answers"] = ""
            return result

        # 3. Pick P points
        point_indices = self._select_point_indices(vis, hist_frames, future_frames, rng)
        hist_pts = pts_3d[hist_frames][:, point_indices, :]         # (H, P, 3)
        hist_pts = np.transpose(hist_pts, (1, 0, 2))                # (P, H, 3)
        fut_pts = pts_3d[future_frames][:, point_indices, :]        # (F_actual, P, 3)
        fut_pts = np.transpose(fut_pts, (1, 0, 2))                  # (P, F_actual, 3)
        hist_vis = (vis[hist_frames][:, point_indices] >= MOLMOSPACES_VIS_THRESHOLD).T  # (P, H)
        fut_vis = (vis[future_frames][:, point_indices] >= MOLMOSPACES_VIS_THRESHOLD).T  # (P, F_actual)

        # 4. Delta encode relative to first point at last history frame.
        anchor = hist_pts[0, -1].copy()      # (3,)
        hist_delta = hist_pts - anchor       # (P, H, 3)
        fut_delta = fut_pts - anchor         # (P, F_actual, 3)

        # 5. Merged prompt (style depends on the init checkpoint's learned format)
        caption = self._goal_for_prompt(result)
        H = self.history_size
        hist_ts = [float(i) for i in range(H)]
        fut_ts = [float(H + i) for i in range(n_actual)]

        if self.prompt_style == "v1_match":
            # Byte-identical to molmo2/olmo/data/trajectory_3d_dataset.py::v1_match_format.
            # Same label "object points" on both sides; F_nominal (static) in prompt text;
            # bare caption (no quotes); "and {H} history frames:\n<tracks>...".
            input_tracks = self._format_tracks_text(
                hist_ts, hist_delta, hist_vis, label=LABEL_TEXT_V1,
            )
            output_tracks = self._format_tracks_text(
                fut_ts, fut_delta, fut_vis, label=LABEL_TEXT_V1,
            )
            F_nominal = self.num_future_frames
            question = (
                f"Predict the future 3D trajectories of {self.num_points} points over "
                f"{F_nominal} timestamps, given action: {caption}, "
                f"and {H} history frames:\n{input_tracks}"
            )
        else:
            # "default" style: "Predict the future 3D point coordinates ..."
            # with dynamic F = n_actual, Oxford-comma-joined fields, quoted caption,
            # distinct labels (history vs trajectories).
            input_tracks = self._format_tracks_text(
                hist_ts, hist_delta, hist_vis, label=LABEL_TEXT_HISTORY,
            )
            output_tracks = self._format_tracks_text(
                fut_ts, fut_delta, fut_vis, label=LABEL_TEXT_FUTURE,
            )
            question = (
                f"Predict the future 3D point coordinates of {self.num_points} points over "
                f"{n_actual} timestamps, given action: \"{caption}\", "
                f"and history 3d point coordinates: \"{input_tracks}\"."
            )

        result["question"] = question
        result["answers"] = output_tracks
        result["metadata"].update({
            "gt_anchor": anchor.tolist(),
            "gt_future_raw": fut_pts.tolist(),     # world/cam-frame coords
            "gt_future_vis": fut_vis.tolist(),
            "point_indices": point_indices.tolist(),
            "hist_frames": hist_frames,
            "future_frames": future_frames,
            "predict_camera": self.predict_camera,
            "prompt_style": self.prompt_style,
            "repo_id": "synthmanip_tracks",
        })
        return result

    # ── helpers ───────────────────────────────────────────────────────────────

    def _goal_for_prompt(self, result: Dict[str, Any]) -> str:
        """Reuse the base class's goal (already randomized if configured).

        ``super().get`` already wrote the task-description (or randomized
        template) into ``result["question"]``. We unwrap any quotation the
        caller expects so the final prompt double-quotes cleanly.
        """
        caption = result["question"] if result["question"] else ""
        return caption.replace('"', "'").strip()
