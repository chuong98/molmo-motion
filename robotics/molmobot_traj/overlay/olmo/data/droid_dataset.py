"""DROID training dataset for the MolmoBot prompt-encoder recipe.

Differences from :class:`SynthmanipDataset`:

* Each "trajectory" = one DROID scene, indexed by the JSON output of
  ``scripts/build_droid_index.py``. A scene's trainable timesteps are the
  union of its padded ``[s−3, e+5]`` clip ranges (in 15 fps space). We
  uniformly sample within those ranges.
* Action targets are read from ``scene_path/trajectory.h5``:
    arm     = ``action/joint_position[t : t + action_horizon, :7]``  (radians)
    gripper = ``action/gripper_position[t : t + action_horizon] × 255``
  Past-end action steps are pad-extended (last valid value); ``action_is_pad``
  flags those entries.
* State is read identically from ``observation/robot_state``.
* Camera frames come straight from the source DROID MP4s
  (``scene_path/recordings/MP4/{cam}.mp4``). Frame index maps from 15 fps
  trajectory step to 60 fps source via ``mp4_frame = trajectory_step × 4``.
  Per training step we randomly pick one of ``{ext1, ext2}`` for the
  ``exo_camera_1`` slot; ``wrist_camera_zed_mini`` is always the wrist serial.
* Caption = ``scenes[i]["caption"]`` (the Molmo2-generated short imperative
  from ``_stage2a.npz``).
* No phase filter — the clip cuts already correspond to the lift / preplace /
  place / retreat (phase 5–8 equivalent) of the trajectory.
* No ``<tracks>`` answer span — prompt-encoder mode only. ``answers = ""``.
"""
from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from olmo.data.synthmanip_config import (
    SynthmanipDatasetConfig,
    DEFAULT_PROMPT_TEMPLATES,  # noqa: F401  (kept for parity)
)
from olmo.data.synthmanip_dataset import SynthmanipDataset

log = logging.getLogger(__name__)


@dataclass
class DroidDatasetConfig(SynthmanipDatasetConfig):
    """SynthmanipDatasetConfig + DROID-specific fields.

    ``data_path`` is interpreted as the path to a ``droid_index.json`` produced
    by ``scripts/build_droid_index.py``.
    """

    droid_index_path: str = ""
    gripper_rescale: float = 255.0   # DROID native gripper [0..1] → MolmoSpaces convention [0..255]
    cam_slot_exo: str = "exo_camera_1"
    cam_slot_wrist: str = "wrist_camera_zed_mini"


class DroidDataset(SynthmanipDataset):
    """DROID-backed subclass of SynthmanipDataset.

    Overrides ``__init__``, ``__len__``, ``get`` and the helpers the dataset's
    sampler / loader uses internally (``_get_camera_frames``, ``_get_actions``,
    ``_get_state``, ``_get_goal``, ``_sample_step_weighted``,
    ``_select_effective_cameras``, ``_phase_eligible_steps``).

    Trajectory-shape index:
        self._scenes[i]                 -> dict with "scene_path", "clips", ...
        self._traj_steps_cumsum[i]      -> total cumulative trainable steps
                                            up to and including scene i
        self.traj_idx_to_length[i]      -> the same per-scene total-clip-steps
        self.traj_indices               -> 0..N (one entry per scene)
    """

    def __init__(self, config: DroidDatasetConfig):
        # We deliberately do NOT call SynthmanipDataset.__init__ — its body
        # walks h5 files in MolmoSpaces' layout. Instead we re-implement the
        # subset of its initialisation needed by ``get``.
        from torch.utils.data import Dataset
        Dataset.__init__(self)

        self.config = config
        self.style = "demo"  # mirrors SynthmanipDataset default
        self.camera_names = list(config.camera_names)
        self.action_move_group_names = list(config.action_move_group_names)
        self.action_spec = dict(config.action_spec)
        self.action_keys = dict(config.action_keys)
        self.action_horizon = config.action_horizon
        self.input_window_size = config.input_window_size
        self.state_dim = sum(config.state_spec.values()) if config.state_spec else 8

        self.robot_preprocessor = (
            config.robot_processor_config.build_preprocessor()
            if config.robot_processor_config else None
        )
        # Reuse SynthmanipDataset's decord bridge setup (dataloader workers
        # need this to be set per-process; SynthmanipDataset.__init__ does it
        # globally).
        try:
            import decord
            decord.bridge.set_bridge("torch")
        except Exception:
            pass

        # Load the precomputed DROID index.
        idx_path = config.droid_index_path or config.data_path
        if not idx_path:
            raise ValueError(
                "DroidDatasetConfig.droid_index_path is empty; expected the "
                "JSON output of scripts/build_droid_index.py"
            )
        with open(idx_path) as f:
            idx = json.load(f)
        scenes: List[dict] = idx["scenes"]
        if not scenes:
            raise ValueError(f"droid_index has 0 scenes: {idx_path}")
        self._scenes = scenes
        self._mp4_stride = int(idx.get("mp4_stride", 4))
        self._gripper_rescale = float(config.gripper_rescale)

        self.traj_indices = list(range(len(scenes)))
        self.traj_idx_to_length = [int(s["total_clip_steps"]) for s in scenes]
        self._step_cumsum: dict[int, np.ndarray] = {}      # populated lazily for parent compat
        self._step_weights: dict[int, np.ndarray] = {}     # ditto

        # For each scene precompute a flattened "step lookup" so we can map a
        # uniform random integer in [0, total_clip_steps) → an absolute
        # trajectory step. ``_clip_offsets[i]`` is a list of (start_frame,
        # length, cum_length_before_this_clip) for each padded clip in scene i.
        self._clip_indices: List[List[Tuple[int, int, int]]] = []
        for s in scenes:
            offs = []
            cum = 0
            for cs, ce in s["clips"]:
                length = ce - cs + 1
                offs.append((int(cs), int(length), int(cum)))
                cum += length
            self._clip_indices.append(offs)

        log.info(
            f"DroidDataset: {len(scenes)} scenes, "
            f"{sum(self.traj_idx_to_length)} total trainable steps, "
            f"avg {sum(self.traj_idx_to_length)/max(1,len(scenes)):.1f} steps/scene"
        )

    # ---------- Dataset shape ----------

    def __len__(self) -> int:
        return len(self._scenes)

    def _get_file_and_traj_idx(self, global_idx: int) -> Tuple[int, int]:
        # We only have one "file" per scene; treat global_idx as the scene
        # index directly.
        return global_idx, 0

    # ---------- Step sampling ----------

    def _phase_eligible_steps(self, global_traj_idx: int) -> np.ndarray:
        """All steps inside any padded clip are eligible. Returned as a flat
        sorted array of absolute trajectory steps."""
        pieces = []
        for cs, length, _ in self._clip_indices[global_traj_idx]:
            pieces.append(np.arange(cs, cs + length, dtype=np.int32))
        return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.int32)

    def _sample_step_weighted(self, global_traj_idx: int, rng) -> int:
        """Uniform sampling over the union of padded clip ranges for this scene.

        Accepts either ``np.random.Generator`` (new API, has ``integers``) or
        ``np.random.RandomState`` (legacy API, has ``randint``); the trainer's
        per-worker rng is the legacy type."""
        total = self.traj_idx_to_length[global_traj_idx]
        if total <= 0:
            return 0
        if hasattr(rng, "integers"):
            u = int(rng.integers(0, total))
        else:
            u = int(rng.randint(0, total))
        for cs, length, cum in self._clip_indices[global_traj_idx]:
            if u < cum + length:
                return int(cs + (u - cum))
        # Defensive fallback (shouldn't happen if cumulative lengths are right)
        cs, length, _ = self._clip_indices[global_traj_idx][-1]
        return int(cs + length - 1)

    def _select_effective_cameras(self, rng, traj_group=None) -> List[str]:
        """Per-step camera selection: ``exo_camera_1`` slot is randomly one of
        ``{ext1, ext2}``; the wrist slot is always the wrist serial. Returned
        in the order ``self.camera_names`` (which is set at training launch)
        so parent helpers don't need to reorder.
        """
        # We don't actually need this method for our overridden _get_camera_frames,
        # but parent get() may call it. Return identity order (the camera_names
        # logical order) — _get_camera_frames does its own ext1/ext2 randomization.
        return list(self.camera_names)

    # ---------- Per-trajectory I/O ----------

    def _open_h5(self, scene: dict):
        import h5py
        return h5py.File(scene["trajectory_h5"], "r")

    def _read_state_and_actions(
        self, scene: dict, step: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One open of trajectory.h5 returns (state[8], actions[H,8], is_pad[H]).

        Slice-only reads (no full-array materialisation) so each get() call is
        O(action_horizon) regardless of trajectory length."""
        T = int(scene["trajectory_length"])
        H = self.action_horizon
        s = max(0, min(T - 1, int(step)))
        a_lo = step + 1
        a_hi = a_lo + H
        a_clip_lo = max(0, a_lo)
        a_clip_hi = min(T, a_hi)
        with self._open_h5(scene) as f:
            arm_st_row = f["observation/robot_state/joint_positions"][s : s + 1].astype(np.float32)
            grip_st_val = (
                f["observation/robot_state/gripper_position"][s : s + 1].astype(np.float32)
                * self._gripper_rescale
            )
            if a_clip_hi > a_clip_lo:
                arm_act = f["action/joint_position"][a_clip_lo:a_clip_hi].astype(np.float32)
                grip_act = (
                    f["action/gripper_position"][a_clip_lo:a_clip_hi].astype(np.float32)
                    * self._gripper_rescale
                )
            else:
                arm_act = np.zeros((0, 7), dtype=np.float32)
                grip_act = np.zeros((0,), dtype=np.float32)
            # For padding we may need the last valid sample explicitly
            arm_last = f["action/joint_position"][T - 1 : T].astype(np.float32)
            grip_last = (
                f["action/gripper_position"][T - 1 : T].astype(np.float32)
                * self._gripper_rescale
            )
        state = np.concatenate([arm_st_row[0], grip_st_val]).astype(np.float32)

        chunk = np.zeros((H, 8), dtype=np.float32)
        is_pad = np.zeros((H,), dtype=np.bool_)
        for k in range(H):
            t = a_lo + k
            if a_clip_lo <= t < a_clip_hi:
                ki = t - a_clip_lo
                chunk[k, :7] = arm_act[ki]
                chunk[k, 7] = grip_act[ki]
            else:
                # past trajectory: pad with last valid command
                chunk[k, :7] = arm_last[0]
                chunk[k, 7] = grip_last[0]
                is_pad[k] = True
        return state, chunk, is_pad

    def _get_camera_frames_droid(
        self,
        scene: dict,
        step: int,
        rng: random.Random,
    ) -> List[np.ndarray]:
        """Build the multi-camera × multi-frame image list for a sample.

        Returns a flat list of np.uint8 (H, W, 3) RGB arrays, in camera order
        (exo first, wrist second), with ``input_window_size`` frames per camera.
        Frame indices: ``mp4_frame_t = max(0, step − (input_window_size − 1 − k)
        × obs_step_delta) × mp4_stride`` for k in 0..input_window_size-1.
        """
        import decord
        # Per-step exo selection (ext1 vs ext2). When the wrist MP4 is
        # unavailable for this scene, fall back to the OTHER ext (the one
        # not picked for exo) in the wrist slot. Mirrors the index-build
        # tolerance for wrist-less scenes.
        ext1_path = scene["mp4_paths"]["ext1"]
        ext2_path = scene["mp4_paths"]["ext2"]
        pick_ext1 = bool(rng.random() < 0.5)
        exo_path = ext1_path if pick_ext1 else ext2_path
        if scene.get("wrist_available", True):
            wrist_path = scene["mp4_paths"]["wrist"]
        else:
            # Use the ext we didn't pick for exo as the "wrist" stand-in.
            wrist_path = ext2_path if pick_ext1 else ext1_path
        cam_paths = [exo_path, wrist_path]

        T_traj = int(scene["trajectory_length"])
        stride = int(self._mp4_stride)
        out: List[np.ndarray] = []
        for cam_path in cam_paths:
            try:
                vr = decord.VideoReader(cam_path, ctx=decord.cpu(0))
            except Exception as e:
                # If a camera fails to open, return zeros with a guessed shape.
                # The trainer's collator will then drop this sample.
                log.warning(f"DroidDataset: failed to open {cam_path}: {e}")
                # Return placeholders sized like a typical DROID frame.
                blank = np.zeros((480, 854, 3), dtype=np.uint8)
                for _ in range(self.input_window_size):
                    out.append(blank.copy())
                continue
            n_mp4 = len(vr)
            for k in range(self.input_window_size):
                t_step = max(0, step - (self.input_window_size - 1 - k) * self.config.obs_step_delta)
                t_step = min(T_traj - 1, t_step)
                mp4_frame = min(n_mp4 - 1, t_step * stride)
                try:
                    val = vr[int(mp4_frame)]
                    arr = (
                        val.asnumpy() if hasattr(val, "asnumpy")
                        else (val.numpy() if hasattr(val, "numpy") else np.asarray(val))
                    )
                    out.append(arr)
                except Exception as e:
                    log.warning(f"DroidDataset: read fail {cam_path}@{mp4_frame}: {e}")
                    out.append(np.zeros((480, 854, 3), dtype=np.uint8))
            del vr
        return out

    # ---------- Main entrypoint ----------

    def get(self, idx: int, rng: np.random.Generator) -> Dict[str, Any]:
        """Return one training example. Mirrors SynthmanipDataset.get's output
        contract: keys ``image``, ``question``, ``answers``, ``style``,
        ``state``, ``action``, ``action_is_pad``, ``metadata``."""
        global_traj_idx = self.traj_indices[idx]
        scene = self._scenes[global_traj_idx]
        step = self._sample_step_weighted(global_traj_idx, rng)

        # Frames. The trainer hands us a legacy ``RandomState`` (no ``integers``
        # method) — derive a python random.Random for the per-step ext1/ext2 coin.
        if hasattr(rng, "integers"):
            seed = int(rng.integers(0, 2**31 - 1))
        else:
            seed = int(rng.randint(0, 2**31 - 1))
        py_rng = random.Random(seed)
        image_list = self._get_camera_frames_droid(scene, step, py_rng)

        # State + action chunk via a single h5 open with sliced reads.
        state, actions, action_is_pad = self._read_state_and_actions(scene, step)

        # Goal: short Molmo2 caption for this scene
        goal = scene.get("caption", "") or ""

        # Normalize using the configured robot preprocessor (per repo_id).
        repo_id = "synthmanip"   # overloaded so eval inference wrapper picks up our stats
        if self.robot_preprocessor is not None:
            state = self.robot_preprocessor.normalize_state(state, repo_id)
            actions = self.robot_preprocessor.normalize_action(actions, repo_id)

        return {
            "image": image_list,
            "question": goal,
            "answers": "",
            "style": self.style,
            "state": state,
            "action": actions,
            "action_is_pad": action_is_pad,
            "metadata": {
                "traj_index": global_traj_idx,
                "step": step,
                "file_path": scene.get("scene_path", ""),
                "scene_uuid": scene.get("scene_uuid", ""),
                "traj_idx": 0,
                "split": self.config.split,
                "repo_id": repo_id,
                "effective_cameras": list(self.camera_names),
                "data_source": "droid",
            },
        }
