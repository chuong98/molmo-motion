"""Shared constants for the trajectory dataset package.

Extracted verbatim from the monolithic `trajectory_3d_dataset.py` (data roots,
prompt/answer label strings, and the molmospaces stride / visibility knobs).
"""

import os

# ── Data roots ──────────────────────────────────────────────────────────

# Legacy split metadata (only for datasets NOT in molmo-motion-1m: davis,
# hotworld — not part of the released recipe).
METADATA_ROOT = os.environ.get(
    "MOLMO_MOTION_METADATA_ROOT", "data/motion_filtering_splits")

# Root of the MolmoMotion-1M corpus (egodex, ytvis, hepic, xperience, droid,
# stereo4d, molmospaces). Each dataset lives under
# `<MOLMO_MOTION_1M_ROOT>/<dataset>/` with sub-directories `annotations/`,
# `tracks/`, `videos/`. Download from HF `allenai/molmo-motion-1m`; see the
# top-level README. Set via `export MOLMO_MOTION_1M_ROOT=...`.
MOLMO_MOTION_1M_ROOT = os.environ.get(
    "MOLMO_MOTION_1M_ROOT", "data/molmo-motion-1m")

# Root of the PointMotionBench eval suite (hot3d, worldtrack, davis).
# Download from HF `allenai/PointMotionBench`; set via
# `export POINTMOTIONBENCH_ROOT=...`.
POINTMOTIONBENCH_ROOT = os.environ.get(
    "POINTMOTIONBENCH_ROOT", "data/PointMotionBench")

# Hand datasets: objects are always left_hand/right_hand, clip = full video
HAND_OBJECTS = ["left_hand", "right_hand"]

LABEL_TEXT_HISTORY = "3d object history"
LABEL_TEXT_FUTURE = "3d object trajectories"
LABEL_TEXT_ENDPOINT = "3d object endpoints"
LABEL_TEXT_2D_COORD = "2d object point"
# B-spline control-point answer mode: the future is emitted as D control-point
# rows (leading number = control-point index 0..D-1) instead of F frame rows.
# The distinct label lets the evaluator/decoder pick the render path.
LABEL_TEXT_CONTROL_POINTS = "3d object control points"
# v1-era (traj3d_droid_v1) label: a single "object points" string was used for
# both prompt and answer <tracks> blocks. Preserved for the _v1match flag.
LABEL_TEXT_V1 = "object points"
# Under molmo-motion-1m, DROID MP4s are pre-resampled to 15 fps (matching
# the track rate). The old multiplier from when video was 60 fps and tracks
# were 15 fps is no longer needed; we keep the constant for any consumer
# that imports it but set it to 1.
DROID_FPS_MULTIPLIER = 1
MOLMOSPACES_VIS_THRESHOLD = 0.5
# Gripper moves too slowly in the source clips; decimate by this factor so each
# strided track frame = every Nth raw frame (and the video is seeked 3× further
# per step). Applied everywhere molmospaces is loaded (entries, 3D, 2D, video).
# Overridable per-job via env var MOLMOSPACES_STRIDE (e.g. set to 2 for F=30
# finetune so molmospaces clips have enough strided frames to fit H+F=33).
MOLMOSPACES_TIME_STRIDE = int(os.environ.get("MOLMOSPACES_STRIDE", "4"))
