"""Trajectory 3D dataset package: base class + per-dataset subclasses + mixture.

Refactored from the former monolithic `trajectory_3d_dataset.py`. Public surface:

- `BaseTrajectoryDataset` — shared, single-dataset base (sampling, `_build_example`,
  text formatting, eval configs, video/depth I/O) with per-dataset hooks.
- 7 core training subclasses + 3 PointMotionBench eval subclasses.
- `TrajectoryMixtureDataset` / `build_from_tokens` — weighted multi-dataset mix.
- `DATASET_REGISTRY` — token → subclass.
"""

from molmo_motion.data.trajectory.base import BaseTrajectoryDataset
from molmo_motion.data.trajectory.constants import (
    LABEL_TEXT_2D_COORD,
    LABEL_TEXT_CONTROL_POINTS,
    LABEL_TEXT_ENDPOINT,
    LABEL_TEXT_FUTURE,
    LABEL_TEXT_HISTORY,
    LABEL_TEXT_V1,
    MOLMOSPACES_TIME_STRIDE,
)
from molmo_motion.data.trajectory.davis_bench import DavisBenchDataset
from molmo_motion.data.trajectory.droid import DroidDataset
from molmo_motion.data.trajectory.egodex import EgoDexDataset
from molmo_motion.data.trajectory.hdepic import HdEpicDataset
from molmo_motion.data.trajectory.hot3d_bench import Hot3DBenchDataset
from molmo_motion.data.trajectory.mixture import (
    DATASET_REGISTRY,
    TrajectoryMixtureDataset,
    build_from_tokens,
)
from molmo_motion.data.trajectory.molmospaces import MolmoSpacesDataset
from molmo_motion.data.trajectory.stereo4d import Stereo4DDataset
from molmo_motion.data.trajectory.worldtrack_bench import WorldTrackBenchDataset
from molmo_motion.data.trajectory.xperience import XperienceDataset
from molmo_motion.data.trajectory.ytvis import YTVisDataset

__all__ = [
    "BaseTrajectoryDataset",
    "TrajectoryMixtureDataset",
    "build_from_tokens",
    "DATASET_REGISTRY",
    "EgoDexDataset",
    "YTVisDataset",
    "HdEpicDataset",
    "XperienceDataset",
    "DroidDataset",
    "Stereo4DDataset",
    "MolmoSpacesDataset",
    "Hot3DBenchDataset",
    "WorldTrackBenchDataset",
    "DavisBenchDataset",
    "LABEL_TEXT_HISTORY",
    "LABEL_TEXT_FUTURE",
    "LABEL_TEXT_ENDPOINT",
    "LABEL_TEXT_2D_COORD",
    "LABEL_TEXT_CONTROL_POINTS",
    "LABEL_TEXT_V1",
    "MOLMOSPACES_TIME_STRIDE",
]
