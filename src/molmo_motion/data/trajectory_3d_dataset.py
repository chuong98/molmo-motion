"""Back-compat shim for the former monolithic `Trajectory3DDataset`.

The dataset was refactored into `molmo_motion.data.trajectory` (a base class +
one subclass per dataset + a weighted mixture). This module preserves the old
import surface:

    from molmo_motion.data.trajectory_3d_dataset import (
        Trajectory3DDataset, DATASET_CONFIG, LABEL_TEXT_CONTROL_POINTS, ...)

`Trajectory3DDataset` is an alias of `BaseTrajectoryDataset` (single-dataset).
For multi-dataset training use `molmo_motion.data.trajectory.build_from_tokens`
(what `get_dataset_by_name` now calls). `DATASET_CONFIG` is reconstructed from
the per-dataset subclass class attrs for any consumer that inspects it.
"""

from molmo_motion.data.trajectory import (  # noqa: F401
    DATASET_REGISTRY,
    LABEL_TEXT_2D_COORD,
    LABEL_TEXT_CONTROL_POINTS,
    LABEL_TEXT_ENDPOINT,
    LABEL_TEXT_FUTURE,
    LABEL_TEXT_HISTORY,
    LABEL_TEXT_V1,
    MOLMOSPACES_TIME_STRIDE,
    BaseTrajectoryDataset,
    DavisBenchDataset,
    DroidDataset,
    EgoDexDataset,
    HdEpicDataset,
    Hot3DBenchDataset,
    MolmoSpacesDataset,
    Stereo4DDataset,
    TrajectoryMixtureDataset,
    WorldTrackBenchDataset,
    XperienceDataset,
    YTVisDataset,
    build_from_tokens,
)

# Old public name → single-dataset base. Multi-dataset mixing now lives in
# TrajectoryMixtureDataset / build_from_tokens.
Trajectory3DDataset = BaseTrajectoryDataset

# Reconstruct the old DATASET_CONFIG dict from the subclass class attrs, so
# consumers that read it (e.g. tests, cache builders) keep working. Only the
# 10 supported datasets (7 core + 3 benchmarks) appear.
DATASET_CONFIG = {
    token: {
        "data_root_env": cls.DATA_ROOT_ENV,
        "data_root_default": cls.DATA_ROOT_DEFAULT,
        "split_file": cls.SPLIT_FILE,
        "split_is_absolute": cls.SPLIT_IS_ABSOLUTE,
    }
    for token, cls in DATASET_REGISTRY.items()
}

__all__ = [
    "Trajectory3DDataset",
    "BaseTrajectoryDataset",
    "TrajectoryMixtureDataset",
    "build_from_tokens",
    "DATASET_REGISTRY",
    "DATASET_CONFIG",
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
