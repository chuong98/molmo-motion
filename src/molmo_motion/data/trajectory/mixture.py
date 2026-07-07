"""Multi-dataset mixture over per-dataset `BaseTrajectoryDataset` instances.

Replaces the monolith's internal `_ds_probs`/`_ds_to_indices` weighting: each
sub-dataset is a single-dataset instance, and this class samples which one to
draw from (by weight) at training time, or concatenates their deterministic eval
configs at eval time. Satisfies the `Dataset` interface (`__len__`, `get`,
`download`) so `DeterministicDataset` / `IterableDatasetMixture` wrap it unchanged.

Weighting parity with the old monolith:
- "sqrt"   → weight_i ∝ sqrt(len(sub_i.entries))   (default)
- "uniform"→ weight_i ∝ 1
- "naive"  → weight_i ∝ len(sub_i.entries)          (== concatenation)
- dict     → manual per-token weights (every token in the mix must appear)
Within the chosen sub-dataset, an entry is sampled uniformly (as before).
"""

from __future__ import annotations

import logging

import numpy as np

from molmo_motion.data.dataset import Dataset
from molmo_motion.data.trajectory.base import BaseTrajectoryDataset
from molmo_motion.data.trajectory.davis_bench import DavisBenchDataset
from molmo_motion.data.trajectory.droid import DroidDataset
from molmo_motion.data.trajectory.egodex import EgoDexDataset
from molmo_motion.data.trajectory.hdepic import HdEpicDataset
from molmo_motion.data.trajectory.hot3d_bench import Hot3DBenchDataset
from molmo_motion.data.trajectory.molmospaces import MolmoSpacesDataset
from molmo_motion.data.trajectory.stereo4d import Stereo4DDataset
from molmo_motion.data.trajectory.worldtrack_bench import WorldTrackBenchDataset
from molmo_motion.data.trajectory.xperience import XperienceDataset
from molmo_motion.data.trajectory.ytvis import YTVisDataset

log = logging.getLogger(__name__)

# token → subclass. Only the 7 core training datasets + 3 PointMotionBench
# eval benchmarks are supported (hand/davis/hotworld legacy variants dropped).
DATASET_REGISTRY: dict[str, type[BaseTrajectoryDataset]] = {
    "egodex": EgoDexDataset,
    "ytvis": YTVisDataset,
    "hepic": HdEpicDataset,
    "xperience": XperienceDataset,
    "droid": DroidDataset,
    "stereo4d": Stereo4DDataset,
    "molmospaces": MolmoSpacesDataset,
    "hot3d_bench": Hot3DBenchDataset,
    "worldtrack_bench": WorldTrackBenchDataset,
    "davis_bench": DavisBenchDataset,
}


class TrajectoryMixtureDataset(Dataset):
    """Weighted mixture of single-dataset `BaseTrajectoryDataset` instances."""

    def __init__(self, sub_datasets, dataset_weighting="sqrt"):
        if not sub_datasets:
            raise ValueError("TrajectoryMixtureDataset needs at least one sub-dataset")
        self.subs = list(sub_datasets)
        self.dataset_weighting = dataset_weighting

        # All sub-datasets share the same split mode.
        self.is_eval = self.subs[0].is_eval
        if any(s.is_eval != self.is_eval for s in self.subs):
            raise ValueError("all sub-datasets must share the same split (train/eval)")

        # Compatibility attrs some consumers read off the dataset object.
        self.num_points = self.subs[0].num_points
        self.num_future_frames = self.subs[0].num_future_frames
        self.history_size = self.subs[0].history_size
        self.data_roots = {s.TOKEN: s.data_root for s in self.subs}
        self.entries = [e for s in self.subs for e in s.entries]

        # Eval: concatenate each sub's deterministic configs, remembering which
        # sub owns each (routing preserves per-dataset order for reproducibility).
        if self.is_eval:
            self._eval_routing = [
                (si, li) for si, s in enumerate(self.subs)
                for li in range(len(s.eval_configs))
            ]
            self.eval_configs = [self.subs[si].eval_configs[li]
                                 for si, li in self._eval_routing]
        else:
            self._eval_routing = None
            self.eval_configs = None

        # Train: per-sub sampling probability.
        self._probs = None if self.is_eval else self._compute_probs()

        names = ", ".join(f"{s.TOKEN}={len(s.entries)}" for s in self.subs)
        mixinfo = ""
        if self._probs is not None:
            mixinfo = " mix=" + ", ".join(
                f"{s.TOKEN}={p:.3f}" for s, p in zip(self.subs, self._probs))
        log.info(f"[TrajMixture] {'eval' if self.is_eval else 'train'}: "
                 f"{len(self.subs)} datasets ({names}){mixinfo}")

    def _compute_probs(self):
        counts = np.array([len(s.entries) for s in self.subs], dtype=np.float64)
        w = self.dataset_weighting
        if isinstance(w, dict):
            tokens = [s.TOKEN for s in self.subs]
            missing = [t for t in tokens if t not in w]
            if missing:
                raise ValueError(f"manual dataset_weighting missing weights for: {missing}")
            weights = np.array([float(w[t]) for t in tokens], dtype=np.float64)
            if (weights < 0).any() or weights.sum() <= 0:
                raise ValueError(f"manual weights must be non-negative and sum > 0: {weights}")
        elif w == "sqrt":
            weights = np.sqrt(counts)
        elif w == "uniform":
            weights = np.ones_like(counts)
        elif w == "naive":
            weights = counts
        else:
            raise NotImplementedError(f"dataset_weighting={w!r}")
        return weights / weights.sum()

    # ── Dataset interface ───────────────────────────────────────────────

    @classmethod
    def download(cls, n_procs=1):
        pass

    def __len__(self):
        if self.is_eval:
            return len(self._eval_routing)
        return sum(len(s.entries) for s in self.subs)

    def get(self, item, rng):
        if self.is_eval:
            sub_idx, local_idx = self._eval_routing[item % len(self._eval_routing)]
            return self.subs[sub_idx].get(local_idx, rng)
        # Train: pick a sub-dataset by weight, then let it sample its own entry.
        sub_idx = int(rng.choice(len(self.subs), p=self._probs))
        return self.subs[sub_idx].get(item, rng)


def build_from_tokens(tokens, split, dataset_weighting="sqrt", **kwargs):
    """Instantiate one `BaseTrajectoryDataset` per token and wrap in a mixture.

    Args:
        tokens: iterable of dataset tokens (keys of DATASET_REGISTRY).
        split: "train"/"validation"/"test"/"train_test".
        dataset_weighting: mixture weighting (see class docstring).
        **kwargs: forwarded to each subclass __init__ (num_points, history_size,
            num_future_frames, use_2d_point_features, use_2d_coordinate,
            use_camera_frame, max_eval_per_dataset, bspline_n_ctrl, ...).

    Returns:
        A `TrajectoryMixtureDataset` (even for a single token, so the interface
        is uniform).
    """
    unknown = [t for t in tokens if t not in DATASET_REGISTRY]
    if unknown:
        raise ValueError(
            f"unknown dataset token(s) {unknown}; supported: {sorted(DATASET_REGISTRY)}")
    subs = [DATASET_REGISTRY[t](split=split, **kwargs) for t in tokens]
    return TrajectoryMixtureDataset(subs, dataset_weighting=dataset_weighting)
