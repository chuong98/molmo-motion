"""Tests for TrajectoryMixtureDataset + build_from_tokens (the refactored
multi-dataset mixing). Data-free: uses stub single-dataset objects and
monkeypatched subclass __init__ so nothing touches disk."""

import numpy as np
import pytest

from molmo_motion.data.trajectory import (
    DATASET_REGISTRY,
    TrajectoryMixtureDataset,
    build_from_tokens,
)


class _StubSub:
    """Minimal stand-in for a single-dataset BaseTrajectoryDataset."""

    def __init__(self, token, n_entries, is_eval=False, n_eval=0):
        self.TOKEN = token
        self.entries = [{"file": f"{token}_{i}", "_dataset": token} for i in range(n_entries)]
        self.is_eval = is_eval
        self.eval_configs = [{"cfg": i} for i in range(n_eval)] if is_eval else None
        self.num_points = 8
        self.num_future_frames = 32
        self.history_size = 3
        self.data_root = f"/data/{token}"

    def get(self, item, rng):
        return {"token": self.TOKEN, "item": item}


def test_sqrt_weighting_matches_sqrt_of_counts():
    subs = [_StubSub("egodex", 900), _StubSub("droid", 100)]
    mix = TrajectoryMixtureDataset(subs, dataset_weighting="sqrt")
    # sqrt(900)=30, sqrt(100)=10 -> [0.75, 0.25]
    assert np.allclose(mix._probs, [0.75, 0.25])


def test_uniform_and_naive_weighting():
    subs = [_StubSub("a", 900), _StubSub("b", 100)]
    assert np.allclose(TrajectoryMixtureDataset(subs, "uniform")._probs, [0.5, 0.5])
    # naive == proportional to counts
    assert np.allclose(TrajectoryMixtureDataset(subs, "naive")._probs, [0.9, 0.1])


def test_manual_dict_weighting():
    subs = [_StubSub("a", 10), _StubSub("b", 10), _StubSub("c", 10)]
    mix = TrajectoryMixtureDataset(subs, {"a": 1.0, "b": 3.0, "c": 0.0})
    assert np.allclose(mix._probs, [0.25, 0.75, 0.0])


def test_manual_dict_missing_token_raises():
    subs = [_StubSub("a", 10), _StubSub("b", 10)]
    with pytest.raises(ValueError):
        TrajectoryMixtureDataset(subs, {"a": 1.0})


def test_train_len_and_sampling_routes_by_weight():
    subs = [_StubSub("a", 3), _StubSub("b", 7)]
    mix = TrajectoryMixtureDataset(subs, "naive")
    assert len(mix) == 10
    # With naive weights [0.3, 0.7], forcing rng.choice to return sub index works.
    rng = np.random.RandomState(0)
    out = mix.get(0, rng)
    assert out["token"] in ("a", "b")


def test_eval_routing_concatenates_in_order():
    subs = [_StubSub("a", 0, is_eval=True, n_eval=2),
            _StubSub("b", 0, is_eval=True, n_eval=3)]
    mix = TrajectoryMixtureDataset(subs, "sqrt")
    assert mix.is_eval
    assert len(mix) == 5
    assert len(mix.eval_configs) == 5
    # item 0,1 -> sub a local 0,1 ; item 2,3,4 -> sub b local 0,1,2
    rng = np.random.RandomState(0)
    assert mix.get(0, rng)["token"] == "a"
    assert mix.get(2, rng)["token"] == "b"
    assert mix.get(4, rng) == {"token": "b", "item": 2}


def test_mixed_split_raises():
    subs = [_StubSub("a", 5, is_eval=False), _StubSub("b", 0, is_eval=True, n_eval=1)]
    with pytest.raises(ValueError):
        TrajectoryMixtureDataset(subs, "sqrt")


def test_registry_has_10_datasets():
    assert set(DATASET_REGISTRY) == {
        "egodex", "ytvis", "hepic", "xperience", "droid", "stereo4d",
        "molmospaces", "hot3d_bench", "worldtrack_bench", "davis_bench",
    }


def test_build_from_tokens_maps_and_forwards(monkeypatch):
    """build_from_tokens instantiates the right subclasses and forwards kwargs."""
    seen = []

    def fake_init(self, split="train", **kwargs):
        # Emulate a constructed single-dataset instance.
        self.TOKEN = type(self).TOKEN
        self.entries = [{"file": "x", "_dataset": self.TOKEN}]
        self.is_eval = split in ("validation", "test", "train_test")
        self.eval_configs = [{"c": 0}] if self.is_eval else None
        self.num_points = kwargs.get("num_points", 8)
        self.num_future_frames = kwargs.get("num_future_frames", 32)
        self.history_size = kwargs.get("history_size", 3)
        self.data_root = f"/data/{self.TOKEN}"
        seen.append((self.TOKEN, split, kwargs.get("bspline_n_ctrl")))

    for cls in DATASET_REGISTRY.values():
        monkeypatch.setattr(cls, "__init__", fake_init)

    mix = build_from_tokens(("egodex", "droid"), "train",
                            dataset_weighting="sqrt", num_points=8, bspline_n_ctrl=10)
    assert isinstance(mix, TrajectoryMixtureDataset)
    assert [s.TOKEN for s in mix.subs] == ["egodex", "droid"]
    assert {t for t, _, _ in seen} == {"egodex", "droid"}
    assert all(ck == 10 for _, _, ck in seen)


def test_build_from_tokens_unknown_token_raises():
    with pytest.raises(ValueError):
        build_from_tokens(("not_a_dataset",), "train")
