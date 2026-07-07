"""Characterization test for the refactored BaseTrajectoryDataset.

Loads the golden snapshot (`tests/data/golden_build_example.json`, generated
from the pre-refactor monolith by `tests/gen_golden_build_example.py`) and
asserts the refactored `BaseTrajectoryDataset._build_example` reproduces it
byte-for-byte in both frame mode and B-spline mode.

Mirrors the construction in `gen_golden_build_example.py`: instantiate via
`__new__` (skip `__init__`), set only the attrs `_build_example` reads, and
monkeypatch video I/O to dummy frames.
"""

import json
from pathlib import Path

import numpy as np

from molmo_motion.data.trajectory.base import BaseTrajectoryDataset


def build_synthetic_case():
    """A deterministic synthetic object trajectory + sampling indices.

    Identical to `tests/gen_golden_build_example.py::build_synthetic_case`.
    """
    rng = np.random.RandomState(0)
    N_obj, T = 12, 40
    P, H, F = 8, 3, 32
    t = np.linspace(0, 1, T)[None, :, None]
    base = rng.randn(N_obj, 1, 3) * 0.1
    pts_3d = (base + np.concatenate([0.3 * t, 0.2 * np.sin(3 * t), 0.15 * t**2], axis=2)).astype(np.float32)
    visibility = np.ones((N_obj, T), dtype=bool)
    chosen_indices = np.arange(P)
    hist_frames = [0, 1, 2]
    future_frames = list(range(3, 3 + F))
    t_query = 2
    entry = {"_dataset": "ytvis", "file": "vid_0001", "caption": "pick up the cup",
             "num_frames": T, "fps": 15}
    return dict(entry=entry, obj_name="obj_0", pts_3d=pts_3d, visibility=visibility,
                chosen_indices=chosen_indices, hist_frames=hist_frames,
                future_frames=future_frames, t=t_query, need_padding=False,
                P=P, H=H, F=F)


def make_dataset(bspline_n_ctrl):
    """Construct a BaseTrajectoryDataset without running __init__, setting only
    the attributes `_build_example` reads. Monkeypatch video I/O to dummy frames."""
    ds = BaseTrajectoryDataset.__new__(BaseTrajectoryDataset)
    ds.num_points = 8
    ds.num_future_frames = 32
    ds.history_size = 3
    ds.use_camera_frame = False
    ds.use_depth_token = False
    ds.use_2d_point_features = False
    ds.use_2d_coordinate = False
    ds.v1_match_format = False
    ds.predict_history_3d = False
    ds.pred_end_point_first = False
    ds.depth_target_size = 378
    ds.bspline_n_ctrl = bspline_n_ctrl
    ds.bspline_reg_lambda = 0.0
    ds.bspline_reg_order = 1
    ds.bspline_ctrl_clip = None
    # Dummy video path/frames so _build_example doesn't touch disk.
    ds._get_video_path = lambda entry: ""
    ds._map_frame_to_video = lambda entry, idxs: list(idxs)
    ds._read_video_frames = lambda path, idxs: np.zeros((len(idxs), 8, 8, 3), dtype=np.uint8)
    return ds


def snapshot(ds, case):
    ex = ds._build_example(
        case["entry"], case["obj_name"], case["pts_3d"], case["visibility"],
        case["chosen_indices"], case["hist_frames"], case["future_frames"],
        case["t"], case["need_padding"])
    msg = ex["message_list"][0]
    md = ex["metadata"]
    return {
        "question": msg["question"],
        "answer": msg["answer"],
        "metadata": {
            "example_id": md["example_id"],
            "gt_answer": md["gt_answer"],
            "gt_anchor": md["gt_anchor"],
            "gt_future_raw": md["gt_future_raw"],
            "gt_future_vis": md["gt_future_vis"],
            "bspline_n_ctrl": md.get("bspline_n_ctrl"),
            "future_horizon": md.get("future_horizon"),
            "bspline_valid_kp": md.get("bspline_valid_kp"),
        },
    }


def _load_golden():
    p = Path(__file__).parent / "data" / "golden_build_example.json"
    return json.loads(p.read_text())


def test_frame_mode_matches_golden():
    golden = _load_golden()["frame_mode"]
    case = build_synthetic_case()
    got = snapshot(make_dataset(0), case)
    assert got == golden


def test_bspline_mode_matches_golden():
    golden = _load_golden()["bspline_mode"]
    case = build_synthetic_case()
    got = snapshot(make_dataset(10), case)
    assert got == golden
