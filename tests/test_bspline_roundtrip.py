"""Round-trip test spanning the dataset serialization (Task 2) and the eval
decode (Task 4): fit control points -> format as <tracks> text (dataset side)
-> parse + render back (evaluator side) -> compare to the original trajectory.

This is the integration check that the two sides agree on the control-point
answer format, without needing the full dataset (no videos required).
"""

import numpy as np
import torch

from molmo_motion.data.bspline import fit_control_points
from molmo_motion.data.trajectory_3d_dataset import (
    LABEL_TEXT_CONTROL_POINTS,
    Trajectory3DDataset,
)
from molmo_motion.eval.egodex_3d_evaluator import (
    control_points_to_traj,
    parse_tracks_text,
    tracks_to_control_points,
)


def _serialize_control_points(ctrl_np, valid_kp_np, n_ctrl):
    """Replicate the dataset's control-point emission (static formatter)."""
    P = ctrl_np.shape[0]
    ctrl_vis = np.broadcast_to(valid_kp_np[:, None], (P, n_ctrl))
    indices = [float(i) for i in range(n_ctrl)]
    return Trajectory3DDataset._format_tracks_text(
        indices, ctrl_np, ctrl_vis, LABEL_TEXT_CONTROL_POINTS)


def test_dataset_to_eval_roundtrip_smooth():
    """Fit -> serialize -> parse -> render recovers a smooth trajectory.

    Tolerance accounts for the ×1000 int quantization in the text format
    (~1e-3 m granularity) on top of the B-spline fit error.
    """
    F, D, P = 32, 10, 3
    t = torch.linspace(0, 1, F).unsqueeze(1)
    # Three distinct smooth trajectories (shared-anchor delta space).
    trajs = []
    for s in (1.0, -0.7, 0.4):
        trajs.append(torch.cat([0.30 * s * t,
                                 0.20 * torch.sin(1.5 * np.pi * t) * s,
                                 0.15 * t**2 * s], dim=1))
    future = torch.stack(trajs, dim=0)  # (P, F, 3)

    # Anchor = each point's t0 position (here 0 at t->0 for these curves).
    anchor = future[:, 0, :] * 0.0
    ctrl, valid_kp = fit_control_points(future, n_ctrl=D, anchor=anchor)

    # Dataset side: serialize to <tracks> text.
    text = _serialize_control_points(ctrl.numpy(), valid_kp.numpy(), D)
    assert LABEL_TEXT_CONTROL_POINTS in text
    assert text.count(";") == D - 1  # D rows -> D-1 separators

    # Eval side: parse + render.
    parsed = parse_tracks_text(text)
    assert parsed is not None
    ctrl_delta, ctrl_vis = tracks_to_control_points(parsed, P, D)
    assert ctrl_delta.shape == (P, D, 3)
    assert ctrl_vis.all()
    rendered = control_points_to_traj(ctrl_delta, F)  # (P, F, 3)
    assert rendered.shape == (P, F, 3)

    ade = np.linalg.norm(rendered - future.numpy(), axis=-1).mean()
    assert ade < 5e-3, f"round-trip ADE too high: {ade}"


def test_invalid_point_dropped_from_text():
    """A point with < D valid future frames is emitted as invalid (all-zero row)."""
    F, D, P = 32, 10, 2
    future = torch.zeros(P, F, 3)
    valid = torch.zeros(P, F, dtype=torch.bool)
    valid[0] = True             # point 0 fully valid
    valid[1, : D - 1] = True    # point 1 has D-1 valid -> dropped
    ctrl, valid_kp = fit_control_points(future, valid=valid, n_ctrl=D)
    assert bool(valid_kp[0]) and not bool(valid_kp[1])

    text = _serialize_control_points(ctrl.numpy(), valid_kp.numpy(), D)
    parsed = parse_tracks_text(text)
    _, ctrl_vis = tracks_to_control_points(parsed, P, D)
    # Point 0 present across all D rows; point 1 absent (sparse emission skipped it).
    assert ctrl_vis[0].all()
    assert not ctrl_vis[1].any()
