"""Tests for the standalone cubic B-spline utility (data/bspline.py).

Validates: basis partition-of-unity, fit→render reconstruction of smooth curves,
near-exact straight-line recovery, occlusion-weighted fitting, and the valid_kp rule.
"""

import numpy as np
import torch

from molmo_motion.data.bspline import (
    build_bspline_basis,
    fit_control_points,
    render_control_points,
)


def test_basis_shapes_and_partition_of_unity():
    F, D = 32, 10
    N_anchor = build_bspline_basis(D, F, include_anchor=True)
    N_render = build_bspline_basis(D, F, include_anchor=False)
    assert tuple(N_anchor.shape) == (F + 1, D)
    assert tuple(N_render.shape) == (F, D)
    # Rows sum to 1 (partition of unity).
    assert torch.allclose(N_anchor.sum(dim=1), torch.ones(F + 1), atol=1e-5)
    assert torch.allclose(N_render.sum(dim=1), torch.ones(F), atol=1e-5)


def test_invalid_n_ctrl_raises():
    import pytest

    with pytest.raises(ValueError):
        build_bspline_basis(5, 32)


def test_smooth_cubic_reconstructs():
    """A smooth curved trajectory is recovered by 10 control points within ~1e-3."""
    F, D = 32, 10
    t = torch.linspace(0, 1, F).unsqueeze(1)
    smooth = torch.cat([0.30 * t, 0.20 * torch.sin(1.5 * np.pi * t), 0.15 * t**2], dim=1)
    future = smooth.unsqueeze(0)  # (1, F, 3)

    ctrl, valid_kp = fit_control_points(future, n_ctrl=D)
    assert tuple(ctrl.shape) == (1, D, 3)
    assert bool(valid_kp[0])

    rendered = render_control_points(ctrl, horizon=F)  # (1, F, 3)
    assert tuple(rendered.shape) == (1, F, 3)
    ade = (rendered - future).norm(dim=-1).mean()
    assert ade < 1e-3, f"smooth reconstruction ADE too high: {ade}"


def test_straight_line_near_exact():
    F, D = 32, 10
    t = torch.linspace(0, 1, F).unsqueeze(1)
    line = (t * torch.tensor([[0.4, -0.3, 0.1]])).unsqueeze(0)  # (1, F, 3)
    ctrl, _ = fit_control_points(line, n_ctrl=D)
    rendered = render_control_points(ctrl, horizon=F)
    assert (rendered - line).norm(dim=-1).max() < 1e-4


def test_anchor_row_pins_start():
    """With a consistent anchor, the rendered curve starts at the anchor position.

    The anchor is a soft (equally-weighted) least-squares row, so it must agree
    with the trajectory's start to be honored exactly. In the dataset the anchor
    is the point's own t0 position, i.e. always consistent with the future's start.
    Here we use a line through the origin with anchor=0 (consistent).
    """
    F, D = 32, 10
    t = torch.linspace(0, 1, F).unsqueeze(1)
    future = (t * torch.tensor([[0.4, -0.3, 0.1]])).unsqueeze(0)  # starts ~0 at t->0
    anchor = torch.zeros(1, 3)
    ctrl_a, _ = fit_control_points(future, n_ctrl=D, anchor=anchor)
    ctrl_n, _ = fit_control_points(future, n_ctrl=D)  # no anchor
    basis = build_bspline_basis(D, F, include_anchor=True)
    start_a = torch.einsum("hd,pdc->phc", basis, ctrl_a)[0, 0]
    start_n = torch.einsum("hd,pdc->phc", basis, ctrl_n)[0, 0]
    # Anchor is a soft (equally-weighted) row: it pulls the t=0 sample close to the
    # anchor (within ~1% of the trajectory scale) and closer than the un-anchored fit.
    assert start_a.norm() < 5e-3
    assert start_a.norm() < start_n.norm()


def test_occlusion_weighted_fit():
    """Masked (occluded) frames are ignored by the weighted fit."""
    F, D = 32, 10
    t = torch.linspace(0, 1, F).unsqueeze(1)
    smooth = torch.cat([0.30 * t, 0.20 * torch.sin(1.5 * np.pi * t), 0.15 * t**2], dim=1)
    future = smooth.unsqueeze(0)
    valid = torch.ones(1, F, dtype=torch.bool)
    valid[0, 5:12] = False  # occlude a chunk
    ctrl, valid_kp = fit_control_points(future, valid=valid, n_ctrl=D)
    assert bool(valid_kp[0])
    rendered = render_control_points(ctrl, horizon=F)
    # Error on the *visible* frames should stay small.
    err = (rendered - future).norm(dim=-1)[0]
    assert err[valid[0]].mean() < 5e-3


def test_valid_kp_rule():
    """A point with fewer than D valid frames is marked invalid."""
    F, D = 32, 10
    future = torch.zeros(2, F, 3)
    valid = torch.zeros(2, F, dtype=torch.bool)
    valid[0, :D] = True       # exactly D valid -> valid
    valid[1, : D - 1] = True  # D-1 valid -> invalid
    _, valid_kp = fit_control_points(future, valid=valid, n_ctrl=D)
    assert bool(valid_kp[0]) is True
    assert bool(valid_kp[1]) is False


def test_accepts_numpy_input():
    F, D = 32, 10
    future = np.random.randn(3, F, 3).astype(np.float32)
    ctrl, valid_kp = fit_control_points(future, n_ctrl=D)
    assert tuple(ctrl.shape) == (3, D, 3)
    rendered = render_control_points(ctrl, horizon=F)
    assert tuple(rendered.shape) == (3, F, 3)


def test_clip_bounds_control_points():
    F, D = 32, 10
    t = torch.linspace(0, 1, F).unsqueeze(1)
    big = (t * torch.tensor([[100.0, 100.0, 100.0]])).unsqueeze(0)
    ctrl, _ = fit_control_points(big, n_ctrl=D, clip=5.0)
    assert ctrl.abs().max() <= 5.0 + 1e-6
