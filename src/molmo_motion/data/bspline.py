"""Cubic B-spline utilities: represent a trajectory as D control points.

MolmoMotion's B-spline mode serializes the future trajectory as ``D`` control
points instead of ``F`` frames (see ``docs/bspline_control_points_plan.md``).
This module provides the three primitives that mode needs:

- ``build_bspline_basis(n_ctrl, horizon, include_anchor)`` — the fixed cubic
  basis matrix (data-independent; rows are a partition of unity).
- ``fit_control_points(...)`` — weighted least-squares fit of a trajectory to
  ``D`` control points (with optional t=0 anchor pin, Tikhonov smoothing, clip).
- ``render_control_points(ctrl, horizon)`` — render control points back to an
  ``(P, F, 3)`` trajectory.

Ported/adapted from mu0 (``lerobot/datasets/bspline_basis.py`` and
``lerobot/datasets/trace_dataset.py`` — the reg-matrix builder and the batched
weighted lstsq). Implemented in torch (batched ``torch.linalg.lstsq``); all
functions accept array-like input via ``torch.as_tensor`` and return torch
tensors, so the numpy-side dataset can call them and convert with ``.numpy()``.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

_DEGREE = 3

# Precomputed knot vectors (multiplicity-3 internal knots -> Bezier-like cubics).
# Only {4, 7, 10} are supported; other sizes would need runtime knot generation.
_PRECOMPUTED_KNOTS: dict[int, Tensor] = {
    4: torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]),
    7: torch.tensor([0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0]),
    10: torch.tensor([
        0.0, 0.0, 0.0, 0.0,
        1 / 3, 1 / 3, 1 / 3,
        2 / 3, 2 / 3, 2 / 3,
        1.0, 1.0, 1.0, 1.0,
    ]),
}


def _knot_diffs(n: int) -> tuple[Tensor, Tensor]:
    knots = _PRECOMPUTED_KNOTS[n]
    denom1 = torch.zeros(n, _DEGREE + 1)
    denom2 = torch.zeros(n, _DEGREE + 1)
    for k in range(_DEGREE + 1):
        for i in range(n):
            denom1[i, k] = knots[i + k] - knots[i] if i + k < len(knots) else 0.0
            denom2[i, k] = (
                knots[i + k + 1] - knots[i + 1] if i + k + 1 < len(knots) else 1.0
            )
    return denom1, denom2


_PRECOMPUTED_DENOMS: dict[int, tuple[Tensor, Tensor]] = {
    n: _knot_diffs(n) for n in (4, 7, 10)
}


def _basis_at(n_ctrl: int, t: Tensor) -> Tensor:
    """Evaluate the ``n_ctrl`` cubic B-spline basis functions at params ``t``."""
    knots = _PRECOMPUTED_KNOTS[n_ctrl].to(dtype=t.dtype, device=t.device)
    denom1, denom2 = (
        d.to(dtype=t.dtype, device=t.device) for d in _PRECOMPUTED_DENOMS[n_ctrl]
    )
    T = t.shape[0]
    basis = torch.zeros(T, n_ctrl, _DEGREE + 1, dtype=t.dtype, device=t.device)

    for i in range(n_ctrl):
        if i == n_ctrl - 1:
            basis[:, i, 0] = ((knots[i] <= t) & (t <= knots[i + 1])).to(t.dtype)
        else:
            basis[:, i, 0] = ((knots[i] <= t) & (t < knots[i + 1])).to(t.dtype)

    for k in range(1, _DEGREE + 1):
        for i in range(n_ctrl):
            term1 = torch.zeros_like(t)
            term2 = torch.zeros_like(t)
            if denom1[i, k] > 0:
                term1 = ((t - knots[i]) / denom1[i, k]) * basis[:, i, k - 1]
            if denom2[i, k] > 0 and i + 1 < n_ctrl:
                term2 = ((knots[i + k + 1] - t) / denom2[i, k]) * basis[:, i + 1, k - 1]
            basis[:, i, k] = term1 + term2

    return basis[:, :, _DEGREE]


def build_bspline_basis(
    n_ctrl: int,
    horizon: int,
    *,
    include_anchor: bool = True,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Cubic B-spline basis matrix at uniform-in-t sample points.

    Args:
        n_ctrl: number of control points (must be in {4, 7, 10}).
        horizon: number of future timesteps H.
        include_anchor: when True, prepend a row at t=0 (returns ``(H+1, n_ctrl)``);
            when False, rows at ``t_i = (i+1)/H`` only (returns ``(H, n_ctrl)``).

    Returns:
        Tensor ``(T, n_ctrl)`` whose rows sum to 1 (partition of unity).
    """
    if n_ctrl not in _PRECOMPUTED_KNOTS:
        raise ValueError(f"n_ctrl must be in {{4, 7, 10}}; got {n_ctrl}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1; got {horizon}")

    if include_anchor:
        t = torch.arange(horizon + 1, dtype=dtype, device=device) / horizon
    else:
        t = (torch.arange(horizon, dtype=dtype, device=device) + 1) / horizon
    return _basis_at(n_ctrl, t)


def _build_reg_matrix(n_ctrl: int, order: int, *, dtype=torch.float32) -> Tensor:
    """Finite-difference regularization matrix Gamma for Tikhonov smoothing.

    order=1 -> (n_ctrl-1, n_ctrl) rows [-1, 1]  (penalize adjacent-ctrl differences).
    order=2 -> (n_ctrl-2, n_ctrl) rows [-1, 2, -1] (penalize second differences).
    """
    if order == 1:
        gamma = torch.zeros(n_ctrl - 1, n_ctrl, dtype=dtype)
        for i in range(n_ctrl - 1):
            gamma[i, i] = -1.0
            gamma[i, i + 1] = 1.0
    elif order == 2:
        gamma = torch.zeros(n_ctrl - 2, n_ctrl, dtype=dtype)
        for i in range(n_ctrl - 2):
            gamma[i, i] = -1.0
            gamma[i, i + 1] = 2.0
            gamma[i, i + 2] = -1.0
    else:
        raise ValueError(f"reg_order must be 1 or 2; got {order}")
    return gamma


def fit_control_points(
    future: Tensor,
    valid: Optional[Tensor] = None,
    n_ctrl: int = 10,
    *,
    anchor: Optional[Tensor] = None,
    reg_lambda: float = 0.0,
    reg_order: int = 1,
    clip: Optional[float] = None,
) -> tuple[Tensor, Tensor]:
    """Fit ``n_ctrl`` cubic B-spline control points to each point's trajectory.

    Solves, per point, the (optionally validity-weighted and Tikhonov-regularized)
    least-squares problem ``min_P ||N P - X||^2 + ||reg_lambda * Gamma P||^2``.

    Args:
        future: ``(P, F, 3)`` trajectory to fit (in whatever delta space the caller uses).
        valid: optional ``(P, F)`` bool/float mask over the F frames; occluded frames
            are dropped from the objective. Defaults to all-valid.
        n_ctrl: number of control points D (in {4, 7, 10}).
        anchor: optional ``(P, 3)`` position to pin at t=0. When given, it is prepended
            as an always-valid t=0 row using the ``include_anchor=True`` basis; when
            None, the fit uses the F-row ``include_anchor=False`` basis.
        reg_lambda: Tikhonov strength (0 disables). Stacks ``reg_lambda * Gamma``.
        reg_order: finite-difference order for Gamma (1 or 2).
        clip: optional symmetric clamp applied to each control-point component.

    Returns:
        ``ctrl`` ``(P, D, 3)`` control points and ``valid_kp`` ``(P,)`` bool
        (True iff the point has >= D valid future frames).
    """
    future = torch.as_tensor(future, dtype=torch.float32)
    if future.ndim != 3 or future.shape[-1] != 3:
        raise ValueError(f"future must be (P, F, 3); got {tuple(future.shape)}")
    P, F, _ = future.shape
    device = future.device

    if valid is None:
        valid = torch.ones(P, F, dtype=torch.float32, device=device)
    else:
        valid = torch.as_tensor(valid, device=device).to(torch.float32)

    use_anchor = anchor is not None
    basis = build_bspline_basis(
        n_ctrl, F, include_anchor=use_anchor, dtype=torch.float32, device=device
    )  # (F+1, D) or (F, D)

    if use_anchor:
        anchor = torch.as_tensor(anchor, dtype=torch.float32, device=device)
        target = torch.cat([anchor[:, None, :], future], dim=1)  # (P, F+1, 3)
        m_anchor = torch.ones(P, 1, dtype=torch.float32, device=device)
        mask = torch.cat([m_anchor, valid], dim=1)  # (P, F+1)
    else:
        target = future
        mask = valid  # (P, F)

    # Row-weighted, batched over points.
    basis_p = basis.unsqueeze(0).expand(P, -1, -1)          # (P, T, D)
    A_w = basis_p * mask.unsqueeze(-1)                      # (P, T, D)
    B_w = target * mask.unsqueeze(-1)                       # (P, T, 3)

    if reg_lambda > 0.0:
        gamma = _build_reg_matrix(n_ctrl, reg_order, dtype=torch.float32).to(device)
        gamma_p = (reg_lambda * gamma).unsqueeze(0).expand(P, -1, -1)   # (P, D-k, D)
        zeros_pad = torch.zeros(P, gamma.shape[0], 3, dtype=torch.float32, device=device)
        A_w = torch.cat([A_w, gamma_p], dim=1)
        B_w = torch.cat([B_w, zeros_pad], dim=1)

    ctrl = torch.linalg.lstsq(A_w, B_w).solution           # (P, D, 3)

    if clip is not None:
        ctrl = ctrl.clamp(min=-clip, max=clip)

    valid_kp = valid.sum(dim=-1) >= n_ctrl                 # (P,)
    ctrl = torch.where(valid_kp[:, None, None], ctrl, torch.zeros_like(ctrl))
    return ctrl, valid_kp


def render_control_points(ctrl: Tensor, horizon: int) -> Tensor:
    """Render control points to a full-horizon trajectory.

    Args:
        ctrl: ``(P, D, 3)`` control points.
        horizon: number of future frames F to render.

    Returns:
        ``(P, F, 3)`` trajectory (``include_anchor=False`` basis — the F future frames).
    """
    ctrl = torch.as_tensor(ctrl, dtype=torch.float32)
    D = ctrl.shape[-2]
    basis = build_bspline_basis(
        D, horizon, include_anchor=False, dtype=ctrl.dtype, device=ctrl.device
    )  # (F, D)
    return torch.einsum("hd,pdc->phc", basis, ctrl)
