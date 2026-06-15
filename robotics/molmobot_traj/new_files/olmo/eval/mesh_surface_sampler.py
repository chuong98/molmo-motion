"""Sample surface points on a MuJoCo body's geoms — used at eval to replace
the ±5 cm random-cube point cloud with actual-object-shape samples so the
prompt's non-anchor deltas encode the real object geometry (matches training).

Strategy
--------
* Resolve the pickup body + all its descendant bodies.
* For each geom on those bodies, compute an approximate surface area and
  sample points uniformly on the surface (analytic for SPHERE/BOX/CAPSULE/
  CYLINDER/ELLIPSOID, face-uniform for MESH via triangles).
* Convert geom-local samples → the pickup body's LOCAL frame using
  (geom_pos, geom_quat) AND the parent-chain transform back to the pickup
  root. Body poses at frame 0 (mj_forward-called) fix the point set.
* Return ``(P, 3)`` samples in the pickup-body-local frame. The caller then
  reprojects via ``obj_pose_world_at(t) @ [p_local, 1]`` at each tick (the
  same flow as the existing cube-sample code path).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import mujoco


# ── Math utils ──────────────────────────────────────────────────────────────


def _quat_wxyz_to_mat(q: np.ndarray) -> np.ndarray:
    """(w,x,y,z) → 3x3 rotation matrix (MuJoCo convention)."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


# ── Body graph helpers ──────────────────────────────────────────────────────


def _descendant_bodies(model: mujoco.MjModel, root_body_id: int) -> List[int]:
    """Return ``root_body_id`` + every body with an ancestor chain reaching it."""
    children = [[] for _ in range(model.nbody)]
    for b in range(model.nbody):
        p = int(model.body_parentid[b])
        if p >= 0 and p != b:
            children[p].append(b)
    out = [root_body_id]
    stack = [root_body_id]
    while stack:
        cur = stack.pop()
        for c in children[cur]:
            out.append(c)
            stack.append(c)
    return out


def find_body_id(model: mujoco.MjModel, name_or_suffix: str) -> int:
    """Find a body id by exact name, and fall back to suffix-only match.

    Raises if ambiguous or missing."""
    exact = mujoco.mj_name2id(model, int(mujoco.mjtObj.mjOBJ_BODY), name_or_suffix)
    if exact >= 0:
        return exact
    needle = f"/{name_or_suffix}"
    hits = []
    for b in range(model.nbody):
        nm = model.body(b).name or ""
        if nm == name_or_suffix or nm.endswith(needle):
            hits.append(b)
    if not hits:
        raise ValueError(f"No body matching {name_or_suffix!r} in MuJoCo model.")
    if len(hits) > 1:
        raise ValueError(
            f"Ambiguous body name {name_or_suffix!r}: {len(hits)} matches. "
            f"Names: {[model.body(b).name for b in hits[:6]]}..."
        )
    return hits[0]


# ── Per-geom surface sampling ───────────────────────────────────────────────


def _sample_sphere_surface(rng: np.random.Generator, n: int, r: float) -> np.ndarray:
    pts = rng.normal(size=(n, 3)).astype(np.float64)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True) + 1e-12
    return (pts * r).astype(np.float64)


def _sample_box_surface(rng: np.random.Generator, n: int, hs: np.ndarray) -> np.ndarray:
    """Uniform over the 6 faces of a box with half-sizes ``hs = (hx, hy, hz)``."""
    hx, hy, hz = float(hs[0]), float(hs[1]), float(hs[2])
    face_areas = np.array([
        4 * hy * hz,  # ±x
        4 * hy * hz,
        4 * hx * hz,  # ±y
        4 * hx * hz,
        4 * hx * hy,  # ±z
        4 * hx * hy,
    ], dtype=np.float64)
    face_probs = face_areas / face_areas.sum()
    faces = rng.choice(6, size=n, p=face_probs)
    uv = rng.uniform(-1.0, 1.0, size=(n, 2))
    out = np.zeros((n, 3), dtype=np.float64)
    for i, f in enumerate(faces):
        u, v = uv[i]
        if f == 0: out[i] = [+hx, u * hy, v * hz]
        elif f == 1: out[i] = [-hx, u * hy, v * hz]
        elif f == 2: out[i] = [u * hx, +hy, v * hz]
        elif f == 3: out[i] = [u * hx, -hy, v * hz]
        elif f == 4: out[i] = [u * hx, v * hy, +hz]
        elif f == 5: out[i] = [u * hx, v * hy, -hz]
    return out


def _sample_capsule_surface(rng: np.random.Generator, n: int, r: float, half_len: float) -> np.ndarray:
    """Capsule aligned along +z, half-length ``half_len``, radius ``r``."""
    side_area = 2 * np.pi * r * (2 * half_len)
    cap_area = 4 * np.pi * r * r
    total = side_area + cap_area
    n_side = rng.binomial(n, side_area / total)
    n_cap = n - n_side
    theta = rng.uniform(0, 2 * np.pi, size=n_side)
    z = rng.uniform(-half_len, half_len, size=n_side)
    side = np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1)
    caps = _sample_sphere_surface(rng, n_cap, r)
    caps[:, 2] = np.where(caps[:, 2] >= 0, caps[:, 2] + half_len, caps[:, 2] - half_len)
    return np.concatenate([side, caps], axis=0)


def _sample_cylinder_surface(rng: np.random.Generator, n: int, r: float, half_len: float) -> np.ndarray:
    side_area = 2 * np.pi * r * (2 * half_len)
    cap_area = 2 * np.pi * r * r
    total = side_area + cap_area
    n_side = rng.binomial(n, side_area / total)
    n_cap = n - n_side
    theta = rng.uniform(0, 2 * np.pi, size=n_side)
    z = rng.uniform(-half_len, half_len, size=n_side)
    side = np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1)
    # Uniform disk sampling on each cap, sign chosen uniformly at random.
    r_d = r * np.sqrt(rng.uniform(0, 1, size=n_cap))
    theta_d = rng.uniform(0, 2 * np.pi, size=n_cap)
    sign_z = rng.choice([+1.0, -1.0], size=n_cap) * half_len
    caps = np.stack([r_d * np.cos(theta_d), r_d * np.sin(theta_d), sign_z], axis=1)
    return np.concatenate([side, caps], axis=0)


def _sample_ellipsoid_surface(rng: np.random.Generator, n: int, hs: np.ndarray) -> np.ndarray:
    """Surface sample on ellipsoid with semi-axes ``hs = (a, b, c)``.

    We use the standard ``unit-sphere → scale-by-axes`` trick. This is NOT
    area-uniform but is a fine proxy for small objects sampled as shape
    features."""
    pts = _sample_sphere_surface(rng, n, 1.0)
    return pts * hs[None, :]


def _sample_mesh_surface(
    rng: np.random.Generator,
    n: int,
    model: mujoco.MjModel,
    mesh_id: int,
) -> np.ndarray:
    """Uniform sampling over a mesh's triangles proportional to triangle area."""
    vadr = int(model.mesh_vertadr[mesh_id])
    vnum = int(model.mesh_vertnum[mesh_id])
    fadr = int(model.mesh_faceadr[mesh_id])
    fnum = int(model.mesh_facenum[mesh_id])
    verts = np.asarray(model.mesh_vert[vadr:vadr + vnum], dtype=np.float64)
    faces = np.asarray(model.mesh_face[fadr:fadr + fnum], dtype=np.int64)
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    probs = areas / (areas.sum() + 1e-12)
    tri_idx = rng.choice(fnum, size=n, p=probs)
    u = rng.uniform(0, 1, size=n)
    v = rng.uniform(0, 1, size=n)
    over = u + v > 1
    u = np.where(over, 1 - u, u)
    v = np.where(over, 1 - v, v)
    a = v0[tri_idx]; b = v1[tri_idx]; c = v2[tri_idx]
    return a + u[:, None] * (b - a) + v[:, None] * (c - a)


def _sample_geom_surface(
    rng: np.random.Generator,
    n: int,
    model: mujoco.MjModel,
    geom_id: int,
) -> Optional[np.ndarray]:
    """Return ``(n, 3)`` surface samples in this geom's local frame, or None
    if the geom type isn't supported."""
    gt = int(model.geom_type[geom_id])
    sz = np.asarray(model.geom_size[geom_id], dtype=np.float64)
    if gt == 2:   # SPHERE
        return _sample_sphere_surface(rng, n, float(sz[0]))
    if gt == 3:   # CAPSULE: radius sz[0], half-length sz[1]
        return _sample_capsule_surface(rng, n, float(sz[0]), float(sz[1]))
    if gt == 4:   # ELLIPSOID: semi-axes sz[0:3]
        return _sample_ellipsoid_surface(rng, n, sz[:3])
    if gt == 5:   # CYLINDER: radius sz[0], half-length sz[1]
        return _sample_cylinder_surface(rng, n, float(sz[0]), float(sz[1]))
    if gt == 6:   # BOX: half-sizes sz[0:3]
        return _sample_box_surface(rng, n, sz[:3])
    if gt == 7:   # MESH
        dataid = int(model.geom_dataid[geom_id])
        if dataid < 0:
            return None
        return _sample_mesh_surface(rng, n, model, dataid)
    return None   # PLANE / HFIELD: no meaningful surface for a small object


# ── Top-level entry ─────────────────────────────────────────────────────────


def sample_body_surface_points_local(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    root_body_id: int,
    num_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample ``num_points`` surface points on the union of all geoms under
    ``root_body_id`` and return their positions in ``root_body_id``'s LOCAL
    frame.

    The data argument must have been ``mj_forward``-ed so ``body_xpos`` /
    ``body_xquat`` / ``geom_xpos`` / ``geom_xmat`` are populated for the
    current episode's pose."""
    mujoco.mj_forward(model, data)

    body_ids = set(_descendant_bodies(model, root_body_id))
    # Root body world pose. MuJoCo exposes body pose as ``data.xpos`` /
    # ``data.xquat`` (N_body-indexed) after ``mj_forward``.
    T_root = np.eye(4, dtype=np.float64)
    T_root[:3, :3] = _quat_wxyz_to_mat(np.asarray(data.xquat[root_body_id]))
    T_root[:3, 3] = np.asarray(data.xpos[root_body_id])
    T_root_inv = np.linalg.inv(T_root)

    # Build the per-geom sampler pool (geom_id, area-weight).
    candidates: List[Tuple[int, float]] = []
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) not in body_ids:
            continue
        if int(model.geom_contype[g]) == 0 and int(model.geom_conaffinity[g]) == 0:
            # Visual-only geoms count too — we want surface points (shape info)
            # not contact points. Keep them all unless clearly degenerate.
            pass
        gt = int(model.geom_type[g])
        sz = np.asarray(model.geom_size[g], dtype=np.float64)
        # Rough area estimate per type (used only for weighting).
        if gt == 2:   area = 4 * np.pi * sz[0] ** 2
        elif gt == 3: area = 2 * np.pi * sz[0] * (2 * sz[1]) + 4 * np.pi * sz[0] ** 2
        elif gt == 4: area = 4 * np.pi * (sz[0] * sz[1] + sz[1] * sz[2] + sz[0] * sz[2]) / 3
        elif gt == 5: area = 2 * np.pi * sz[0] * (2 * sz[1]) + 2 * np.pi * sz[0] ** 2
        elif gt == 6: area = 8 * (sz[0] * sz[1] + sz[1] * sz[2] + sz[0] * sz[2])
        elif gt == 7:
            dataid = int(model.geom_dataid[g])
            if dataid < 0:
                continue
            vadr = int(model.mesh_vertadr[dataid])
            vnum = int(model.mesh_vertnum[dataid])
            verts = np.asarray(model.mesh_vert[vadr:vadr + vnum], dtype=np.float64)
            bbox = verts.max(0) - verts.min(0)
            area = 2 * (bbox[0] * bbox[1] + bbox[1] * bbox[2] + bbox[0] * bbox[2])
        else:
            continue
        if area <= 0 or not np.isfinite(area):
            continue
        candidates.append((g, float(area)))

    if not candidates:
        raise RuntimeError(
            f"No supported geoms found under body {model.body(root_body_id).name} "
            f"(descendants: {sorted(body_ids)})"
        )

    geom_ids = np.array([c[0] for c in candidates])
    weights = np.array([c[1] for c in candidates], dtype=np.float64)
    probs = weights / weights.sum()
    counts = rng.multinomial(num_points, probs)

    out_local = []
    for gi, n_i in zip(geom_ids, counts):
        if n_i == 0:
            continue
        samples_geom = _sample_geom_surface(rng, int(n_i), model, int(gi))
        if samples_geom is None:
            continue
        # geom frame → world frame: use mj_data.geom_xpos / geom_xmat
        R_gw = np.asarray(data.geom_xmat[int(gi)], dtype=np.float64).reshape(3, 3)
        t_gw = np.asarray(data.geom_xpos[int(gi)], dtype=np.float64)
        samples_world = (R_gw @ samples_geom.T).T + t_gw[None, :]
        # world → root-body-local
        homo = np.concatenate([samples_world, np.ones((samples_world.shape[0], 1))], axis=1)
        samples_local = (T_root_inv @ homo.T).T[:, :3]
        out_local.append(samples_local)

    pts = np.concatenate(out_local, axis=0)
    if pts.shape[0] > num_points:
        pts = pts[:num_points]
    elif pts.shape[0] < num_points:
        # Pad by duplicating the last one (rare — only happens when multinomial
        # sampling + one-geom skip hits an edge case).
        pad = np.tile(pts[-1:], (num_points - pts.shape[0], 1))
        pts = np.concatenate([pts, pad], axis=0)
    return pts.astype(np.float32)
