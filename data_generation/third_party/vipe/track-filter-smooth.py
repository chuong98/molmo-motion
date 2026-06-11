#!/usr/bin/env python3
"""
3D Track Pipeline: Consensus-Gated Stereo4D Ray-Only Smoothing

Pipeline:
  Step 0 (optional): Robust backprojection with depth patch median
  Step 1: Compute per-frame trust weights from anchor pairwise distance consistency
  Step 1.5: Spatial auto-split (split hand-like groups into left/right by 3D position)
  Step 2: Track filtering (drop tracks with too many low-trust frames)
  Step 3: Consensus-gated ray-only optimization (pin coherent frames, smooth incoherent)

Key idea: trust weights w_{t,n} in [0,1] drive everything:
  - w~1: coherent frame → pin delta~0 (no smoothing)
  - w~0: incoherent frame → allow delta to smooth (ray-acceleration penalty)
  - Tracks with too many w<w_min frames are dropped

Reference:
  Stereo4D track_optimization.py for ray-only optimization scaffold.
  Lessons: epsilon in sqrt, gradient clipping, per-object processing.

Usage:
  python track-filter-smooth.py --vid VID \
      --src 3d_tracks.npz --dst out_3d.npz \
      --tracks_2d merged.npz --tracks_2d_out out_2d.npz \
      --pose_npz pose.npz
"""
import argparse
import io
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ============================================================
# Step 0: Robust backprojection with depth patch median
# ============================================================


def _load_depth_frame(zf, frame_idx):
    """Load a single depth frame from an already-opened zip of EXR files."""
    import OpenEXR
    import Imath

    name = f"{frame_idx:05d}.exr"
    data = zf.read(name)
    exr = OpenEXR.InputFile(io.BytesIO(data))
    header = exr.header()
    dw = header["dataWindow"]
    w = dw.max.x - dw.min.x + 1
    h = dw.max.y - dw.min.y + 1
    pt = Imath.PixelType(Imath.PixelType.HALF)
    z_data = exr.channel("Z", pt)
    depth = np.frombuffer(z_data, dtype=np.float16).reshape(h, w).astype(np.float32)
    return depth


def robust_backproject(tracks_2d, visibility_2d, depth_zip_path, poses, intrinsics,
                       patch_k=5):
    """Step 0: Backproject 2D tracks to 3D using depth patch median."""
    from scipy.ndimage import median_filter

    T, N, _ = tracks_2d.shape
    P = np.full((T, N, 3), np.nan, dtype=np.float32)
    vis = np.zeros((T, N), dtype=bool)

    with zipfile.ZipFile(depth_zip_path) as zf:
        names_in_zip = set(zf.namelist())
        for t in range(T):
            name = f"{t:05d}.exr"
            if name not in names_in_zip:
                continue
            valid_mask = visibility_2d[t]
            if not np.any(valid_mask):
                continue
            depth_raw = _load_depth_frame(zf, t)
            H, W = depth_raw.shape
            depth_filtered = median_filter(depth_raw, size=patch_k)
            xy = tracks_2d[t, valid_mask]
            x_int = np.clip(np.round(xy[:, 0]).astype(int), 0, W - 1)
            y_int = np.clip(np.round(xy[:, 1]).astype(int), 0, H - 1)
            depth_vals = depth_filtered[y_int, x_int]
            valid_depth = np.isfinite(depth_vals) & (depth_vals > 0)
            if not np.any(valid_depth):
                continue
            fx, fy, cx, cy = intrinsics[t]
            X = (x_int[valid_depth] - cx) * depth_vals[valid_depth] / fx
            Y = (y_int[valid_depth] - cy) * depth_vals[valid_depth] / fy
            Z = depth_vals[valid_depth]
            pcd = np.stack([X, Y, Z], axis=1)
            c2w = poses[t]
            pcd_world = pcd @ c2w[:3, :3].T + c2w[:3, 3][None]
            visible_indices = np.where(valid_mask)[0][valid_depth]
            P[t, visible_indices] = pcd_world
            vis[t, visible_indices] = True
    valid_total = vis.sum()
    print(f"[{now()}] Step 0 done: {valid_total}/{T * N} valid points "
          f"({100 * valid_total / (T * N):.1f}%)")
    return P, vis


# ============================================================
# Step 1: Compute per-frame trust weights from anchor consensus
# ============================================================
#
# For each track n:
#   1. Compute pairwise distances to K anchor tracks: d_{n,a}(t) = ||P_{t,n} - P_{t,a}||
#   2. Temporal median: hat_d_{n,a} = median_t(d_{n,a}(t))
#   3. Deviation: e_n(t) = median_a( |d_{n,a}(t) - hat_d_{n,a}| )
#   4. Scale: s_n = MAD_t(e_n(t)) + epsilon
#   5. Trust: w_{t,n} = exp(-e_n(t) / s_n)
#
# Anchors: K tracks with lowest median speed (most stable).


def compute_trust_weights(P, vis, K=16, epsilon=1e-6):
    """Step 1: Compute per-frame trust weights from anchor distance consensus.

    Args:
        P: [T, N, 3] world points (NaN for invisible)
        vis: [T, N] bool visibility
        K: number of anchor tracks
        epsilon: scale floor for MAD

    Returns:
        w: [T, N] float in [0,1] — trust weight per frame per track
        anchor_idx: [K'] int — indices of anchor tracks used
    """
    T, N, _ = P.shape
    P_clean = np.nan_to_num(P, nan=0.0)

    # --- Select anchors: lowest median speed ---
    speeds = np.full(N, np.inf, dtype=np.float32)
    for n in range(N):
        both = vis[1:, n] & vis[:-1, n]
        if both.sum() < 3:
            continue
        s = np.linalg.norm(P_clean[1:, n] - P_clean[:-1, n], axis=-1)  # [T-1]
        speeds[n] = np.median(s[both])

    K_actual = min(K, (speeds < np.inf).sum())
    if K_actual < 2:
        # Not enough tracks — return neutral weights
        print(f"  [Trust] Only {K_actual} valid tracks, returning neutral weights")
        w = np.where(vis, 0.5, 0.0).astype(np.float32)
        return w, np.array([], dtype=int)

    anchor_idx = np.argsort(speeds)[:K_actual]
    print(f"  [Trust] Using {K_actual} anchors (median speed range: "
          f"{speeds[anchor_idx[0]]:.6f} to {speeds[anchor_idx[-1]]:.6f})")

    # --- Compute pairwise distance deviation to anchors ---
    # e_n(t) = median_a |d_{n,a}(t) - hat_d_{n,a}|
    e = np.full((T, N), np.nan, dtype=np.float32)

    for n in range(N):
        deviations = []  # list of [T] arrays
        for a in anchor_idx:
            if a == n:
                continue
            both_vis = vis[:, n] & vis[:, a]
            if both_vis.sum() < 3:
                continue
            # d_{n,a}(t) = ||P_{t,n} - P_{t,a}||
            d_na = np.full(T, np.nan, dtype=np.float32)
            d_na[both_vis] = np.linalg.norm(
                P[both_vis, n] - P[both_vis, a], axis=-1
            )
            hat_d_na = np.nanmedian(d_na)
            if np.isnan(hat_d_na):
                continue
            deviations.append(np.abs(d_na - hat_d_na))

        if len(deviations) == 0:
            continue

        # Stack and take median across anchors
        dev_stack = np.stack(deviations, axis=0)  # [K', T]
        e[:, n] = np.nanmedian(dev_stack, axis=0)

    # --- Convert deviation to trust weight ---
    # s_n = median_t(e_n(t)) + epsilon
    # w_{t,n} = exp(-e_n(t) / s_n)
    #
    # Using median(e) as scale (not MAD(e)) so that trust weights reflect
    # relative deviation: frames near the median get w ≈ 0.37, frames
    # well below median get w → 1, frames well above get w → 0.
    # This works even when noise is uniform (MAD(e) → 0 would kill all weights).
    w = np.zeros((T, N), dtype=np.float32)
    for n in range(N):
        en = e[:, n]
        valid = np.isfinite(en) & vis[:, n]
        if valid.sum() < 3:
            # Not enough data — assign neutral weight
            w[vis[:, n], n] = 0.5
            continue

        en_valid = en[valid]
        med_e = np.median(en_valid)
        s_n = med_e + epsilon

        # Trust weight
        w[valid, n] = np.exp(-en[valid] / s_n)
        # Visible but no deviation data → neutral
        no_dev = vis[:, n] & ~valid
        w[no_dev, n] = 0.5

    # Invisible frames get 0
    w[~vis] = 0.0

    # Stats
    vis_w = w[vis]
    print(f"  [Trust] Weight stats (visible frames): "
          f"median={np.median(vis_w):.3f}, "
          f"mean={np.mean(vis_w):.3f}, "
          f"<0.3: {(vis_w < 0.3).sum()}/{len(vis_w)} "
          f"({100 * (vis_w < 0.3).sum() / len(vis_w):.1f}%)")

    return w, anchor_idx


# ============================================================
# Step 1.5: Spatial auto-split (split hand-like groups into sub-objects)
# ============================================================


def spatial_auto_split(P, vis, min_gap_ratio=2.0, min_cluster_size=10):
    """Split an object group into spatial sub-groups using mean-shift on 3D positions.

    At the frame where most tracks are visible, run mean-shift on 3D positions
    to find spatially distinct clusters (e.g., left hand vs right hand).
    Sub-groups smaller than min_cluster_size are marked as noise and excluded.

    Args:
        P: [T, N, 3] world points (NaN for invisible)
        vis: [T, N] bool visibility
        min_gap_ratio: minimum ratio of inter-cluster to intra-cluster distance
        min_cluster_size: minimum number of tracks per sub-group (smaller → noise)

    Returns:
        sub_groups: list of index arrays for valid sub-groups.
        noise_idx: array of track indices considered noise (tiny sub-groups).
                   Empty array if no noise.
    """
    from sklearn.cluster import MeanShift, estimate_bandwidth

    T, N, _ = P.shape
    if N < 20:
        return [np.arange(N)], np.array([], dtype=int)

    # Find frame with most visible tracks
    vis_count = vis.sum(axis=1)
    best_t = int(np.argmax(vis_count))
    vis_t = vis[best_t]
    n_vis = vis_t.sum()

    if n_vis < 10:
        return [np.arange(N)], np.array([], dtype=int)

    pos = P[best_t, vis_t]  # [n_vis, 3]
    vis_indices = np.where(vis_t)[0]

    try:
        bw = estimate_bandwidth(pos, quantile=0.3)
        if bw < 1e-6:
            return [np.arange(N)], np.array([], dtype=int)
        ms = MeanShift(bandwidth=bw, bin_seeding=True)
        labels = ms.fit_predict(pos)
    except ValueError:
        try:
            ms = MeanShift(bandwidth=bw, bin_seeding=False)
            labels = ms.fit_predict(pos)
        except ValueError:
            return [np.arange(N)], np.array([], dtype=int)

    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    if n_clusters <= 1:
        return [np.arange(N)], np.array([], dtype=int)

    # Validate gap
    centroids = np.array([pos[labels == l].mean(axis=0) for l in unique_labels])
    min_inter = np.inf
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            d = np.linalg.norm(centroids[i] - centroids[j])
            min_inter = min(min_inter, d)

    max_intra = 0.0
    for l in unique_labels:
        pts = pos[labels == l]
        if len(pts) > 1:
            dists = np.linalg.norm(pts - pts.mean(axis=0), axis=1)
            max_intra = max(max_intra, np.percentile(dists, 90))

    if max_intra < 1e-8 or min_inter / max_intra < min_gap_ratio:
        return [np.arange(N)], np.array([], dtype=int)

    # Build sub-groups (assign non-visible tracks to nearest cluster)
    full_labels = np.full(N, -1, dtype=int)
    full_labels[vis_indices] = labels

    unassigned = np.where(full_labels == -1)[0]
    for n in unassigned:
        vis_frames = np.where(vis[:, n])[0]
        if len(vis_frames) == 0:
            full_labels[n] = unique_labels[0]
            continue
        t_first = vis_frames[0]
        pt = P[t_first, n]
        if not np.isfinite(pt).all():
            full_labels[n] = unique_labels[0]
            continue
        dists = np.linalg.norm(centroids - pt, axis=1)
        full_labels[n] = unique_labels[np.argmin(dists)]

    sub_groups = []
    noise_idx = []
    for l in unique_labels:
        idx = np.where(full_labels == l)[0]
        if len(idx) >= min_cluster_size:
            sub_groups.append(idx)
        elif len(idx) > 0:
            noise_idx.extend(idx.tolist())

    noise_idx = np.array(noise_idx, dtype=int)

    all_sizes = [len(sg) for sg in sub_groups]
    print(f"  [Spatial split] {len(sub_groups)} sub-groups at frame {best_t} "
          f"(gap ratio={min_inter / max(max_intra, 1e-8):.2f}): "
          f"sizes={all_sizes}")
    if len(noise_idx) > 0:
        print(f"    Noise tracks dropped: {len(noise_idx)} "
              f"(sub-groups < {min_cluster_size} tracks)")

    if len(sub_groups) == 0:
        # All sub-groups were too small — keep everything as one group
        return [np.arange(N)], np.array([], dtype=int)

    return sub_groups, noise_idx


# ============================================================
# Step 2: Track-level filtering based on trust weights
# ============================================================


def filter_tracks_by_trust(w, vis, z_thresh=2.0):
    """Step 2: Drop tracks whose mean trust weight is significantly below the group mean.

    Uses relative z-score filtering: computes each track's mean trust weight,
    then drops tracks whose mean is more than z_thresh MAD-based standard
    deviations below the group median. This handles uniform noise gracefully
    (won't drop everything when all tracks have similar noise).

    Args:
        w: [T, N] trust weights
        vis: [T, N] bool visibility
        z_thresh: z-score threshold (higher = more lenient, 2.0 keeps ~95%)

    Returns:
        drop: [N] bool — True for tracks to drop
    """
    T, N = w.shape

    # Compute per-track mean trust weight (over visible frames)
    mean_w = np.full(N, np.nan, dtype=np.float32)
    for n in range(N):
        vis_n = vis[:, n]
        if vis_n.sum() == 0:
            continue
        mean_w[n] = w[vis_n, n].mean()

    valid = np.isfinite(mean_w)
    drop = np.zeros(N, dtype=bool)

    # Drop tracks with zero visible frames
    drop[~valid] = True

    if valid.sum() < 3:
        print(f"[{now()}] Step 2 done: too few valid tracks ({valid.sum()}), no filtering")
        return drop

    # Robust z-score: (mean_w - median) / (1.4826 * MAD)
    mw_valid = mean_w[valid]
    median_mw = np.median(mw_valid)
    mad_mw = np.median(np.abs(mw_valid - median_mw))
    sigma = 1.4826 * mad_mw  # scale estimator (MAD → σ for normal dist)

    if sigma < 1e-8:
        # All tracks have same mean trust weight — no outliers
        print(f"[{now()}] Step 2 done: all tracks have similar trust (σ≈0), no filtering")
        return drop

    z_scores = (median_mw - mean_w) / sigma  # positive = below median
    drop[valid] = z_scores[valid] > z_thresh

    n_drop = int(drop.sum())
    print(f"[{now()}] Step 2 done: dropping {n_drop}/{N} tracks "
          f"({100 * n_drop / N:.1f}%)")
    print(f"  Mean trust weight: median={median_mw:.3f}, σ={sigma:.4f}")
    print(f"  Threshold: mean_w < {median_mw - z_thresh * sigma:.3f} (z>{z_thresh})")
    if n_drop > 0:
        dropped_mw = mean_w[drop & valid]
        print(f"  Dropped tracks: mean_w range [{dropped_mw.min():.3f}, {dropped_mw.max():.3f}]")

    return drop


# ============================================================
# Step 3: Consensus-gated ray-only optimization
# ============================================================
# L = L_pin + L_dyn + L_reg
#   L_pin = alpha * sum w * delta^2        (pin coherent frames)
#   L_dyn = sum (1-w)^p * a_ray^2          (smooth incoherent frames)
#   L_reg = lambda_reg * sum delta^2        (small global regularizer)


def _dilate_visibility(vis, window_size=5):
    """Shrink valid regions at visibility boundaries (following Stereo4D).

    vis: [N, T] bool
    Returns: [N, T] bool (shrunk)
    """
    N, T = vis.shape
    vis_out = vis.copy()
    half = window_size // 2
    for n in range(N):
        v = vis[n].astype(np.float32)
        for d in range(1, half + 1):
            left = np.zeros_like(v)
            left[d:] = v[:-d]
            left[:d] = v[0]
            right = np.zeros_like(v)
            right[:-d] = v[d:]
            right[-d:] = v[-1]
            vis_out[n] &= (left > 0.5) & (right > 0.5)
    return vis_out


def consensus_gated_smooth(P, cam_centers, vis, w,
                           alpha=1000.0, lambda_reg=1e-4, p=2,
                           deltas=(1, 3, 5), steps=100, lr=0.05,
                           device="cuda"):
    """Step 3: Consensus-gated ray-only optimization.

    Optimizes per-track per-frame scalar displacement delta[n,t] so that:
        P'[n,t] = P[n,t] + delta[n,t] * ray_unit[n,t]

    Loss:
        L_pin  = alpha * sum w_{t,n} * delta_{t,n}^2     (pin coherent)
        L_dyn  = sum_Delta (1-w_{t,n})^p * a_ray^2       (smooth incoherent)
        L_reg  = lambda_reg * sum delta_{t,n}^2           (global regularizer)

    Args:
        P: [T, N, 3] world points (NaN for invisible)
        cam_centers: [T, 3] camera centers
        vis: [T, N] bool visibility
        w: [T, N] float trust weights in [0,1]
        alpha: pin-to-zero strength for coherent frames
        lambda_reg: global delta regularizer
        p: gating power (higher = more localized smoothing)
        deltas: acceleration window sizes
        steps: Adam iterations
        lr: Adam learning rate
        device: torch device

    Returns:
        P_smooth: [T, N, 3] smoothed world points (NaN preserved)
    """
    T, N, _ = P.shape

    # Transpose to [N, T, 3]
    P_nt = np.nan_to_num(P.transpose(1, 0, 2), nan=0.0)  # [N, T, 3]
    vis_nt = vis.T.copy()  # [N, T]
    w_nt = w.T.copy()  # [N, T]

    # Dilate visibility (shrink valid regions at boundaries, following Stereo4D)
    vis_dilated = _dilate_visibility(vis_nt, window_size=5)

    # Camera centers: [T, 3] → [N, T, 3]
    C_nt = np.tile(cam_centers[None, :, :], (N, 1, 1))

    # Convert to torch
    P_t = torch.from_numpy(P_nt).float().to(device)
    C_t = torch.from_numpy(C_nt).float().to(device)
    mask_t = torch.from_numpy(vis_dilated).float().to(device)  # [N, T]
    w_t = torch.from_numpy(w_nt).float().to(device)  # [N, T]

    # Ray directions and distances (eps to avoid sqrt(0) gradient explosion)
    R = P_t - C_t  # [N, T, 3]
    R_sq = (R ** 2).sum(dim=-1)  # [N, T]
    R_norm = (R_sq + 1e-12).sqrt().unsqueeze(-1)  # [N, T, 1]
    R_unit = R / R_norm  # [N, T, 3]

    # Optimizable scalar displacement
    d = torch.zeros((N, T), device=device, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([d], lr=lr)

    for step in range(steps):
        opt.zero_grad(set_to_none=True)

        # Adjusted positions: P' = P + d * R_unit
        P_adj = P_t + d.unsqueeze(-1) * R_unit  # [N, T, 3]

        # --- L_pin: pin coherent frames to zero delta ---
        # L_pin = alpha * sum w * delta^2 * mask
        L_pin = alpha * (w_t * mask_t * d ** 2).sum()

        # --- L_dyn: gated ray-acceleration smoothing ---
        # gamma = (1 - w)^p * mask
        gamma = ((1.0 - w_t).clamp(min=0.0) ** p) * mask_t  # [N, T]

        L_dyn = torch.tensor(0.0, device=device)
        for delta in deltas:
            if T <= 2 * delta:
                continue
            # Shifted positions with edge padding
            P_plus = torch.cat(
                [P_adj[:, delta:, :], P_adj[:, -1:, :].expand(-1, delta, -1)], dim=1)
            P_minus = torch.cat(
                [P_adj[:, :1, :].expand(-1, delta, -1), P_adj[:, :-delta, :]], dim=1)
            mask_plus = torch.cat(
                [mask_t[:, delta:], mask_t[:, -1:].expand(-1, delta)], dim=1)
            mask_minus = torch.cat(
                [mask_t[:, :1].expand(-1, delta), mask_t[:, :-delta]], dim=1)

            valid_mask = mask_t * mask_plus * mask_minus  # [N, T]

            # Ray-projected acceleration
            accel = P_plus - 2 * P_adj + P_minus  # [N, T, 3]
            a_ray = (accel * R_unit).sum(dim=-1)  # [N, T]

            L_dyn = L_dyn + (gamma * valid_mask * a_ray ** 2).sum()

        # --- L_reg: small global delta regularizer ---
        L_reg = lambda_reg * (mask_t * d ** 2).sum()

        # --- Total ---
        total = L_pin + L_dyn + L_reg
        total.backward()

        # Gradient clipping for stability (lesson from Stereo4D implementation)
        torch.nn.utils.clip_grad_norm_([d], max_norm=1.0)
        opt.step()

        if step % 25 == 0 or step == steps - 1:
            print(f"  [{now()}] Step 3: iter {step}/{steps}, "
                  f"pin={L_pin.item():.4f}, dyn={L_dyn.item():.4f}, "
                  f"reg={L_reg.item():.6f}, |d|_max={d.abs().max().item():.6f}")

    # Apply smoothing
    with torch.no_grad():
        P_adj_final = P_t + d.unsqueeze(-1) * R_unit
        P_smooth_nt = P_adj_final.cpu().numpy()

    # Transpose back to [T, N, 3] and restore NaN
    P_smooth = P_smooth_nt.transpose(1, 0, 2)
    P_smooth[~vis] = np.nan

    d_np = d.detach().cpu().numpy()
    d_vis = d_np[vis_nt]
    # Sanity: check good vs bad frame deltas
    w_vis = w_nt[vis_nt]
    good_d = np.abs(d_vis[w_vis > 0.8])
    bad_d = np.abs(d_vis[w_vis < 0.3])
    print(f"[{now()}] Step 3 done: median |d| = {np.median(np.abs(d_vis)):.6f}")
    if len(good_d) > 0:
        print(f"  Good frames (w>0.8): median|d|={np.median(good_d):.6f}, "
              f"max|d|={np.max(good_d):.6f}")
    if len(bad_d) > 0:
        print(f"  Bad frames  (w<0.3): median|d|={np.median(bad_d):.6f}, "
              f"max|d|={np.max(bad_d):.6f}")

    return P_smooth


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="3D Track Pipeline: Consensus-Gated Stereo4D Smoothing"
    )
    parser.add_argument("--vid", required=True, help="Video ID")
    parser.add_argument("--src", required=True,
                        help="Unfiltered 3D tracks npz (N,T,3)")
    parser.add_argument("--dst", required=True,
                        help="Output filtered 3D tracks npz")
    parser.add_argument("--tracks_2d", required=True,
                        help="Merged 2D tracks npz (T,N,2)")
    parser.add_argument("--tracks_2d_out", required=True,
                        help="Output filtered 2D tracks npz")

    # Data for Step 0
    parser.add_argument("--depth_zip", default=None)
    parser.add_argument("--intrinsics_npz", default=None)

    # Camera poses (needed for Step 3 ray-only smoothing)
    parser.add_argument("--pose_npz", required=True,
                        help="Camera pose npz (c2w matrices)")

    # Step 0
    parser.add_argument("--patch_k", type=int, default=5)

    # Object grouping
    parser.add_argument("--object_sizes", default=None,
                        help="Comma-separated track counts per object (e.g. '200,100')")
    parser.add_argument("--object_names", default=None,
                        help="Pipe-separated object names matching object_sizes (e.g. 'knife|bowl')")

    # Step 1: Trust weight computation
    parser.add_argument("--n_anchors", type=int, default=16,
                        help="Number of anchor tracks for trust weights")
    parser.add_argument("--trust_epsilon", type=float, default=1e-6,
                        help="MAD floor for trust weight scale")

    # Step 2: Track-level filtering (relative z-score)
    parser.add_argument("--z_thresh", type=float, default=1.5,
                        help="z-score threshold: drop tracks whose mean trust "
                             "weight is > z_thresh MAD-σ below group median")

    # Step 3: Consensus-gated smoothing
    parser.add_argument("--smooth_steps", type=int, default=100,
                        help="Adam iterations")
    parser.add_argument("--smooth_lr", type=float, default=0.05,
                        help="Adam learning rate")
    parser.add_argument("--alpha", type=float, default=1000.0,
                        help="Pin-to-zero strength for coherent frames")
    parser.add_argument("--lambda_reg", type=float, default=1e-4,
                        help="Global delta regularizer weight")
    parser.add_argument("--gating_power", type=int, default=2,
                        help="Gating power p: gamma=(1-w)^p")

    # Device
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    print(f"[{now()}] Starting track pipeline for vid={args.vid}")

    # --- Load data ---
    d2 = np.load(args.tracks_2d)
    tracks_2d = d2["tracks"]       # [T, N, 2]
    vis_2d = d2["visibility"]      # [T, N]
    dim_2d = d2["dim"]             # [2] = (H, W)
    T, N, _ = tracks_2d.shape
    print(f"[{now()}] Loaded 2D tracks: T={T}, N={N}, dim={dim_2d}")

    # Camera poses
    pose_data = np.load(args.pose_npz)
    poses = pose_data["data"]
    T_pose = min(poses.shape[0], T)
    poses = poses[:T_pose]
    cam_centers = poses[:, :3, 3].copy()
    if T_pose < T:
        pad = np.tile(cam_centers[-1:], (T - T_pose, 1))
        cam_centers = np.concatenate([cam_centers, pad], axis=0)

    # --- Step 0: Robust backprojection (optional) ---
    do_step0 = (args.depth_zip is not None and args.intrinsics_npz is not None)
    if do_step0:
        print(f"\n[{now()}] === Step 0: Robust backprojection ===")
        intr_data = np.load(args.intrinsics_npz)
        intrinsics = intr_data["data"][:T]
        if intrinsics.shape[0] < T:
            pad = np.tile(intrinsics[-1:], (T - intrinsics.shape[0], 1))
            intrinsics = np.concatenate([intrinsics, pad], axis=0)
        P, vis = robust_backproject(
            tracks_2d, vis_2d, args.depth_zip, poses, intrinsics,
            patch_k=args.patch_k,
        )
    else:
        print(f"\n[{now()}] === Step 0: SKIPPED (using existing 3D tracks) ===")
        d3 = np.load(args.src)
        pts3d = d3["points_3d"]     # [N, T, 3]
        vis_3d = d3["visibility"]   # [N, T, 1]
        P = pts3d.transpose(1, 0, 2).copy()  # [T, N, 3]
        vis = vis_3d[:, :, 0].T.copy()       # [T, N]
        T_3d = P.shape[0]
        if T_3d != T:
            T = min(T_3d, T)
            P = P[:T]
            vis = vis[:T]
            tracks_2d = tracks_2d[:T]
            vis_2d = vis_2d[:T]
            cam_centers = cam_centers[:T]

    print(f"  P shape: {P.shape}, vis: {vis.sum()}/{vis.size} valid "
          f"({100 * vis.sum() / vis.size:.1f}%)")

    # --- Parse object sizes and names ---
    if args.object_sizes:
        obj_sizes = [int(x) for x in args.object_sizes.split(',')]
        assert sum(obj_sizes) == N, \
            f"Object sizes {obj_sizes} sum to {sum(obj_sizes)}, expected {N}"
        print(f"\n[{now()}] Per-object processing: {len(obj_sizes)} objects, sizes={obj_sizes}")
    else:
        obj_sizes = [N]
        print(f"\n[{now()}] Single-object processing: {N} tracks")

    # Parse object names (pipe-separated, must match obj_sizes count)
    obj_names = None
    if args.object_names:
        obj_names = args.object_names.split('|')
        assert len(obj_names) == len(obj_sizes), \
            f"Object names count {len(obj_names)} doesn't match object sizes count {len(obj_sizes)}"
        print(f"  Object names: {obj_names}")

    # Build object index ranges
    obj_ranges = []
    offset = 0
    for s in obj_sizes:
        obj_ranges.append((offset, offset + s))
        offset += s

    # --- Step 1 + 1.5 + 2: Per-object trust weights, spatial split, filtering ---
    trust_weights = np.zeros((T, N), dtype=np.float32)
    drop = np.zeros(N, dtype=bool)
    # sub_object_labels: which sub-object each track belongs to (for visualization)
    sub_object_labels = np.full(N, -1, dtype=int)
    sub_obj_counter = 0

    for obj_i, (start, end) in enumerate(obj_ranges):
        n_obj = end - start
        print(f"\n[{now()}] === Object {obj_i} (tracks {start}-{end-1}, {n_obj} tracks) ===")

        P_obj = P[:, start:end, :]
        vis_obj = vis[:, start:end]

        # Step 1.5: Spatial auto-split
        sub_groups, noise_idx = spatial_auto_split(P_obj, vis_obj)

        # Drop noise tracks from spatial split (tiny sub-groups)
        if len(noise_idx) > 0:
            noise_global = noise_idx + start
            drop[noise_global] = True
            print(f"  Dropping {len(noise_idx)} noise tracks from spatial split")

        for sg_i, sg_idx in enumerate(sub_groups):
            n_sub = len(sg_idx)
            global_idx = sg_idx + start  # Map back to global indices
            label_str = f"obj{obj_i}" if len(sub_groups) == 1 else f"obj{obj_i}_sub{sg_i}"
            print(f"\n  [{now()}] --- {label_str} ({n_sub} tracks) ---")

            P_sub = P_obj[:, sg_idx, :]
            vis_sub = vis_obj[:, sg_idx]

            # Assign sub-object label
            sub_object_labels[global_idx] = sub_obj_counter
            sub_obj_counter += 1

            # Step 1: Trust weights
            print(f"  Step 1: Computing trust weights (K={args.n_anchors})")
            w_sub, anchors = compute_trust_weights(
                P_sub, vis_sub,
                K=args.n_anchors,
                epsilon=args.trust_epsilon,
            )
            trust_weights[:, global_idx] = w_sub

            # Step 2: Track-level filtering (relative z-score)
            print(f"  Step 2: Filtering tracks (z_thresh={args.z_thresh})")
            drop_sub = filter_tracks_by_trust(
                w_sub, vis_sub,
                z_thresh=args.z_thresh,
            )
            drop[global_idx] = drop_sub

    # --- Step 3: Consensus-gated smoothing on kept tracks ---
    # Smoothing is frame-level: L_pin pins good frames (w~1 → delta~0),
    # L_dyn smooths bad frames (w~0 → allow delta to adjust along ray).
    keep_mask = ~drop
    N_keep = int(keep_mask.sum())
    print(f"\n[{now()}] === Step 3: Consensus-gated ray-only smoothing "
          f"({N_keep}/{N} kept tracks, alpha={args.alpha}, "
          f"lambda_reg={args.lambda_reg}, p={args.gating_power}) ===")

    if N_keep == 0:
        print(f"[{now()}] WARNING: All tracks dropped! Keeping all tracks instead.")
        drop[:] = False
        keep_mask = ~drop
        N_keep = N

    # Smooth only kept tracks
    P_kept = P[:, keep_mask, :]
    vis_kept = vis[:, keep_mask]
    w_kept = trust_weights[:, keep_mask]

    P_smoothed_kept = consensus_gated_smooth(
        P_kept, cam_centers, vis_kept, w_kept,
        alpha=args.alpha,
        lambda_reg=args.lambda_reg,
        p=args.gating_power,
        steps=args.smooth_steps,
        lr=args.smooth_lr,
        device=args.device,
    )

    # Build full smoothed array (all tracks, smoothing applied only to kept)
    P_smoothed_all = P.copy()
    P_smoothed_all[:, keep_mask, :] = P_smoothed_kept

    # --- Save outputs ---
    print(f"\n[{now()}] Final: keeping {N_keep}/{N} tracks ({100 * N_keep / N:.1f}%)")

    # 3D output: [N_keep, T, 3]
    P_out = P_smoothed_kept.transpose(1, 0, 2)  # [N_keep, T, 3]
    vis_out = vis_kept.T[:, :, None]  # [N_keep, T, 1]

    dst = Path(args.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    if obj_names is not None:
        # Nested per-object format: {obj_name: array} for each object
        # Build per-object keep masks relative to the original object ranges
        per_obj_3d = {}
        per_obj_2d_tracks = {}
        per_obj_2d_vis = {}
        for obj_i, (start, end) in enumerate(obj_ranges):
            name = obj_names[obj_i]
            obj_keep = keep_mask[start:end]
            # 3D: from smoothed kept tracks, select those belonging to this object
            obj_P = P_smoothed_all[:, start:end, :][:, obj_keep, :]  # [T, n_kept_obj, 3]
            per_obj_3d[name] = obj_P.transpose(1, 0, 2)  # [n_kept_obj, T, 3]
            # 2D: same filtering
            per_obj_2d_tracks[name] = tracks_2d[:, start:end, :][:, obj_keep, :]  # [T, n_kept_obj, 2]
            per_obj_2d_vis[name] = vis_2d[:, start:end][:, obj_keep]  # [T, n_kept_obj]
            print(f"  {name}: {obj_keep.sum()}/{end - start} tracks kept")

        np.savez(dst, points_3d=per_obj_3d)
        print(f"[{now()}] Saved nested per-object 3D tracks to {dst}")

        tracks_2d_out_path = Path(args.tracks_2d_out)
        tracks_2d_out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(tracks_2d_out_path, tracks=per_obj_2d_tracks,
                 visibility=per_obj_2d_vis, dim=dim_2d)
        print(f"[{now()}] Saved nested per-object 2D tracks to {tracks_2d_out_path}")
    else:
        # Flat format (legacy, used by egodex)
        np.savez(dst, points_3d=P_out, visibility=vis_out)
        print(f"[{now()}] Saved filtered 3D tracks to {dst}")
        print(f"  points_3d: {P_out.shape}, visibility: {vis_out.shape}")

        # 2D output: [T, N_keep, 2]
        tracks_2d_out = tracks_2d[:, keep_mask, :]
        vis_2d_out = vis_2d[:, keep_mask]

        tracks_2d_out_path = Path(args.tracks_2d_out)
        tracks_2d_out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(tracks_2d_out_path, tracks=tracks_2d_out, visibility=vis_2d_out,
                 dim=dim_2d)
        print(f"[{now()}] Saved filtered 2D tracks to {tracks_2d_out_path}")

    # Save intermediate metadata for visualization
    meta_path = dst.parent / f"{args.vid}_filter_meta.npz"
    np.savez(
        meta_path,
        # Original 3D tracks (before any processing) — all tracks [N, T, 3]
        P_original=P.transpose(1, 0, 2),
        # Smoothed 3D tracks (after Step 3) — all tracks [N, T, 3]
        P_smoothed=P_smoothed_all.transpose(1, 0, 2),
        # Visibility — all tracks [N, T]
        visibility_all=vis.T,
        # Trust weights — all tracks [N, T]
        trust_weights=trust_weights.T,  # [N, T]
        # Keep mask [N]
        keep_mask=keep_mask,
        # Drop mask [N]
        drop=drop,
        # Sub-object labels [N] (for coloring in visualization)
        sub_object_labels=sub_object_labels,
    )
    print(f"[{now()}] Saved filter metadata to {meta_path}")

    print(f"\n[{now()}] Finished track pipeline for vid={args.vid}")
    print(f"  Input:  {N} tracks, {T} frames")
    print(f"  Output: {N_keep} tracks ({N - N_keep} dropped)")


if __name__ == "__main__":
    main()
