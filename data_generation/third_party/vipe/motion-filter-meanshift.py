#!/usr/bin/env python3
"""
Adaptive motion filtering using mean-shift clustering on 3D motion deltas.

Replaces the PCA-based motion-filter-rb.py which uses a fixed keep percentage.
Instead of hard thresholds, this uses mean-shift clustering to automatically
identify coherent motion patterns and filter out outlier points (tracking failures,
noise, static background anomalies).

Algorithm:
1. Compute 3D motion deltas between consecutive frames for every point
2. Build a per-point feature vector from those deltas
3. Run mean-shift clustering on the feature space
4. Keep points belonging to large clusters (dominant motion patterns)
5. Filter out points in small outlier clusters

Usage:
    python motion-filter-meanshift.py \
        --vid VIDEO_ID \
        --src /path/to/3d_tracks.npz \
        --dst /path/to/filtered_3d.npz \
        --tracks_2d /path/to/2d_tracks.npz \
        --tracks_2d_out /path/to/filtered_2d.npz \
        --min_cluster_size 5
"""

import argparse
from datetime import datetime

import numpy as np


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def select_points_with_meanshift(X, bandwidth=None, min_cluster_size=5):
    """
    Filter 3D trajectory points using mean-shift clustering on motion deltas.

    Args:
        X: (N, T, 3) array of 3D point trajectories. May contain NaN for
           invalid/occluded frames.
        bandwidth: Mean-shift bandwidth. None = auto-estimate via
                   sklearn.cluster.estimate_bandwidth().
        min_cluster_size: Minimum number of points a cluster must contain
                          to be kept. Clusters with fewer points are filtered
                          out as outliers.

    Returns:
        good_idx: Array of indices of inlier points to keep.
        labels: Cluster assignment for each point (-1 for filtered).
    """
    from sklearn.cluster import MeanShift, estimate_bandwidth

    N, T, C = X.shape
    assert C == 3

    if N == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    if T < 2:
        # Can't compute deltas with single frame, keep all
        return np.arange(N), np.zeros(N, dtype=int)

    # --- Step 1: Compute motion deltas between consecutive frames ---
    # X_filled: fill NaN with forward-fill, then backward-fill, then zero
    X_filled = X.copy()
    for n in range(N):
        for c in range(C):
            series = X_filled[n, :, c]
            nan_mask = np.isnan(series)
            if np.all(nan_mask):
                X_filled[n, :, c] = 0.0
                continue
            if np.any(nan_mask):
                # Forward fill
                last_valid = np.nan
                for t in range(T):
                    if np.isnan(series[t]):
                        if not np.isnan(last_valid):
                            series[t] = last_valid
                    else:
                        last_valid = series[t]
                # Backward fill remaining
                next_valid = np.nan
                for t in range(T - 1, -1, -1):
                    if np.isnan(series[t]):
                        if not np.isnan(next_valid):
                            series[t] = next_valid
                    else:
                        next_valid = series[t]
                X_filled[n, :, c] = series

    # Motion deltas: (N, T-1, 3)
    deltas = np.diff(X_filled, axis=1)

    # --- Step 2: Build feature vectors ---
    # Use the raw flattened deltas as features. Two points on the same rigid
    # object will have nearly identical per-frame displacements, so the raw
    # delta trajectory is a direct representation of motion.
    # deltas shape: (N, T-1, 3) -> flatten to (N, (T-1)*3)
    features = deltas.reshape(N, -1)  # (N, (T-1)*3)
    print(f"[{now()}] Feature dim: {features.shape[1]} (raw deltas, {deltas.shape[1]} frames x 3)")

    # Replace any remaining NaN/inf with 0
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # --- Step 3: Mean-shift clustering ---
    if bandwidth is None:
        # Auto-estimate bandwidth
        bandwidth = estimate_bandwidth(features, quantile=0.3, n_samples=min(500, N))
        if bandwidth <= 0:
            # Fallback: use median pairwise distance
            from sklearn.metrics import pairwise_distances
            sample_idx = np.random.choice(N, min(200, N), replace=False)
            dists = pairwise_distances(features[sample_idx])
            bandwidth = float(np.median(dists[dists > 0]))
            if bandwidth <= 0:
                # All points are identical, keep all
                return np.arange(N), np.zeros(N, dtype=int)

    print(f"[{now()}] Mean-shift bandwidth: {bandwidth:.6f}")

    try:
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        labels = ms.fit_predict(features)
    except ValueError as e:
        # When bandwidth is too small for bin_seeding, retry without it
        print(f"[{now()}] Mean-shift with bin_seeding failed ({e}), retrying without bin_seeding...")
        try:
            ms = MeanShift(bandwidth=bandwidth, bin_seeding=False)
            labels = ms.fit_predict(features)
        except ValueError:
            # Still fails — keep all points
            print(f"[{now()}] Mean-shift failed entirely, keeping all {N} points")
            labels = np.zeros(N, dtype=int)
            ms = None

    # --- Step 4: Identify dominant clusters ---
    unique_labels, counts = np.unique(labels, return_counts=True)
    min_size = min_cluster_size

    good_clusters = unique_labels[counts >= min_size]

    print(f"[{now()}] Found {len(unique_labels)} clusters:")
    for lbl, cnt in sorted(zip(unique_labels, counts), key=lambda x: -x[1]):
        status = "KEEP" if lbl in good_clusters else "FILTER"
        print(f"  Cluster {lbl}: {cnt} points ({100*cnt/N:.1f}%) [{status}]")

    # --- Step 5: Build good index mask ---
    good_mask = np.isin(labels, good_clusters)
    good_idx = np.where(good_mask)[0]

    print(f"[{now()}] Keeping {len(good_idx)}/{N} points ({100*len(good_idx)/N:.1f}%)")

    # Build cluster info for visualization
    cluster_info = {
        'labels': labels,                         # (N,) cluster assignment per point
        'cluster_centers': ms.cluster_centers_ if ms is not None else features[:1],  # (K, D)
        'unique_labels': unique_labels,
        'counts': counts,
        'good_clusters': good_clusters,
        'min_size': min_size,
        'bandwidth': bandwidth,
    }

    return good_idx, labels, cluster_info


# ─── CLI ─────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Mean-shift motion filtering for 3D tracks")
parser.add_argument("--vid", type=str, required=True, help="Video identifier")
parser.add_argument(
    "--src", type=str, required=True,
    help="Path to input *_3d_tracks.npz (key: points_3d, shape (N, T, 3))",
)
parser.add_argument(
    "--dst", type=str, required=True,
    help="Path to output filtered 3D tracks .npz",
)
parser.add_argument(
    "--tracks_2d", type=str, default="",
    help="Optional: path to 2D tracks .npz to filter alongside 3D tracks.",
)
parser.add_argument(
    "--tracks_2d_out", type=str, default="",
    help="Optional: path to save filtered 2D tracks .npz.",
)
parser.add_argument(
    "--bandwidth", type=float, default=None,
    help="Mean-shift bandwidth. Default: auto-estimate.",
)
parser.add_argument(
    "--min_cluster_size", type=int, default=5,
    help="Minimum number of points for a cluster to be kept (default: 5).",
)
args = parser.parse_args()

# ─── Main ────────────────────────────────────────────────────────────────────

print(f"[{now()}] Starting mean-shift motion filtering for vid={args.vid}")

X = np.load(args.src)['points_3d']  # (N, T, 3)
N = X.shape[0]

print(f"[{now()}] Input: N={N}, T={X.shape[1]}, shape={X.shape}")

good_idx, labels, cluster_info = select_points_with_meanshift(
    X, bandwidth=args.bandwidth, min_cluster_size=args.min_cluster_size
)

# Apply mask: set filtered points to NaN
mask = np.zeros(N, dtype=bool)
mask[good_idx] = True

points_3d_masked = X.copy()
points_3d_masked[~mask] = np.nan

# Save filtered 3D tracks
np.savez(args.dst, points_3d=points_3d_masked)
print(f"[{now()}] Saved filtered 3D tracks to {args.dst}")

# Compute and save cluster centroid 3D tracks for visualization.
# For each cluster, the centroid track is the mean 3D position of all member
# points at each timestep (using NaN-aware mean).
unique_labels = cluster_info['unique_labels']
counts_arr = cluster_info['counts']
good_clusters = cluster_info['good_clusters']

# centroid_tracks: (K, T, 3) — mean trajectory per cluster
# centroid_sizes: (K,) — number of member points
# centroid_kept: (K,) — bool, whether cluster is kept or filtered
K = len(unique_labels)
T = X.shape[1]
centroid_tracks = np.full((K, T, 3), np.nan, dtype=np.float32)
centroid_sizes = np.zeros(K, dtype=np.int32)
centroid_kept = np.zeros(K, dtype=bool)

for ki, lbl in enumerate(unique_labels):
    member_mask = (labels == lbl)
    member_tracks = X[member_mask]  # (n_members, T, 3)
    # NaN-aware mean per timestep
    with np.errstate(all='ignore'):
        centroid_tracks[ki] = np.nanmean(member_tracks, axis=0)
    centroid_sizes[ki] = int(counts_arr[ki])
    centroid_kept[ki] = lbl in good_clusters

# Save cluster metadata alongside the filtered output
import os
cluster_meta_path = os.path.splitext(args.dst)[0] + '_cluster_meta.npz'
np.savez(
    cluster_meta_path,
    centroid_tracks=centroid_tracks,   # (K, T, 3)
    centroid_sizes=centroid_sizes,     # (K,)
    centroid_kept=centroid_kept,       # (K,) bool
    labels=labels,                     # (N,) per-point cluster labels
    min_size=np.array([cluster_info['min_size']]),
    bandwidth=np.array([cluster_info['bandwidth']]),
)
print(f"[{now()}] Saved cluster metadata ({K} clusters) to {cluster_meta_path}")

# Filter and save 2D tracks if provided
if args.tracks_2d and args.tracks_2d_out:
    print(f"[{now()}] Loading 2D tracks from {args.tracks_2d}")
    tracks_2d_data = np.load(args.tracks_2d, allow_pickle=True)
    tracks_2d = tracks_2d_data['tracks']  # (T, N, 2) from merged tracks
    visibility = tracks_2d_data['visibility']  # (T, N)
    dim = tracks_2d_data['dim']  # (2,)

    # Transpose from (T, N, 2) to (N, T, 2) to match mask dimension
    tracks_2d = np.transpose(tracks_2d, (1, 0, 2))
    visibility = np.transpose(visibility, (1, 0))

    # Apply the same mask to 2D tracks
    tracks_2d_masked = tracks_2d.copy()
    visibility_masked = visibility.copy()

    tracks_2d_masked[~mask] = np.nan
    visibility_masked[~mask] = False

    # Save filtered 2D tracks (keep in N, T format for consistency with 3D)
    np.savez(args.tracks_2d_out,
             tracks=tracks_2d_masked,
             visibility=visibility_masked,
             dim=dim)
    print(f"[{now()}] Saved filtered 2D tracks to {args.tracks_2d_out}")
    print(f"[{now()}] Kept {mask.sum()}/{len(mask)} points ({100*mask.sum()/len(mask):.1f}%)")

print(f"[{now()}] Finished mean-shift filtering for vid={args.vid}")
