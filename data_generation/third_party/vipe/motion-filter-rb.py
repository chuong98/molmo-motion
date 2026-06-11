import numpy as np
import os
import argparse
from datetime import datetime


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def select_points_with_pca(X, n_components=6, n_keep=8):
    N, T, C = X.shape
    assert C == 3
    
    # reorder to (T, N, 3)
    X_tnc = np.transpose(X, (1, 0, 2))  # (T,N,3)

    # ---- robust NaN filling ----
    X_tnc = X_tnc.copy()

    for t in range(T):
        frame = X_tnc[t]
        mask = np.any(np.isnan(frame), axis=1)

        if np.all(mask):
            if t == 0:
                X_tnc[t] = 0.0
            else:
                X_tnc[t] = X_tnc[t-1]
            continue

        centroid = np.nanmean(frame, axis=0)
        frame[mask] = centroid

    # ---- remove per-frame translation ----
    centroids = X_tnc.mean(axis=1, keepdims=True)
    X_centered = X_tnc - centroids

    Y = X_centered.reshape(T, 3*N)
    Y_mean = Y.mean(axis=0, keepdims=True)
    Yc = Y - Y_mean

    Yc = Yc + 1e-6 * np.random.randn(*Yc.shape)

    if np.allclose(Yc, 0):
        good_idx = np.arange(min(N, n_keep))
        point_errors = np.zeros(N)
        return good_idx, point_errors

    U, S, Vt = np.linalg.svd(Yc, full_matrices=False)

    k = min(n_components, Vt.shape[0])
    Vt_k = Vt[:k]

    scores = Yc @ Vt_k.T
    Yc_hat = scores @ Vt_k
    Y_hat = Yc_hat + Y_mean

    X_hat_centered = Y_hat.reshape(T, N, 3)

    err = X_centered - X_hat_centered
    point_errors = np.sqrt((err**2).sum(axis=(0,2)) / T)

    good_idx = np.argsort(point_errors)[:n_keep]

    return good_idx, point_errors


# ----------------------------------------------------------------------
# argparse
parser = argparse.ArgumentParser()
parser.add_argument("--vid", type=str, required=True, help="Identifier used for default src/dst naming")
parser.add_argument(
    "--src",
    type=str,
    default="",
    help="Optional: path to input *_3d_tracks.npz (overrides default HD-EPIC path).",
)
parser.add_argument(
    "--dst",
    type=str,
    default="",
    help="Optional: path to output .npz (overrides default HD-EPIC path).",
)
parser.add_argument(
    "--tracks_2d",
    type=str,
    default="",
    help="Optional: path to 2D tracks .npz file to filter alongside 3D tracks.",
)
parser.add_argument(
    "--tracks_2d_out",
    type=str,
    default="",
    help="Optional: path to save filtered 2D tracks .npz file.",
)
parser.add_argument("--group_size", type=int, default=100)
parser.add_argument("--n_keep", type=int, default=80)
parser.add_argument("--n_components", type=int, default=4)
args = parser.parse_args()

# ----------------------------------------------------------------------
print(f"[{now()}] Starting PCA filtering for vid={args.vid}")

X = np.load(args.src)['points_3d']

N = X.shape[0]
group_size = args.group_size
num_groups = N // group_size

mask = np.zeros(N, dtype=bool)
all_point_errors = np.zeros(N)

print(f"[{now()}] N={N}, num_groups={num_groups}, group_size={group_size}")
# --- loop ---
for g in range(num_groups):
    start = g * group_size
    end   = (g+1) * group_size
    
    Xg = X[start:end]

    good_idx, pe = select_points_with_pca(
        Xg,
        n_components=args.n_components,
        n_keep=args.n_keep
    )

    mask[start:end][good_idx] = True
    all_point_errors[start:end] = pe

points_3d_masked = X.copy()
points_3d_masked[~mask] = np.nan

# Save filtered 3D tracks
np.savez(args.dst, points_3d=points_3d_masked)
print(f"[{now()}] Saved filtered 3D tracks to {args.dst}")

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

    # Set filtered-out points to NaN
    tracks_2d_masked[~mask] = np.nan
    visibility_masked[~mask] = False

    # Save filtered 2D tracks (keep in N, T format for consistency with 3D)
    np.savez(args.tracks_2d_out,
             tracks=tracks_2d_masked,
             visibility=visibility_masked,
             dim=dim)
    print(f"[{now()}] Saved filtered 2D tracks to {args.tracks_2d_out}")
    print(f"[{now()}] Kept {mask.sum()}/{len(mask)} points ({100*mask.sum()/len(mask):.1f}%)")

print(f"[{now()}] Finished PCA filtering for vid={args.vid}")
