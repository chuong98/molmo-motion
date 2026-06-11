"""Visualize a predicted 3D trajectory from MolmoMotion.

Saves a per-history-frame PNG with predicted points overlaid as a coloured
polyline per object. Either matplotlib (top-down + side views) or a plain
2D projection onto the t_0 image if `--mode=overlay`.

Usage:

    python scripts/visualize_trajectory.py \\
        --frames examples/data/egodex_grasp_cup \\
        --pred trajectory_prediction.pt \\
        --output viz_output.png \\
        --mode overlay

Inputs:
    --frames   Directory of history JPGs (frame_t-2.jpg, frame_t-1.jpg, frame_t+0.jpg)
               or a single JPG to overlay on.
    --pred     Path to a .pt file with the (P, F, 3) tensor returned by
               `MolmoMotion.predict_trajectory(...).future_3d`.
    --points-2d Optional .pt file with the (P, 2) pixel coords used as input.
    --intrinsics Optional .npz with K (3,3) — required for `overlay` mode.
                 If not given, an identity assumption is used (approximation).
    --output   Output PNG path. Defaults to `<pred-basename>.png`.
    --mode     `overlay` (project 3D → 2D over t_0 image) or `3d`
               (matplotlib 3D scatter). Default = overlay.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


def project_camera_xyz_to_pixel(xyz: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Project (..., 3) camera-frame XYZ to (..., 2) pixel coords using K."""
    eps = 1e-6
    z = np.clip(xyz[..., 2], eps, None)
    u = K[0, 0] * (xyz[..., 0] / z) + K[0, 2]
    v = K[1, 1] * (xyz[..., 1] / z) + K[1, 2]
    return np.stack([u, v], axis=-1)


def overlay_mode(args):
    """Project the predicted 3D trajectory onto the t_0 image plane."""
    frames_path = Path(args.frames)
    if frames_path.is_dir():
        candidates = list(frames_path.glob("frame_t+0.*")) + list(frames_path.glob("*t0*.*"))
        if not candidates:
            raise FileNotFoundError(f"Could not find a t_0 frame in {frames_path}")
        t0_path = candidates[0]
    else:
        t0_path = frames_path

    img = Image.open(t0_path).convert("RGB").copy()
    W, H = img.size

    pred = torch.load(args.pred, map_location="cpu").numpy()  # (P, F, 3)
    P, F, _ = pred.shape

    if args.intrinsics is not None:
        K = np.load(args.intrinsics)["K"].astype(np.float32)
    else:
        # Fallback identity-style K: principal point at image center, fx=fy=W.
        # Coarse — only acceptable for "rough preview" use.
        K = np.array([[W, 0, W / 2], [0, W, H / 2], [0, 0, 1]], dtype=np.float32)

    pixels = project_camera_xyz_to_pixel(pred, K)  # (P, F, 2)
    draw = ImageDraw.Draw(img)

    # Distinct colours per object.
    base_colours = [(255, 30, 30), (30, 200, 30), (40, 100, 255),
                    (255, 200, 0), (200, 0, 200), (0, 200, 200),
                    (180, 100, 0), (100, 100, 180)]
    if args.points_2d is not None:
        anchor_2d = torch.load(args.points_2d, map_location="cpu").numpy()
    else:
        anchor_2d = None

    for pi in range(P):
        col = base_colours[pi % len(base_colours)]
        pts = pixels[pi]
        if anchor_2d is not None:
            line = [tuple(anchor_2d[pi].tolist())] + [tuple(p) for p in pts.tolist()]
        else:
            line = [tuple(p) for p in pts.tolist()]
        if len(line) >= 2:
            draw.line(line, fill=col, width=2)
        for px, py in pts:
            if 0 <= px < W and 0 <= py < H:
                draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=col)

    out = args.output or (Path(args.pred).with_suffix(".png"))
    img.save(out)
    print(f"Wrote {out}")


def threed_mode(args):
    """3D scatter of the predicted trajectory using matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    pred = torch.load(args.pred, map_location="cpu").numpy()  # (P, F, 3)
    P, F, _ = pred.shape

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    for pi in range(P):
        xs, ys, zs = pred[pi].T
        ax.plot(xs, zs, -ys, label=f"point {pi+1}", alpha=0.8)
        ax.scatter(xs[-1], zs[-1], -ys[-1], s=30)
    ax.set_xlabel("X (right, m)")
    ax.set_ylabel("Z (forward, m)")
    ax.set_zlabel("-Y (up, m)")
    ax.legend(fontsize=7)
    ax.set_title(f"Predicted trajectory — {P} points × {F} frames")

    out = args.output or (Path(args.pred).with_suffix(".png"))
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frames", help="Directory of history JPGs or a single t_0 JPG.")
    parser.add_argument("--pred", required=True, help="Path to predicted (P, F, 3) tensor.")
    parser.add_argument("--points-2d", help="Optional (P, 2) pixel coords at t_0.")
    parser.add_argument("--intrinsics", help="Optional .npz with K (3,3).")
    parser.add_argument("--output", help="Output PNG path.")
    parser.add_argument("--mode", choices=["overlay", "3d"], default="overlay")
    args = parser.parse_args()

    if args.mode == "overlay":
        if args.frames is None:
            parser.error("--frames is required in overlay mode")
        overlay_mode(args)
    else:
        threed_mode(args)


if __name__ == "__main__":
    main()
