"""
Run MolmoMotion trajectory prediction from a bounding box + depth image.

Inputs: a single RGB image, a 2D bounding box, a metric depth image (.npz),
a grid size, camera intrinsics, and an action action.

  1. Samples a regular grid of 2D points inside the box.
  2. Back-projects each sampled point to 3D using the depth map and
     camera intrinsics.
  3. Builds a pseudo-history (H copies of the t_0 3D points) to satisfy
     the MolmoMotion input contract.
  4. Runs the model.
  5. Saves the predicted 3D trajectory (.npy) and a 2D curve overlay (.png).

Usage
-----
python scripts/infer.py \\
    --model checkpoints/MolmoMotion-4B-H1-F32 \\
    --image frame_t+0.jpg \\
    --bbox 200 150 400 350 \\
    --depth depth_frame.npz \\
    --calib calib.json --camera head_rgbd \\
    --grid 4 4 \\
    --action "Pick up the green ball" \\
    --output ./output
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from molmo_motion import MolmoMotion, MolmoMotionProcessor
from data_generation.depth_model.data_structure import read_calib_file, CameraIntrinsics
matplotlib.use("Agg")

# ===========================================================================
# Depth -> 3D back-projection
# ===========================================================================

def load_depth_npz(path: str) -> tuple[np.ndarray, float]:
    """Load a metric-depth array and scale_factor from an .npz file."""
    data = np.load(path)
    for key in ("metric_depth", "depth", "pred_depth"):
        if key in data:
            depth = data[key].astype(np.float32)
            break
    else:
        first = next(k for k in data.files if isinstance(data[k], np.ndarray))
        depth = data[first].astype(np.float32)

    scale_factor = float(data.get("scale_factor", 1.0))
    return depth, scale_factor


def sample_grid_in_bbox(
    bbox: tuple[int, int, int, int],
    grid: tuple[int, int],
) -> np.ndarray:
    """Sample a regular grid of 2D points inside a bounding box.

    Returns ``(cols * rows, 2)`` array of *(u, v)* pixel coordinates.
    """
    x1, y1, x2, y2 = bbox
    cols, rows = grid
    u = np.linspace(x1, x2, cols, dtype=np.float32)
    v = np.linspace(y1, y2, rows, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    return np.stack([uu.ravel(), vv.ravel()], axis=-1)


def select_closest_to_center(
    points_2d_px: np.ndarray,   # (P, 2) pixel coords
    depth_vals: np.ndarray,     # (P,) per-point depth (<=0 means invalid)
    bbox: tuple[int, int, int, int],
    k: int,
) -> np.ndarray:
    """Return indices of the ``k`` points closest to the bbox center.

    Points with valid depth are preferred over invalid ones (so the anchor,
    point 0, never lands on a point with no depth); within each group points
    are ordered by 2D distance to the box center.
    """
    x1, y1, x2, y2 = bbox
    center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)
    d2 = ((points_2d_px.astype(np.float32) - center) ** 2).sum(axis=1)
    valid = depth_vals > 0
    # lexsort's last key is primary: valid-first, then nearest-to-center.
    order = np.lexsort((d2, ~valid))
    return order[:k]

def load_example(example_dir: Path, history_size: int):
    """Load the bundled clip's frames, query points, history, intrinsics, action."""
    meta = json.loads((example_dir / "meta.json").read_text())
    history_frames = [
        Image.open(example_dir / f"frame_t{i:+d}.jpg").convert("RGB")
        for i in range(-(history_size - 1), 1)        # H=3 -> t-2, t-1, t+0
    ]
    points_2d_at_t0 = torch.load(example_dir / "points_2d_at_t0.pt")     # (P, 2)
    # Bundled tensor ships with 3 history frames; slice the last `history_size`
    # so the H=1 model gets just t_0 (shape (1, P, 3)) and the H=3 model gets
    # t-2..t_0 (shape (3, P, 3)). Same indexing as `history_frames` above.
    points_3d_history = torch.load(example_dir / "points_3d_history.pt")[-history_size:]
    intrinsics = torch.load(example_dir / "intrinsics_K.pt")             # (3, 3)
    action_file = example_dir / "action.txt"
    action = action_file.read_text().strip() if action_file.exists() else meta["caption"]
    return meta, history_frames, points_2d_at_t0, points_3d_history, intrinsics, action

# ===========================================================================
# Visualization helpers
# ===========================================================================

def render_trajectory_2d(
    future_3d: np.ndarray,          # (P, F, 3) absolute camera-frame XYZ (model output)
    image: Image.Image,
    cam_intr: CameraIntrinsics,     # camera intrinsics (calibration resolution)
    resize_w: int,                  # width of the frame points/overlay live in
    resize_h: int,                  # height of the frame points/overlay live in
    points_2d_px: np.ndarray,       # (P, 2) t_0 pixel coords (selected query points)
    output_path: str,
    # bbox: tuple[int, int, int, int] = None,      # (x1, y1, x2, y2) sampling box
    # sampled_points_2d: np.ndarray = None,        # (M, 2) all sampled grid points
    action: str = None,                         # action action, drawn as title
) -> str:
    """Draw a predicted 2D curve per point overlaid on the input image.

    ``future_3d`` is already absolute camera-frame XYZ (see modeling.py: the
    model emits anchor-relative deltas but predict_trajectory adds the anchor
    back), so it is projected directly. Each curve is drawn from the t_0 query
    pixel through the projected future points.
    """
    pred = np.asarray(future_3d).astype(np.float64)   # (P, F, 3) absolute XYZ
    img = np.asarray(image.convert("RGB"))

    P, F, _ = pred.shape
    px = cam_intr.forward_project(
        pred.reshape(-1, 3), resize_w, resize_h
    ).reshape(P, F, 2)                               # (P, F, 2)
    # Connect the t_0 query pixel through the predicted future pixels.
    full = np.concatenate([points_2d_px[:, None, :], px], axis=1)  # (P, F+1, 2)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(img)

    # Sampling bounding box.
    # if bbox is not None:
    #     x1, y1, x2, y2 = bbox
    #     ax.add_patch(plt.Rectangle(
    #         (x1, y1), x2 - x1, y2 - y1,
    #         fill=False, edgecolor="lime", linewidth=2.0, linestyle="--", zorder=4,
    #         label="sampling bbox",
    #     ))

    # # All sampled grid points (candidates), before closest-to-center selection.
    # if sampled_points_2d is not None:
    #     ax.scatter(sampled_points_2d[:, 0], sampled_points_2d[:, 1],
    #                c="cyan", s=18, marker="o", alpha=0.7,
    #                edgecolors="black", linewidths=0.4, zorder=4,
    #                label="sampled points")

    T = full.shape[1]                          # F + 1 timesteps (t_0 .. t_end)
    line_colors = plt.cm.magma(np.linspace(0.15, 0.88, P))
    tvals = np.arange(T)
    sc = None
    for p in range(P):
        # Per-point coloured connecting curve.
        ax.plot(full[p, :, 0], full[p, :, 1], color=line_colors[p], linewidth=2.0, alpha=0.7)
        # A marker at every timestep, coloured by time (t_0 -> t_end).
        sc = ax.scatter(full[p, :, 0], full[p, :, 1], c=tvals, cmap="viridis",
                        vmin=0, vmax=T - 1, s=16, marker="o",
                        edgecolors="white", linewidths=0.3, zorder=5)
        # Emphasise start (o) and end (X) with the per-point colour.
        ax.scatter(full[p, 0, 0], full[p, 0, 1], color=line_colors[p], s=55, marker="o",
                   edgecolors="white", linewidths=0.6, zorder=6)   # start
        ax.scatter(full[p, -1, 0], full[p, -1, 1], color=line_colors[p], s=90, marker="X",
                   edgecolors="white", linewidths=0.6, zorder=6)   # end

    if sc is not None:
        cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label("timestep (t_0 → t_end)", fontsize=9)
    # if bbox is not None or sampled_points_2d is not None:
    #     ax.legend(loc="upper right", fontsize=9, framealpha=0.7)
    if action:
        ax.set_title(action, fontsize=12, wrap=True)
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_trajectory_3d(
    future_3d: np.ndarray,          # (P, F, 3) absolute camera-frame XYZ
    start_xyz: np.ndarray,          # (P, 3) t_0 camera-frame XYZ
    output_path: str,
    action: str = None,            # action action, added to the title
) -> str:
    """3D plot of the predicted trajectories, one connected curve per point.

    ``future_3d`` is absolute camera-frame XYZ. Each curve runs from the t_0
    position through the predicted future points. Axes follow the camera-view
    convention used in visualize_trajectory.py: X right, Z forward, -Y up.
    """
    pred = np.asarray(future_3d).astype(np.float64)     # (P, F, 3)
    start = np.asarray(start_xyz).astype(np.float64)    # (P, 3)
    P, F, _ = pred.shape

    # Prepend the t_0 position so each curve starts at the query point.
    full = np.concatenate([start[:, None, :], pred], axis=1)  # (P, F+1, 3)

    T = full.shape[1]                          # F + 1 timesteps (t_0 .. t_end)
    tvals = np.arange(T)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    line_colors = plt.cm.magma(np.linspace(0.15, 0.88, P))
    sc = None
    for p in range(P):
        xs, ys, zs = full[p].T
        # X right, Z forward, -Y up (camera-view friendly).
        ax.plot(xs, zs, -ys, color=line_colors[p], linewidth=1.8, alpha=0.75)
        # A marker at every timestep, coloured by time (t_0 -> t_end).
        sc = ax.scatter(xs, zs, -ys, c=tvals, cmap="viridis", vmin=0, vmax=T - 1,
                        s=16, marker="o", edgecolors="white", linewidths=0.2,
                        depthshade=False)
        # Emphasise start (o) and end (X) with the per-point colour.
        ax.scatter(xs[0], zs[0], -ys[0], color=line_colors[p], s=55, marker="o",
                   edgecolors="white", linewidths=0.6,
                   label="t_0" if p == 0 else None)          # start
        ax.scatter(xs[-1], zs[-1], -ys[-1], color=line_colors[p], s=85, marker="X",
                   edgecolors="white", linewidths=0.6,
                   label="t_end" if p == 0 else None)         # end

    if sc is not None:
        cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.08)
        cbar.set_label("timestep (t_0 → t_end)", fontsize=9)
    ax.set_xlabel("X (right, m)")
    ax.set_ylabel("Z (forward, m)")
    ax.set_zlabel("-Y (up, m)")
    title = f"Predicted 3D Trajectory — {P} points × {F} frames"
    if action:
        title = f"{action}\n{title}"
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


# ===========================================================================
# Argument parser
# ===========================================================================

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="MolmoMotion inference — bbox + depth -> 3D trajectory prediction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ---- Model ----
    ap.add_argument("--model", default="allenai/MolmoMotion-4B-H1-F32", type=str,
                    help="Path to (or HF repo id of) the MolmoMotion checkpoint.")
    ap.add_argument("--future-horizon", type=int, default=32,
                    help="Number of future frames to predict (default: %(default)s).")

    # ---- Input image ----
    ap.add_argument("--example", default="davis_bmx_trees",
                    help="Sub-directory of examples/data/ to run on.")
    ap.add_argument("--image", required=True, type=str,
                    help="Path to the RGB image at t_0 (will be replicated "
                         "for history frames if the model requires H > 1).")

    # ---- Bounding box + depth -> 3D points ----
    ap.add_argument("--bbox", nargs=4, type=int, required=True,
                    metavar=("X1", "Y1", "X2", "Y2"),
                    help="2D bounding box in pixel coordinates (inclusive).")
    ap.add_argument("--grid", nargs=2, type=int, default=(4, 4),
                    metavar=("COLS", "ROWS"),
                    help="Grid size for sampling 2D points inside the bbox "
                         "(default: %(default)s).")
    ap.add_argument("--depth", required=True, type=str,
                    help="Path to a .npz file containing a metric-depth array "
                         "(keys: metric_depth / depth / pred_depth).")
    ap.add_argument("--calib", required=True, type=str,
                    help="Path to the camera calibration JSON file.")
    ap.add_argument("--camera", default="head_rgbd", type=str,
                    help="Camera name in the calib file (default: %(default)s).")
    ap.add_argument("--baseline-cams", default=None, type=str,
                    help="Stereo pair for baseline (auto-detected if omitted).")

    # ---- Action / action ----
    ap.add_argument("--action", required=True, type=str,
                    help="Language action description (e.g. 'Pick up the green ball').")

    # ---- Output ----
    ap.add_argument("--output", "-o", default="output_trajectory", type=str,
                    help="Output directory (default: %(default)s).")

    return ap


# ===========================================================================
# Main
# ===========================================================================

def main():
    args = build_parser().parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    # ------------------------------------------------------------------
    # 4. Load model & processor
    # ------------------------------------------------------------------
    print(f"\nLoading model from: {args.model}")
    processor = MolmoMotionProcessor.from_pretrained(args.model)
    model = MolmoMotion.from_pretrained(args.model)
    model._internal = model._internal.to(torch.bfloat16).cuda()
    H = processor.config.history_size
    print(f"History size: {H}")

    if args.example:
        EXAMPLES_DIR = Path("/home/chuong/workspace/point_models/molmo-motion/examples/data")
        example_dir = EXAMPLES_DIR / args.example
        (meta, history_frames, points_2d_tensor,
         points_3d_history, intrinsics, action) = load_example(example_dir, history_size=H)
        resolution =meta["image_size_wh"]
        cam_intr = CameraIntrinsics.from_intrinsics_matrix(intrinsics.numpy(), resolution=resolution)
        P = points_2d_tensor.shape[0]
        t0_image = history_frames[-1]
        points_2d_px = points_2d_tensor.numpy()
        start_xyz = points_3d_history[-1]
        depth_h, depth_w = resolution[1], resolution[0]
        print(f"Loaded example '{args.example}' with {P} points, "
              f"history size {H}, image size {resolution[0]}x{resolution[1]}.")
    else:
        # ------------------------------------------------------------------
        # 1. Load camera intrinsics
        # ------------------------------------------------------------------
        cam_intr = CameraIntrinsics.from_calib_file(
            args.calib, camera=args.camera, baseline_cams=args.baseline_cams,
        )
        print(f"Camera    : {args.camera}  res={cam_intr.w}x{cam_intr.h}")
        print(f"Intrinsics: fx={cam_intr.fx:.1f} fy={cam_intr.fy:.1f} "
            f"cx={cam_intr.cx:.1f} cy={cam_intr.cy:.1f}")

        # ------------------------------------------------------------------
        # 2. Load depth map
        # ------------------------------------------------------------------
        depth_map, scale_factor = load_depth_npz(args.depth)
        print(f"Depth map : {depth_map.shape}  "
            f"min={depth_map[depth_map > 0].min():.3f}m  "
            f"max={depth_map.max():.3f}m  "
            f"scale_factor={scale_factor:.4f}")

        # Depth-map / overlay frame size (differs from the calibration resolution
        # when the aspect ratio changes, e.g. 1280×720 → 640×480). Intrinsics are
        # scaled to this frame per-axis by back_project / forward_project.
        depth_h, depth_w = depth_map.shape[:2]

        # ------------------------------------------------------------------
        # 3. Sample grid points in bbox + back-project to 3D
        # ------------------------------------------------------------------
        bbox = tuple(args.bbox)
        grid = tuple(args.grid)

        points_2d_px = sample_grid_in_bbox(bbox, grid)                  # (P, 2) pixel coords
        points_2d_px_all = points_2d_px  # full grid, kept for visualization
        uv_scaled = points_2d_px * scale_factor
        u_s = np.clip(np.round(uv_scaled[:, 0]).astype(int), 0, depth_map.shape[1] - 1)
        v_s = np.clip(np.round(uv_scaled[:, 1]).astype(int), 0, depth_map.shape[0] - 1)
        depth_vals = depth_map[v_s, u_s]
        # uv_scaled and depth are in the depth-map frame (depth_w×depth_h), which
        # differs from the calibration resolution (cam_intr.w×h). back_project
        # scales the intrinsics to that frame per-axis.
        points_3d_at_t0 = cam_intr.back_project(uv_scaled, depth_vals, depth_w, depth_h)

        # Model expects normalized 2D coords in [0, 1], as fractions of the frame the
        # points live in (the depth-map frame), NOT the calibration resolution.
        points_2d_norm = uv_scaled / np.array([depth_w, depth_h], dtype=np.float32)

        P = points_2d_norm.shape[0]
        print(f"BBox      : {bbox}")
        print(f"Grid      : {grid}  -> {P} points")
        print(f"3D range  : "
            f"X [{points_3d_at_t0[:, 0].min():.3f}, {points_3d_at_t0[:, 0].max():.3f}]  "
            f"Y [{points_3d_at_t0[:, 1].min():.3f}, {points_3d_at_t0[:, 1].max():.3f}]  "
            f"Z [{points_3d_at_t0[:, 2].min():.3f}, {points_3d_at_t0[:, 2].max():.3f}]")


        # The model handles a fixed number of query points (config.num_points). If we
        # sampled more, keep the ones closest to the bbox center (point 0 = anchor).
        target_p = processor.config.num_points
        if P > target_p:
            sel = select_closest_to_center(points_2d_px, depth_vals, bbox, target_p)
            points_2d_px = points_2d_px[sel]
            points_3d_at_t0 = points_3d_at_t0[sel]
            points_2d_norm = points_2d_norm[sel]
            P = target_p
            print(f"Points    : sampled {grid[0] * grid[1]}, kept {P} closest to bbox center")
        elif P < target_p:
            raise ValueError(
                f"Model needs {target_p} points but only {P} were sampled; "
                f"increase --grid (e.g. so cols*rows >= {target_p})."
            )

        # ------------------------------------------------------------------
        # 5. Load image + replicate for history
        # ------------------------------------------------------------------
        t0_image = Image.open(args.image).convert("RGB")
        history_frames = [t0_image] * H
        print(f"Image     : {args.image}  ({t0_image.size[0]}x{t0_image.size[1]})")

        # ------------------------------------------------------------------
        # 6. Build points_3d_history
        # ------------------------------------------------------------------
        points_3d_history_np = np.tile(points_3d_at_t0[None, ...], (H, 1, 1))   # (H, P, 3)
        points_3d_history = torch.from_numpy(points_3d_history_np)
        points_2d_tensor = torch.from_numpy(points_2d_norm)
        start_xyz = points_3d_at_t0

    action = args.action
    # ------------------------------------------------------------------
    # 7. Run inference
    # ------------------------------------------------------------------
    print(f"action   : {action}")

    inputs = processor(
        history_frames=history_frames,
        points_2d_at_t0=points_2d_tensor,
        points_3d_history=points_3d_history,
        action=action,
        future_horizon=args.future_horizon,
    )
    inputs = {k: v.cuda() if torch.is_tensor(v) else v for k, v in inputs.items()}

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.predict_trajectory(**inputs)

    future_3d = out.future_3d                                   # (P, F, 3)
    print(f"Predicted future_3d {tuple(future_3d.shape)}  "
          f"(expect ({P}, {args.future_horizon}, 3))")

    # ------------------------------------------------------------------
    # 8. Save outputs
    # ------------------------------------------------------------------
    future_3d_np = future_3d.detach().cpu().numpy().astype(np.float32)

    # 8a. 3D trajectory as .npy
    npy_path = out_dir / "trajectory_3d.npy"
    np.save(npy_path, future_3d_np)
    print(f"Wrote {npy_path}")

    # Also save start points for reference
    # np.save(out_dir / "start_xyz.npy", start_xyz.astype(np.float32))
    # np.save(out_dir / "points_2d_px.npy", points_2d_px.astype(np.float32))

    # 8b. 2D curve overlay on the input image
    png_2d_path = out_dir / "trajectory_2d.png"
    render_trajectory_2d(
        future_3d_np,
        image=t0_image,
        cam_intr=cam_intr,
        resize_w=depth_w,
        resize_h=depth_h,
        points_2d_px=points_2d_px,
        output_path=str(png_2d_path),
        # bbox=bbox,
        # sampled_points_2d=points_2d_px_all,
        action=args.action,
    )
    print(f"Wrote {png_2d_path}")

    # 8c. Static 3D plot
    png_3d_path = out_dir / "trajectory_3d.png"
    render_trajectory_3d(
        future_3d_np,
        start_xyz=start_xyz,
        output_path=str(png_3d_path),
        action=args.action,
    )
    print(f"Wrote {png_3d_path}")


if __name__ == "__main__":
    main()
