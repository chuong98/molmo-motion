"""MolmoMotion quickstart: predict a 3D point trajectory and render it as an MP4.

Single end-to-end script:

  1. Load the model + processor.
  2. Build one inference example from a bundled clip (history RGB frames, the
     query points' 2D pixel coords at t_0, their 3D camera-frame history, and an
     action caption).
  3. ``out = model.predict_trajectory(**inputs)`` -> ``out.future_3d`` is a
     ``(P, F, 3)`` tensor of absolute camera-frame XYZ (meters).
  4. ``render_trajectory_mp4(out.future_3d, ...)`` projects that trajectory back
     onto the static t_0 frame and animates it as a growing 2D track, coloured
     with the ``magma`` gradient (one polyline per point, a bright dot at the
     moving end). The visualization is taken *directly* from ``out`` -- no
     intermediate files.

Run (needs a GPU for the 4B model)::

    pip install -e ".[viz]"
    python examples/01_quickstart.py

No GPU? Render the MP4 from the bundled released-model prediction instead --
this exercises the exact same ``render_trajectory_mp4`` path on an identical
``(P, F, 3)`` array::

    python examples/01_quickstart.py --from-prediction

Output: ``<example>_2d.mp4`` next to this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

EXAMPLES_DIR = Path(__file__).parent / "data"


# ──────────────────────────────────────────────────────────────────────────
# Visualization: project (P, F, 3) -> image plane and animate a magma 2D track.
# Depends only on the ``[viz]`` extra (matplotlib + imageio[ffmpeg]).
# ──────────────────────────────────────────────────────────────────────────

def project_camera_xyz_to_pixel(xyz: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Pinhole-project (..., 3) camera-frame XYZ (meters) to (..., 2) pixels."""
    z = np.clip(xyz[..., 2], 1e-6, None)
    u = K[0, 0] * (xyz[..., 0] / z) + K[0, 2]
    v = K[1, 1] * (xyz[..., 1] / z) + K[1, 2]
    return np.stack([u, v], axis=-1)


def render_trajectory_mp4(
    future_3d,
    *,
    t0_image: Image.Image,
    intrinsics,
    points_2d_at_t0,
    output_path: str,
    fps: int = 30,
    seconds: float = 3.6,
    cmap_name: str = "magma",
    gradient_floor: float = 0.15,
    gradient_top: float = 0.88,
    line_width: float = 2.4,
    dot_size: float = 36.0,
    pad: bool = False,
    pad_margin: float = 0.05,
) -> str:
    """Animate a predicted 3D trajectory as a 2D track over the static t_0 frame.

    Each of the ``P`` points gets one polyline that grows from its t_0 query
    pixel along the projected future path. The polyline is vertex-coloured by
    cumulative arc length through a colormap sub-range (oldest = dark, newest =
    bright), with a filled dot at the moving end.

    Args:
        future_3d: ``(P, F, 3)`` camera-frame XYZ in meters -- exactly
            ``model.predict_trajectory(...).future_3d``. Tensor or ndarray.
        t0_image: the t_0 RGB frame the track is drawn over (kept static).
        intrinsics: ``(3, 3)`` camera matrix ``[[fx,0,cx],[0,fy,cy],[0,0,1]]``.
        points_2d_at_t0: ``(P, 2)`` query-point pixel coords at t_0 -- the
            anchor each polyline grows out of.
        output_path: where to write the ``.mp4``.
        fps, seconds: frame rate and total duration of the reveal.
        cmap_name: matplotlib colormap (default ``magma``).
        gradient_floor, gradient_top: map arc length into this colormap
            sub-range so the oldest end is not pure black and the newest end is
            not washed out.
        line_width, dot_size: trail thickness and moving-dot area (pt^2).
        pad: if True, extend the canvas (filled black) so trail portions that
            project outside the image stay visible (e.g. the bmx clip rides off
            the right edge). If False, the canvas is the image and off-frame
            track is clipped.
        pad_margin: extra border around the trajectory when ``pad`` is on,
            as a fraction of the image's larger side.

    Returns:
        ``output_path``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio
    from matplotlib.collections import LineCollection

    def _np(x):
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

    pred = _np(future_3d).astype(np.float64)          # (P, F, 3)
    K = _np(intrinsics).astype(np.float64)            # (3, 3)
    anchor_2d = _np(points_2d_at_t0).astype(np.float64)  # (P, 2)
    img = np.asarray(t0_image.convert("RGB"))
    H, W = img.shape[:2]

    px = project_camera_xyz_to_pixel(pred, K)         # (P, F, 2)
    # Prepend the t_0 query pixel so each trail starts on the tracked point.
    full = np.concatenate([anchor_2d[:, None, :], px], axis=1)  # (P, T, 2)
    P, T, _ = full.shape

    # Per-point gradient parametrised by cumulative arc length (stable as the
    # trail grows), remapped into [floor, top] of the colormap.
    seg_len = np.linalg.norm(np.diff(full, axis=1), axis=2)        # (P, T-1)
    arclen = np.concatenate([np.zeros((P, 1)), np.cumsum(seg_len, axis=1)], axis=1)
    arcnorm = arclen / np.clip(arclen[:, -1:], 1e-6, None)         # (P, T) in [0,1]
    cvals = gradient_floor + (gradient_top - gradient_floor) * arcnorm

    cmap = matplotlib.colormaps[cmap_name]
    n_frames = max(2, int(round(fps * seconds)))
    dpi = 100

    # Canvas extent: the image by default, or grown to fit the whole trajectory
    # (plus a margin) when padding is requested. Rounded to even pixels for h264.
    if pad:
        m = pad_margin * max(W, H)
        x0 = min(0.0, float(full[..., 0].min()) - m)
        x1 = max(float(W), float(full[..., 0].max()) + m)
        y0 = min(0.0, float(full[..., 1].min()) - m)
        y1 = max(float(H), float(full[..., 1].max()) + m)
    else:
        x0, x1, y0, y1 = 0.0, float(W), 0.0, float(H)
    out_w = int(round(x1 - x0)) // 2 * 2
    out_h = int(round(y1 - y0)) // 2 * 2

    frames = []
    for k in range(n_frames):
        head = (k / (n_frames - 1)) * (T - 1)         # fractional playhead in [0, T-1]
        ni = int(np.floor(head))
        fig = plt.figure(figsize=((x1 - x0) / dpi, (y1 - y0) / dpi), dpi=dpi,
                         facecolor="black")
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        ax.set_facecolor("black")
        ax.imshow(img, extent=[0, W, H, 0])            # image at its native pixels
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)                            # image coords: y grows down

        for p in range(P):
            if head < 1.0:                             # before the first step: just the dot
                tip = full[p, 0] + (full[p, 1] - full[p, 0]) * head
            else:
                pts = full[p, : ni + 1]
                nxt = min(ni + 1, T - 1)
                tip = full[p, ni] + (full[p, nxt] - full[p, ni]) * (head - ni)
                line_pts = np.concatenate([pts, tip[None, :]], axis=0)
                line_c = np.concatenate([cvals[p, : ni + 1], [cvals[p, nxt]]])
                segs = np.stack([line_pts[:-1], line_pts[1:]], axis=1)
                lc = LineCollection(segs, cmap=cmap, norm=plt.Normalize(0, 1), zorder=4)
                lc.set_array(0.5 * (line_c[:-1] + line_c[1:]))
                lc.set_linewidth(line_width)
                lc.set_capstyle("round")
                ax.add_collection(lc)
            ax.scatter([tip[0]], [tip[1]], s=dot_size, color=cmap(0.95),
                       zorder=6, edgecolors="white", linewidths=0.4)

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        # Force exact even dimensions so yuv420p / h264 is happy.
        frame = np.asarray(Image.fromarray(buf).resize((out_w, out_h)))
        frames.append(frame)
        plt.close(fig)

    imageio.mimsave(output_path, frames, fps=fps, codec="libx264",
                    macro_block_size=1, ffmpeg_params=["-pix_fmt", "yuv420p"])
    return output_path


# ──────────────────────────────────────────────────────────────────────────
# Inputs.
# ──────────────────────────────────────────────────────────────────────────

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
    caption_file = example_dir / "caption.txt"
    action = caption_file.read_text().strip() if caption_file.exists() else meta["action"]
    return meta, history_frames, points_2d_at_t0, points_3d_history, intrinsics, action


def prediction_from_jsonl(jsonl_path: Path, video: str) -> torch.Tensor:
    """Load the released-model ``(P, F, 3)`` prediction for ``video`` from the
    eval JSONL (``pred_raw_combined``) -- identical in shape/units to
    ``out.future_3d``."""
    for line in jsonl_path.read_text().splitlines():
        row = json.loads(line)
        if row["video"] == video:
            return torch.tensor(row["pred_raw_combined"], dtype=torch.float32)
    raise SystemExit(f"No prediction for video={video!r} in {jsonl_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--example", default="davis_bmx_trees",
                    help="Sub-directory of examples/data/ to run on.")
    ap.add_argument("--model", default="allenai/MolmoMotion-4B-H3-F30")
    ap.add_argument("--future-horizon", type=int, default=30)
    ap.add_argument("--output", default=None, help="Output MP4 path.")
    ap.add_argument("--pad", action="store_true",
                    help="Extend the canvas (black border) so track that "
                         "projects off the frame stays visible.")
    ap.add_argument("--from-prediction", action="store_true",
                    help="Skip the model (no GPU needed); use the bundled "
                         "released-model prediction from predictions_h3.jsonl.")
    args = ap.parse_args()

    example_dir = EXAMPLES_DIR / args.example
    output = args.output or f"{args.example}_2d.mp4"

    if args.from_prediction:
        # No model: pull the shipped (P, F, 3) prediction. Same array the live
        # model would hand back as out.future_3d.
        meta = json.loads((example_dir / "meta.json").read_text())
        future_3d = prediction_from_jsonl(EXAMPLES_DIR / "predictions_h3.jsonl", meta["video"])
        points_2d_at_t0 = torch.load(example_dir / "points_2d_at_t0.pt")
        intrinsics = torch.load(example_dir / "intrinsics_K.pt")
        t0_image = Image.open(example_dir / "frame_t+0.jpg")
        print(f"Loaded bundled prediction: future_3d {tuple(future_3d.shape)}")
    else:
        from molmo_motion import MolmoMotion, MolmoMotionProcessor

        processor = MolmoMotionProcessor.from_pretrained(args.model)
        model = MolmoMotion.from_pretrained(args.model)
        model._internal = model._internal.to(torch.bfloat16).cuda()  # 4B params
        H = processor.config.history_size

        (meta, history_frames, points_2d_at_t0, points_3d_history,
         intrinsics, action) = load_example(example_dir, H)
        t0_image = history_frames[-1]

        inputs = processor(
            history_frames=history_frames,
            points_2d_at_t0=points_2d_at_t0,
            points_3d_history=points_3d_history,
            action=action,
            future_horizon=args.future_horizon,
        )
        inputs = {k: v.cuda() if torch.is_tensor(v) else v for k, v in inputs.items()}

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.predict_trajectory(**inputs)

        future_3d = out.future_3d                       # (P, F, 3), camera-frame meters
        print(f"Predicted future_3d {tuple(future_3d.shape)}  (expect (8, {args.future_horizon}, 3))")

    # Visualize straight from the prediction tensor.
    render_trajectory_mp4(
        future_3d,
        t0_image=t0_image,
        intrinsics=intrinsics,
        points_2d_at_t0=points_2d_at_t0,
        output_path=output,
        pad=args.pad,
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
