#!/usr/bin/env python3
"""
Extract undistorted RGB videos from HOT3D Aria clips.

For each clip in train_aria, extracts the RGB stream (214-1), undistorts
from fisheye to pinhole using camera calibration from the TAR, applies
rot90(k=3) to get upright orientation, and saves as MP4.

Requires clip_util from the HOT3D SDK (facebookresearch/hot3d) on PYTHONPATH:
    export PYTHONPATH=/path/to/hot3d-sdk/hot3d/clips:$PYTHONPATH

Usage:
    python hot3d/extract_rgbs.py \
        --clips_dir  /path/to/train_aria \
        --output_dir /path/to/rgbs

    # Large-scale: shard across N workers
    python hot3d/extract_rgbs.py \
        --clips_dir  /path/to/train_aria \
        --output_dir /path/to/rgbs \
        --shard_idx  0 --num_shards 8
"""

import argparse
import os
import tarfile
import time

import cv2
import imageio.v2 as imageio
import numpy as np

import clip_util


def compute_warp_maps(src_camera, dst_camera):
    """Compute (map_x, map_y) once for reuse with cv2.remap."""
    W, H = dst_camera.width, dst_camera.height
    px, py = np.meshgrid(np.arange(W), np.arange(H))
    dst_win_pts = np.column_stack((px.flatten(), py.flatten()))
    dst_eye_pts = dst_camera.window_to_eye(dst_win_pts)
    world_pts = dst_camera.eye_to_world(dst_eye_pts)
    src_eye_pts = src_camera.world_to_eye(world_pts)
    src_win_pts = src_camera.eye_to_window(src_eye_pts)
    mask = src_eye_pts[:, 2] < 0
    src_win_pts[mask] = -1
    src_win_pts = src_win_pts.astype(np.float32)
    map_x = src_win_pts[:, 0].reshape((H, W))
    map_y = src_win_pts[:, 1].reshape((H, W))
    return map_x, map_y


def extract_rgb(clip_path, output_dir, fps=30):
    """Extract undistorted RGB video from a single Aria clip tar."""
    clip_name = os.path.basename(clip_path).split(".tar")[0]
    out_path = os.path.join(output_dir, f"{clip_name}_rgb.mp4")

    if os.path.exists(out_path):
        return True

    tar = tarfile.open(clip_path, mode="r")
    stream_id = "214-1"

    num_frames = clip_util.get_number_of_frames(tar)

    # Load camera from first frame to compute warp maps
    cameras, _ = clip_util.load_cameras(tar, f"{0:06d}")
    camera_model = cameras[stream_id]
    camera_pinhole = clip_util.convert_to_pinhole_camera(camera_model)
    warp_map_x, warp_map_y = compute_warp_maps(camera_model, camera_pinhole)

    writer = imageio.get_writer(out_path, fps=fps, codec="libx264",
                                quality=8, pixelformat="yuv420p")

    for frame_id in range(num_frames):
        frame_key = f"{frame_id:06d}"
        image = clip_util.load_image(tar, frame_key, stream_id)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        # Undistort fisheye -> pinhole
        image = cv2.remap(image, warp_map_x, warp_map_y, cv2.INTER_LINEAR)
        # Rotate to upright (Aria RGB is rotated 90 degrees)
        image = np.ascontiguousarray(np.rot90(image, k=3))
        # Trim to even dimensions (H.264 requirement)
        h, w = image.shape[:2]
        if w % 2 != 0:
            image = image[:, :w - 1]
        if h % 2 != 0:
            image = image[:h - 1, :]
        writer.append_data(image)

    writer.close()
    tar.close()
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clips_dir",  required=True,
                        help="Directory containing train_aria TAR files")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for RGB MP4 files")
    parser.add_argument("--fps",        type=int, default=30)
    parser.add_argument("--shard_idx",  type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    clip_files = sorted([
        os.path.join(args.clips_dir, f)
        for f in os.listdir(args.clips_dir)
        if f.endswith(".tar")
    ])

    clip_files = clip_files[args.shard_idx::args.num_shards]
    print(f"Shard {args.shard_idx}/{args.num_shards}: {len(clip_files)} clips")

    for i, clip_path in enumerate(clip_files):
        clip_name = os.path.basename(clip_path).split(".tar")[0]
        t0 = time.time()
        try:
            extract_rgb(clip_path, args.output_dir, args.fps)
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(clip_files)}] {clip_name} done in {elapsed:.1f}s")
        except Exception as e:
            print(f"  [{i+1}/{len(clip_files)}] {clip_name} FAILED: {e}")


if __name__ == "__main__":
    main()
