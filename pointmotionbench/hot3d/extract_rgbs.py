#!/usr/bin/env python3
"""
Extract undistorted RGB videos from HOT3D Aria clips.

For each clip in train_aria, extracts the RGB stream (214-1), undistorts
from fisheye to pinhole using camera calibration from the TAR, applies
rot90(k=3) to get upright orientation, and saves as MP4.

Requirements: imageio[ffmpeg], imageio-ffmpeg, opencv-python-headless, numpy

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
import json
import os
import tarfile
import time

import cv2
import imageio.v2 as imageio
import numpy as np


def get_number_of_frames(tar):
    max_frame_id = -1
    for x in tar.getnames():
        if x.endswith(".info.json"):
            frame_id = int(x.split(".info.json")[0])
            if frame_id > max_frame_id:
                max_frame_id = frame_id
    return max_frame_id + 1


def load_image(tar, frame_key, stream_key, dtype=np.uint8):
    file = tar.extractfile(f"{frame_key}.image_{stream_key}.jpg")
    return imageio.imread(file).astype(dtype)


def load_fisheye_params(tar, frame_key, stream_id):
    """Return (projection_params, width, height) for the given stream."""
    cameras_raw = json.load(tar.extractfile(f"{frame_key}.cameras.json"))
    cal = cameras_raw[stream_id]["calibration"]
    return cal["projection_params"], cal["image_width"], cal["image_height"]


def _fisheye624_project(params, X, Y, Z):
    """Project 3D directions using the Kannala-Brandt fisheye model.

    FISHEYE624 uses a single focal length and a 6-term theta polynomial.
    The tangential/thin-prism terms (params[9:15]) are all <0.001 for the
    Aria RGB camera and are omitted — error is sub-pixel, imperceptible in video.

    Reference: Kannala & Brandt, IEEE TPAMI 2006.
    """
    f, cx, cy = params[0], params[1], params[2]
    k = params[3:9]
    r = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(r, Z)
    t2 = theta**2
    theta_d = theta * (1 + k[0]*t2 + k[1]*t2**2 + k[2]*t2**3
                       + k[3]*t2**4 + k[4]*t2**5 + k[5]*t2**6)
    with np.errstate(divide='ignore', invalid='ignore'):
        mx = np.where(r > 1e-9, X / r * theta_d, 0.0)
        my = np.where(r > 1e-9, Y / r * theta_d, 0.0)
    return f * mx + cx, f * my + cy


def compute_warp_maps(fisheye_params, W, H):
    """Compute cv2.remap maps to undistort fisheye to pinhole.

    The undistorted pinhole shares f, cx, cy with the fisheye and the same
    extrinsics, so the warp reduces to: for each output pixel, unproject
    through pinhole then project through the fisheye model.
    """
    f, cx, cy = fisheye_params[0], fisheye_params[1], fisheye_params[2]
    px, py = np.meshgrid(np.arange(W, dtype=np.float64),
                         np.arange(H, dtype=np.float64))
    X = (px - cx) / f
    Y = (py - cy) / f
    Z = np.ones_like(X)
    map_x, map_y = _fisheye624_project(fisheye_params, X, Y, Z)
    return map_x.astype(np.float32), map_y.astype(np.float32)


def extract_rgb(clip_path, output_dir, fps=30):
    """Extract undistorted RGB video from a single Aria clip tar."""
    clip_name = os.path.basename(clip_path).split(".tar")[0]
    out_path = os.path.join(output_dir, f"{clip_name}_rgb.mp4")

    if os.path.exists(out_path):
        return True

    tar = tarfile.open(clip_path, mode="r")
    stream_id = "214-1"

    num_frames = get_number_of_frames(tar)
    fisheye_params, W, H = load_fisheye_params(tar, f"{0:06d}", stream_id)
    warp_map_x, warp_map_y = compute_warp_maps(fisheye_params, W, H)

    writer = imageio.get_writer(out_path, fps=fps, codec="libx264",
                                quality=8, pixelformat="yuv420p")

    for frame_id in range(num_frames):
        frame_key = f"{frame_id:06d}"
        image = load_image(tar, frame_key, stream_id)
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
