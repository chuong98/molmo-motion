#!/usr/bin/env python3

import argparse
import logging

from pathlib import Path
from typing import Tuple

import cv2
import imageio
import numpy as np
import torch

from scipy.spatial.transform import Rotation

from vipe.slam.interface import SLAMMap
from vipe.utils.cameras import CameraType
from vipe.utils.depth import reliable_depth_mask_range
from vipe.utils.io import (
    ArtifactPath,
    read_depth_artifacts,
    read_intrinsics_artifacts,
    read_pose_artifacts,
    read_rgb_artifacts,
)


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def quaternion_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to quaternion (w, x, y, z)."""
    rotation = Rotation.from_matrix(matrix[:3, :3])
    quat_xyzw = rotation.as_quat()  # Returns [x, y, z, w]
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])  # Convert to [w, x, y, z]


def matrix_to_colmap_pose(c2w_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert camera-to-world matrix to COLMAP format.
    COLMAP uses world-to-camera transformation.
    """
    w2c = np.linalg.inv(c2w_matrix)
    quaternion = quaternion_from_matrix(w2c)
    translation = w2c[:3, 3]
    return quaternion, translation


def write_cameras_txt(output_dir: Path, artifact: ArtifactPath, frame_width: int, frame_height: int):
    """Write COLMAP cameras.txt file."""
    cameras_file = output_dir / "cameras.txt"

    _, intrinsics, camera_types = read_intrinsics_artifacts(artifact.intrinsics_path)

    # Use first frame's intrinsics (assuming constant intrinsics)
    assert camera_types[0] == CameraType.PINHOLE, "Only PINHOLE camera type is supported"
    fx, fy, cx, cy = intrinsics[0].cpu().numpy()

    with open(cameras_file, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")

        fx, fy, cx, cy = intrinsics[0]

        # COLMAP camera format: CAMERA_ID MODEL WIDTH HEIGHT fx fy cx cy
        f.write(f"1 PINHOLE {frame_width} {frame_height} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")

    logger.info(f"Written cameras.txt with intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")


def write_images_txt(output_dir: Path, artifact: ArtifactPath):
    """Write COLMAP images.txt file."""
    images_file = output_dir / "images.txt"

    # Load pose data
    pose_data = np.load(artifact.pose_path)
    poses = pose_data["data"]  # Shape: (N, 4, 4)
    indices = pose_data["inds"]  # Frame indices

    with open(images_file, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(poses)}\n")

        for i, (pose_matrix, frame_idx) in enumerate(zip(poses, indices)):
            # Convert pose to COLMAP format
            quaternion, translation = matrix_to_colmap_pose(pose_matrix)
            qw, qx, qy, qz = quaternion
            tx, ty, tz = translation

            # Image filename
            image_name = f"images/frame_{frame_idx:06d}.jpg"

            # Write image line
            f.write(f"{i + 1} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {tx:.9f} {ty:.9f} {tz:.9f} 1 {image_name}\n")
            # Empty points2D line (no 2D-3D correspondences)
            f.write("\n")

    logger.info(f"Written images.txt with {len(poses)} images")


def write_points3d_txt_from_slam_map(output_dir: Path, artifact: ArtifactPath):
    """Write points3D.txt from SLAM map (placeholder implementation)."""
    assert artifact.slam_map_path.exists(), "SLAM map not found, please refer to README.md for more details."

    slam_map = SLAMMap.load(artifact.slam_map_path, device=torch.device("cpu"))

    points3d_file = output_dir / "points3D.txt"
    with open(points3d_file, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {slam_map.dense_disp_xyz.shape[0]}\n")

        point_id = 1
        for keyframe_idx, frame_idx in enumerate(slam_map.dense_disp_frame_inds):
            xyz, rgb = slam_map.get_dense_disp_pcd(keyframe_idx)
            xyz = xyz.cpu().numpy()
            rgb = rgb.cpu().numpy()

            for xyz, rgb in zip(xyz, rgb):
                x, y, z = xyz
                r, g, b = (rgb * 255).astype(np.uint8)
                f.write(f"{point_id} {x:.6f} {y:.6f} {z:.6f} {r} {g} {b} 0.0 {frame_idx} {point_id} 0 0 0 0\n")
                point_id += 1


def write_points3d_txt_from_depth(
    output_dir: Path,
    artifact: ArtifactPath,
    depth_step: int,
    args,
    spatial_subsample: int = 1,
):
    """
    Lift tracked 2D points into 3D using ViPE depth and poses,
    and save as an .npz file with NxTx3 coordinates and NxTx1 visibility.
    """
    import numpy as np

    # === Load camera and pose info ===
    _, pose_data = read_pose_artifacts(artifact.pose_path)
    _, intrinsics, camera_types = read_intrinsics_artifacts(artifact.intrinsics_path)
    camera_type = camera_types[0]

    def _load_tracks_from_path(track_path_str: str):
        """
        Load 2D tracks either from a single .npz file OR from a directory of .npz files.

        Expected per-file keys:
          - tracks: [T, N, 2]
          - visibility: [T, N]
          - dim: [2] -> (h, w)

        If a directory is provided, concatenate all tracks/visibility along N (axis=1).
        """
        if track_path_str is None:
            return None

        p = Path(track_path_str)
        if not p.exists():
            return None

        def _load_one(npz_path: Path):
            d = np.load(npz_path)
            t = d["tracks"]
            v = d["visibility"]
            dim = d["dim"]
            h0, w0 = int(dim[0]), int(dim[1])
            if t.ndim != 3 or t.shape[2] != 2:
                raise ValueError(f"Bad tracks shape in {npz_path}: expected [T,N,2], got {t.shape}")
            if v.ndim != 2:
                raise ValueError(f"Bad visibility shape in {npz_path}: expected [T,N], got {v.shape}")
            if t.shape[0] != v.shape[0] or t.shape[1] != v.shape[1]:
                raise ValueError(
                    f"Mismatched tracks/visibility in {npz_path}: tracks={t.shape}, visibility={v.shape}"
                )
            return t, v, (h0, w0)

        if p.is_dir():
            npz_files = sorted([x for x in p.iterdir() if x.is_file() and x.suffix == ".npz"])
            if len(npz_files) == 0:
                raise FileNotFoundError(f"No .npz files found in track_path directory: {p}")

            tracks_list = []
            vis_list = []
            ref_T = None
            ref_dim = None
            total_N = 0
            for npz in npz_files:
                t, v, dim = _load_one(npz)
                T, N, _ = t.shape
                if ref_T is None:
                    ref_T = T
                    ref_dim = dim
                else:
                    if T != ref_T:
                        raise ValueError(f"T mismatch in {npz}: got T={T}, expected T={ref_T}")
                    if dim != ref_dim:
                        raise ValueError(f"dim mismatch in {npz}: got dim={dim}, expected dim={ref_dim}")
                tracks_list.append(t)
                vis_list.append(v)
                total_N += N
            logger.info(f"Loaded {len(npz_files)} track files from {p}; concatenating to N={total_N}")
            tracks_cat = np.concatenate(tracks_list, axis=1)
            vis_cat = np.concatenate(vis_list, axis=1)
            return tracks_cat, vis_cat, ref_dim

        # Single file path
        return _load_one(p)

    # === Load your 2D track data ===
    # track_path can be a single .npz OR a directory of .npz files (concatenated along N).
    loaded = _load_tracks_from_path(args.track_path)
    if loaded is None:
        raise FileNotFoundError(f"Failed to load 2D tracks from: {args.track_path}")
    tracks, visibility_2d, (h, w) = loaded

    T, N, _ = tracks.shape
    logger.info(f"Loaded track data for {args.sequence}: {T} frames, {N} points.")

    # === Allocate output arrays ===
    points_3d = np.full((N, T, 3), np.nan, dtype=np.float32)
    visibility = np.zeros((N, T, 1), dtype=bool)

    # Detect depth resolution on first frame so we can scale tracks if needed
    # (e.g. tracks are in original 1280x720 space but ViPE ran on a 480p video)
    _depth_iter = read_depth_artifacts(artifact.depth_path)
    _first_idx, _first_depth = next(_depth_iter)
    _depth_h, _depth_w = _first_depth.shape[-2], _first_depth.shape[-1]
    _track_to_depth_sx = _depth_w / w  # scale tracks x → depth x
    _track_to_depth_sy = _depth_h / h  # scale tracks y → depth y
    if abs(_track_to_depth_sx - 1.0) > 0.01 or abs(_track_to_depth_sy - 1.0) > 0.01:
        logger.info(
            f"Track dim ({h}x{w}) differs from depth dim ({_depth_h}x{_depth_w}); "
            f"scaling tracks by sx={_track_to_depth_sx:.4f}, sy={_track_to_depth_sy:.4f}"
        )

    def _depth_frames():
        yield _first_idx, _first_depth
        yield from _depth_iter

    # === Iterate through depth frames ===
    for idx, (_, depth) in enumerate(_depth_frames()):
        if idx >= T:
            break
        if idx % depth_step != 0:
            continue

        valid_mask = visibility_2d[idx] > 0
        if not np.any(valid_mask):
            continue

        xy = tracks[idx, valid_mask]  # (M, 2) in AllTracker coordinate space
        # Scale to depth coordinate space (no-op when resolutions match)
        xy_scaled = xy * np.array([_track_to_depth_sx, _track_to_depth_sy])
        x_coords = np.clip(np.round(xy_scaled[:, 0]).astype(int), 0, _depth_w - 1)
        y_coords = np.clip(np.round(xy_scaled[:, 1]).astype(int), 0, _depth_h - 1)

        depth_np = depth.numpy()
        depth_vals = depth_np[y_coords, x_coords]
        valid_depth = np.isfinite(depth_vals) & (depth_vals > 0)

        if not np.any(valid_depth):
            continue

        # === Backproject to camera coordinates ===
        fx, fy, cx, cy = intrinsics[idx].cpu().numpy()
        X = (x_coords[valid_depth] - cx) * depth_vals[valid_depth] / fx
        Y = (y_coords[valid_depth] - cy) * depth_vals[valid_depth] / fy
        Z = depth_vals[valid_depth]
        pcd = np.stack([X, Y, Z], axis=1)

        # === Transform to world coordinates ===
        c2w = pose_data[idx].matrix().numpy()
        pcd_world = pcd @ c2w[:3, :3].T + c2w[:3, 3][None]

        # === Store results ===
        visible_indices = np.where(valid_mask)[0][valid_depth]
        points_3d[visible_indices, idx, :] = pcd_world
        visibility[visible_indices, idx, 0] = True

        if idx % 10 == 0:
            logger.info(f"Processed frame {idx}: {len(visible_indices)} 3D track points")

    # === Save results ===
    save_path = output_dir / f"{args.sequence}_3d_tracks.npz"
    np.savez_compressed(save_path, points_3d=points_3d, visibility=visibility)

    logger.info(f"Saved 3D tracks to {save_path}")
    logger.info(f"points_3d shape: {points_3d.shape}, visibility shape: {visibility.shape}")



def extract_frames(artifact: ArtifactPath, output_dir: Path) -> Tuple[int, int]:
    """Extract frames from video to individual image files."""
    video_path = artifact.rgb_path
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    logger.info(f"Extracting frames from {video_path}")

    for frame_idx, rgb in read_rgb_artifacts(video_path):
        frame_path = images_dir / f"frame_{frame_idx:06d}.jpg"
        frame_height, frame_width = rgb.shape[:2]
        imageio.imwrite(str(frame_path), (rgb.cpu().numpy() * 255).astype(np.uint8))
        if frame_idx % 30 == 0:
            logger.info(f"Extracted {frame_idx} frames")

    logger.info(f"Extracted {frame_idx} frames to {images_dir}")

    return frame_width, frame_height


def convert_vipe_to_colmap(artifact: ArtifactPath, output_path: Path, depth_step: int, use_slam_map: bool, args):
    """Convert ViPE reconstruction results to COLMAP format."""

    logger.info(
        f"Converting ViPE results from {artifact.base_path} ({artifact.artifact_name}) to COLMAP format at {output_path}"
    )

    # Verify required files exist
    required_files = [artifact.rgb_path, artifact.pose_path, artifact.intrinsics_path, artifact.depth_path]
    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(f"Required file not found: {file_path}")

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Extract frames and get video dimensions

    
    write_points3d_txt_from_depth(output_path, artifact, depth_step, args)

    logger.info("COLMAP conversion completed successfully!")
    logger.info(f"Output directory: {output_path}")



def main():
    """Main function for ViPE to COLMAP conversion script."""
    parser = argparse.ArgumentParser(
        description="Convert ViPE reconstruction results to COLMAP format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("vipe_path", type=Path, help="Path to ViPE results directory")
    parser.add_argument("--track_path", type=str, help="Path to track data", default=None)
    parser.add_argument(
        "--sequence",
        "-s",
        type=str,
        help="Sequence name (if not provided will convert all sequences in the directory)",
        default=None,
    )
    parser.add_argument(
        "--video_stem",
        type=str,
        help="Video stem to match artifact name (e.g., '0' for cached video '0.mp4'). If not provided, extracts from sequence.",
        default=None,
    )
    parser.add_argument("--use_slam_map", action="store_true", help="Use SLAM map to unproject depth maps")
    parser.add_argument("--depth_step", type=int, default=1, help="Step size for depth extraction")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Base output directory for COLMAP format (default: <vipe_path>_colmap)",
    )
    parser.add_argument(
        "--dataset_tag",
        type=str,
        default="ssv2",
        help="Subdirectory under output for this dataset/split (e.g., test/val)",
    )

    args = parser.parse_args()

    if not args.vipe_path.exists():
        print(f"Error: ViPE path '{args.vipe_path}' does not exist.")
        return 1

    # Find artifacts
    artifacts = list(ArtifactPath.glob_artifacts(args.vipe_path, use_video=True))

    # Determine which artifact to use
    if args.video_stem is not None:
        # Use video_stem to match artifact name
        artifacts = [artifact for artifact in artifacts if artifact.artifact_name == args.video_stem]
    elif args.sequence is not None:
        # Try to extract video stem from sequence name (e.g., "0_plates_f0" -> "0")
        # This is a fallback for backward compatibility
        artifacts = [artifact for artifact in artifacts if artifact.artifact_name == args.sequence]

    # Validate at least one artifact was found
    if not artifacts:
        if args.video_stem:
            raise ValueError(f"No ViPE artifacts found matching video_stem='{args.video_stem}' in {args.vipe_path}")
        elif args.sequence:
            raise ValueError(f"No ViPE artifacts found matching sequence='{args.sequence}' in {args.vipe_path}")
        else:
            raise ValueError(f"No ViPE artifacts found in {args.vipe_path}")

    # Set default output path (base), then append dataset_tag
    if args.output is None:
        args.output = args.vipe_path.parent / f"{args.vipe_path.name}_colmap"
    output_dir = args.output / args.dataset_tag

    for artifact in artifacts:
        convert_vipe_to_colmap(artifact, output_dir, args.depth_step, args.use_slam_map, args)
    return 0


if __name__ == "__main__":
    exit(main())
