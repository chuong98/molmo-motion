#!/usr/bin/env python3
"""
Generate query_points (.npz) (and optionally masks) for one input video and one-or-more
text prompts using SAM3 streaming detection.

For each prompt, tries detection at uniformly sampled frames (25%, 50%, 75% of video) until first success. When detected, writes:
  - query npz: <query_dir>/<video_id>_<safe_prompt>_f<frame_idx>.npz
      keys: query_points (N,3) int32 [frame_idx, x, y], dim (2,) int32 [H, W]
  - mask npz (if enabled): <masks_dir>/<video_id>_<safe_prompt>_f<frame_idx>.npz
      keys: mask (H,W) bool, frame (int)
  - combined npz: <query_dir>/<video_id>_combined.npz
      keys: query_points (all prompts concatenated), dim

Also writes a manifest file:
  <query_dir>/querypoints_manifest.txt
containing one generated query npz path per line (absolute paths).
"""

import argparse
import os
import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from sam3.model_builder import build_sam3_video_predictor
from transformers.video_utils import load_video

# Workaround: transformers 5.0.0 passes CLIPTextModelOutput where a tensor is expected.
# Extract pooler_output before it reaches Sam3Model internals.
# Only needed for transformers >= 5.0.0; not needed for 4.57.1.
try:
    import transformers.models.sam3.modeling_sam3 as sam3_module

    _original_forward = sam3_module.Sam3Model.forward

    def _patched_forward(self, pixel_values=None, vision_embeds=None, input_ids=None,
                         attention_mask=None, text_embeds=None, input_boxes=None,
                         input_boxes_labels=None, **kwargs):
        if text_embeds is not None and hasattr(text_embeds, 'pooler_output'):
            text_embeds = text_embeds.pooler_output
        return _original_forward(self, pixel_values, vision_embeds, input_ids,
                                 attention_mask, text_embeds, input_boxes,
                                 input_boxes_labels, **kwargs)

    sam3_module.Sam3Model.forward = _patched_forward
except (ImportError, ModuleNotFoundError):
    pass  # transformers 4.57.1 does not have transformers.models.sam3


def _safe_filename_component(s: str, max_len: int = 120) -> str:
    """
    Make a string safe to embed in a filename.
    Important: dataset prompts can include '/' (e.g. 'a/c remote controller'), which would
    otherwise be interpreted as a path separator and crash saving.
    """
    s = str(s).strip()
    s = re.sub(r"[^0-9A-Za-z]+", "", s)
    if not s:
        s = "prompt"
    return s[:max_len]


def kmeans_sample(mask_bool: np.ndarray, k: int, frame_idx: int = 0, seed: int = 0) -> np.ndarray:
    """
    Sample up to k (frame_idx, x, y) points from a boolean mask using K-means centers.
    Returns (N, 3) int32 array with rows [frame, x, y], N <= k. Empty mask -> (0, 3).
    """
    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        return np.zeros((0, 3), dtype=np.int32)

    coords = np.stack([xs, ys], axis=1)  # (num_pixels, 2) in (x, y)
    n_clusters = min(int(k), len(coords))

    try:
        from sklearn.cluster import KMeans
    except Exception as e:
        raise RuntimeError(
            "scikit-learn is required for KMeans sampling. Install it in your environment "
            "(e.g. `pip install scikit-learn`)."
        ) from e

    km = KMeans(n_clusters=n_clusters, random_state=seed).fit(coords)
    centers = np.rint(km.cluster_centers_).astype(np.int32)  # (N, 2) in (x, y)

    frame_col = np.full((len(centers),), int(frame_idx), dtype=np.int32)
    return np.stack([frame_col, centers[:, 0], centers[:, 1]], axis=1)


def process_one_prompt(
    video_predictor,
    session_id: str,
    video_path: str,
    video_frames: np.ndarray,
    prompt: str,
    args: argparse.Namespace,
    video_id: str,
) -> Optional[str]:
    """
    Process one prompt for a video that already has an active session.
    Tries detection at uniformly sampled frames (25%, 50%, 75% of video). Reports failure if no objects found.

    Returns the query npz path if detection successful, None otherwise.
    """
    total_frames = len(video_frames)
    safe_text = _safe_filename_component(prompt)
    print(f"  Prompt: '{prompt}'")

    # Sample at 25%, 50%, 75% of the video
    if total_frames <= 3:
        frames_to_try = list(range(total_frames))
    else:
        # Sample at 1/4, 1/2, 3/4 of video duration
        frames_to_try = [
            int(total_frames * 0.25),
            int(total_frames * 0.50),
            int(total_frames * 0.75)
        ]
    print(f"  Will try frames: {frames_to_try} (total frames: {total_frames})")

    for frame_idx in frames_to_try:
        # Check if frame exists
        if frame_idx >= total_frames:
            print(f"    Frame {frame_idx} out of bounds (total: {total_frames}), skipping")
            continue

        print(f"    Trying frame {frame_idx}...", end=" ")

        try:
            response = video_predictor.handle_request(
                request=dict[str, str | int](
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=frame_idx,
                    text=prompt,
                )
            )

            output = response.get("outputs", {})
            object_ids = output.get("out_obj_ids", output.get("object_ids", []))

            if len(object_ids) > 0:
                print(f"✓ DETECTED!")

                # Get the frame
                frame = video_frames[frame_idx]

                # Get masks from output
                masks = output.get("out_binary_masks", output.get("masks", None))
                if masks is None:
                    print("    ERROR: No masks in output")
                    continue

                if isinstance(masks, torch.Tensor):
                    masks = masks.cpu().numpy()

                # Sample query points from each object mask separately
                print(f"  Sampling {args.kmeans_k} points per object from {len(masks)} masks")

                args.query_dir.mkdir(parents=True, exist_ok=True)
                all_query_points = []
                for obj_idx, mask in enumerate(masks):
                    q_obj = kmeans_sample(mask.astype(bool), k=args.kmeans_k, frame_idx=frame_idx, seed=args.kmeans_seed + obj_idx)
                    all_query_points.append(q_obj)
                    print(f"    Object {obj_idx}: sampled {len(q_obj)} points")

                q = np.concatenate(all_query_points, axis=0) if all_query_points else np.zeros((0, 3), dtype=np.int32)

                # Combine all masks into a single binary mask for visualization
                combined_mask = np.zeros(masks[0].shape, dtype=bool)
                for mask in masks:
                    combined_mask = np.logical_or(combined_mask, mask.astype(bool))

                h, w = frame.shape[:2]
                qp_npz_path = args.query_dir / f"{video_id}_{safe_text}_f{frame_idx}.npz"
                np.savez_compressed(
                    qp_npz_path,
                    query_points=q,
                    dim=np.asarray([h, w], dtype=np.int32),
                )
                print(f"  Wrote {qp_npz_path} with {len(q)} points")

                # Always save mask NPZ for visualization
                args.masks_dir.mkdir(parents=True, exist_ok=True)
                mask_npz_path = args.masks_dir / f"{video_id}_{safe_text}_f{frame_idx}.npz"
                np.savez(mask_npz_path, mask=combined_mask, frame=frame_idx)

                # ---- optional debug PNG ----
                if args.debug:
                    args.debug_dir.mkdir(parents=True, exist_ok=True)
                    overlay = frame.copy()
                    color = np.array([0, 255, 0], dtype=np.uint8)  # Green
                    overlay[combined_mask] = (overlay[combined_mask] * 0.4 + color * 0.6).astype(np.uint8)
                    png_path = args.debug_dir / f"{video_id}_{safe_text}_f{frame_idx}.png"
                    cv2.imwrite(str(png_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

                return str(qp_npz_path.resolve())
            else:
                print("✗ no detection")

        except Exception as e:
            print(f"✗ ERROR: {e}")

    print(f"  FAILURE: No object found for '{prompt}' at frames {frames_to_try}")
    return None


def process_point_prompts(
    sam3_tracker,
    video_path: str,
    video_frames: np.ndarray,
    point_coords: list[tuple[int, float, float]],
    args: argparse.Namespace,
    video_id: str,
    video_predictor=None,
    session_id: str = None,
    object_text: str = None,
) -> list[str]:
    """
    Process point-based prompts (from Molmo2) using SAM3.

    Combined text+point strategy:
      1. If object_text is provided, run SAM3 text prompt on the detection frame
         to get candidate masks.
      2. Select the text-detected mask that contains the Molmo2 point.
      3. If no text mask contains the point (or text detection fails), fall back
         to the SAM3 tracker with direct point prompts.

    This effectively conditions on BOTH the Qwen3 object name (text) and
    the Molmo2 spatial location (point).

    Args:
        sam3_tracker: SAM3 tracker predictor (sam3_model.tracker with backbone set).
        video_path: Path to video file for tracker init_state.
        video_frames: Loaded video frames (for shape info and visualization).
        point_coords: List of (frame_idx, x_pixel, y_pixel) tuples from Molmo2.
        args: Namespace with query_dir, masks_dir, debug_dir, kmeans_k, etc.
        video_id: Video identifier.
        video_predictor: SAM3 unified predictor (for text prompts). Optional.
        session_id: Active SAM3 session ID. Required if video_predictor is provided.
        object_text: Qwen3-extracted object name for text-based detection. Optional.

    Returns list of absolute query npz paths written.
    """
    if not point_coords:
        print("  No point prompts to process")
        return []

    total_frames = len(video_frames)
    written: list[str] = []

    # Group points by frame
    frame_groups: dict[int, list[tuple[float, float]]] = {}
    for fidx, x, y in point_coords:
        fidx = int(fidx)
        if fidx >= total_frames:
            print(f"  WARNING: Point frame {fidx} out of bounds (total: {total_frames}), skipping")
            continue
        if fidx not in frame_groups:
            frame_groups[fidx] = []
        frame_groups[fidx].append((x, y))

    # Use only the first detected frame per object — avoids extra AllTracker runs
    if len(frame_groups) > 1:
        first_fidx = min(frame_groups.keys())
        dropped = sorted(set(frame_groups.keys()) - {first_fidx})
        print(f"  [first-frame-only] Keeping frame {first_fidx}; dropping frames {dropped}")
        frame_groups = {first_fidx: frame_groups[first_fidx]}

    use_text = (object_text is not None and video_predictor is not None and session_id is not None)
    strategy = "text+point (combined)" if use_text else "point-only (tracker)"
    print(f"  Processing {len(point_coords)} points across {len(frame_groups)} frame(s) via SAM3 {strategy}")

    # Initialize SAM3 tracker session for fallback
    # offload_video_to_cpu=True keeps raw video frames in CPU memory.
    # The SAM3 video predictor session already holds all frames in GPU memory;
    # loading them again to GPU for the tracker causes OOM/SIGSEGV on long videos.
    inference_state = sam3_tracker.init_state(video_path=video_path, offload_video_to_cpu=True)
    next_obj_id = 1

    for fidx in sorted(frame_groups.keys()):
        points_xy = frame_groups[fidx]
        frame = video_frames[fidx]
        h, w = frame.shape[:2]

        all_masks = []
        used_text = False

        # ── Strategy A: Text prompt + point selection ──
        if use_text:
            print(f"    Frame {fidx}: text='{object_text}' + {len(points_xy)} point(s)...", end=" ")
            try:
                # Reset session for fresh detection
                try:
                    video_predictor.handle_request(
                        request=dict(type="reset_session", session_id=session_id)
                    )
                except Exception:
                    pass

                response = video_predictor.handle_request(
                    request=dict[str, str | int](
                        type="add_prompt",
                        session_id=session_id,
                        frame_index=fidx,
                        text=object_text,
                    )
                )
                output = response.get("outputs", {})
                object_ids = output.get("out_obj_ids", output.get("object_ids", []))

                if len(object_ids) > 0:
                    masks = output.get("out_binary_masks", output.get("masks", None))
                    if masks is not None:
                        if isinstance(masks, torch.Tensor):
                            masks = masks.cpu().numpy()

                        n_text_masks = len(masks)
                        print(f"text detected {n_text_masks} mask(s),", end=" ")

                        # For each Molmo2 point, find the best text mask.
                        # Primary: relaxed containment — any mask pixel within POINT_RADIUS of
                        #   the Molmo2 point counts as a hit (handles small pointing offsets).
                        #   Among hits, pick the smallest mask (most specific).
                        # Fallback: use ALL text masks when point misses every mask.
                        POINT_RADIUS = 20  # pixels
                        for pt_idx, (px, py) in enumerate(points_xy):
                            px_int, py_int = int(round(px)), int(round(py))
                            # Clamp to image bounds
                            px_int = max(0, min(px_int, w - 1))
                            py_int = max(0, min(py_int, h - 1))

                            # Primary: relaxed containment — check neighborhood window
                            best_mask = None
                            best_area = float('inf')
                            for mi, m in enumerate(masks):
                                m_bool = m.astype(bool)
                                y0 = max(0, py_int - POINT_RADIUS)
                                y1 = min(h, py_int + POINT_RADIUS + 1)
                                x0 = max(0, px_int - POINT_RADIUS)
                                x1 = min(w, px_int + POINT_RADIUS + 1)
                                if m_bool[y0:y1, x0:x1].any():
                                    area = m_bool.sum()
                                    if area < best_area:
                                        best_mask = m_bool
                                        best_area = area

                            if best_mask is not None:
                                all_masks.append(best_mask)
                                print(f"pt({px_int},{py_int})->text_mask:{best_area}px(r={POINT_RADIUS})", end=" ")
                                used_text = True
                            else:
                                # Fallback: point missed all masks exactly — use ALL text masks.
                                # The text prompt already filters for the right object type,
                                # so all detected masks are valid parts of the target object.
                                nearest_dist = float('inf')
                                total_area = 0
                                valid_masks = []
                                for mi, m in enumerate(masks):
                                    m_bool = m.astype(bool)
                                    ys, xs = np.where(m_bool)
                                    if len(xs) == 0:
                                        continue
                                    cx, cy = float(xs.mean()), float(ys.mean())
                                    dist = ((cx - px_int) ** 2 + (cy - py_int) ** 2) ** 0.5
                                    nearest_dist = min(nearest_dist, dist)
                                    total_area += int(m_bool.sum())
                                    valid_masks.append(m_bool)
                                if valid_masks:
                                    all_masks.extend(valid_masks)
                                    print(f"pt({px_int},{py_int})->all_text_masks:{len(valid_masks)}masks/{total_area}px(nearest_dist={nearest_dist:.0f}px)", end=" ")
                                    used_text = True
                                else:
                                    print(f"pt({px_int},{py_int})->no text mask", end=" ")

            except Exception as e:
                print(f"text prompt failed ({e}),", end=" ")

        # ── Strategy B: Point-only tracker fallback ──
        if not used_text:
            print(f"    Frame {fidx}: {len(points_xy)} point(s) -> tracker point prompt...", end=" ") if use_text else None
            if not use_text:
                print(f"    Frame {fidx}: {len(points_xy)} point(s) -> SAM3 tracker point prompt...", end=" ")
            else:
                print("falling back to tracker...", end=" ")

            try:
                sam3_tracker.clear_all_points_in_video(inference_state)

                for pt_idx, (px, py) in enumerate(points_xy):
                    cx_norm = px / w
                    cy_norm = py / h
                    obj_id = next_obj_id
                    next_obj_id += 1

                    points_tensor = torch.tensor([[cx_norm, cy_norm]], dtype=torch.float32)
                    labels_tensor = torch.tensor([1], dtype=torch.int32)

                    _, out_obj_ids, low_res_masks, video_res_masks = sam3_tracker.add_new_points(
                        inference_state=inference_state,
                        frame_idx=fidx,
                        obj_id=obj_id,
                        points=points_tensor,
                        labels=labels_tensor,
                        clear_old_points=True,
                    )

                    if video_res_masks is not None and len(video_res_masks) > 0:
                        if obj_id in list(out_obj_ids):
                            idx = list(out_obj_ids).index(obj_id)
                        else:
                            idx = -1
                        mask = (video_res_masks[idx] > 0.0).cpu().numpy().squeeze()
                        all_masks.append(mask)
                        print(f"obj{obj_id}:{mask.sum()}px", end=" ")

            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()

        if len(all_masks) > 0:
            print(f"-> {len(all_masks)} mask(s) ({'text+point' if used_text else 'tracker'})")

            print(f"  Sampling {args.kmeans_k} points per object from {len(all_masks)} masks")

            args.query_dir.mkdir(parents=True, exist_ok=True)
            all_query_points = []
            for obj_idx, mask in enumerate(all_masks):
                q_obj = kmeans_sample(
                    mask.astype(bool), k=args.kmeans_k,
                    frame_idx=fidx, seed=args.kmeans_seed + obj_idx
                )
                all_query_points.append(q_obj)
                print(f"    Object {obj_idx}: sampled {len(q_obj)} points")

            q = np.concatenate(all_query_points, axis=0) if all_query_points else np.zeros((0, 3), dtype=np.int32)

            combined_mask = np.zeros(all_masks[0].shape, dtype=bool)
            for mask in all_masks:
                combined_mask = np.logical_or(combined_mask, mask.astype(bool))

            obj_tag = _safe_filename_component(object_text) if object_text else "obj"
            safe_text = f"molmo2_{obj_tag}_f{fidx}"
            qp_npz_path = args.query_dir / f"{video_id}_{safe_text}_f{fidx}.npz"
            np.savez_compressed(
                qp_npz_path,
                query_points=q,
                dim=np.asarray([h, w], dtype=np.int32),
            )
            print(f"  Wrote {qp_npz_path} with {len(q)} points")
            written.append(str(qp_npz_path.resolve()))

            # Debug PNG
            if args.debug:
                args.debug_dir.mkdir(parents=True, exist_ok=True)
                overlay = frame.copy()
                color = np.array([0, 255, 0], dtype=np.uint8)
                overlay[combined_mask] = (overlay[combined_mask] * 0.4 + color * 0.6).astype(np.uint8)
                for px, py in points_xy:
                    cv2.circle(overlay, (int(px), int(py)), 5, (255, 0, 0), -1)
                    cv2.circle(overlay, (int(px), int(py)), 8, (255, 255, 255), 2)
                png_path = args.debug_dir / f"{video_id}_{safe_text}_f{fidx}.png"
                cv2.imwrite(str(png_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            # Save masks for visualization
            args.masks_dir.mkdir(parents=True, exist_ok=True)
            mask_npz_path = args.masks_dir / f"{video_id}_{safe_text}_f{fidx}.npz"
            np.savez(mask_npz_path, mask=combined_mask, frame=fidx)
        else:
            print("no detection from any strategy")

    return written


def process_one_video_with_molmo2(
    video_predictor,
    molmo2_model,
    molmo2_processor,
    video_path: str,
    args: argparse.Namespace,
    video_id: Optional[str] = None,
    text_prompts: Optional[list[str]] = None,
    instruction: Optional[str] = None,
    fps: float = 15.0,
    molmo2_prompt: Optional[str] = None,
    molmo2_per_object_prompts: Optional[list[str]] = None,
    sam3_tracker=None,
    object_text: Optional[str] = None,
    per_object_pairs: Optional[list[tuple[str, str]]] = None,
) -> list[str]:
    """
    Process a video using Molmo2 for object detection + SAM3 point prompts (tracker).
    Optionally also processes text prompts (e.g. for hands) using the existing text-prompt path.

    Args:
        video_predictor: SAM3 video predictor
        molmo2_model: Loaded Molmo2 model
        molmo2_processor: Loaded Molmo2 processor
        video_path: Path to video file
        args: Namespace with query_dir, masks_dir, debug_dir, kmeans_k, etc.
        video_id: Optional video identifier
        text_prompts: Optional list of text prompts to also process (e.g. ["left hand", "right hand"])
        instruction: DROID language instruction for building specific Molmo2 prompt
        fps: Video FPS for timestamp mapping
        molmo2_prompt: Explicit single Molmo2 prompt (overrides instruction-based prompt building)
        molmo2_per_object_prompts: List of per-object Molmo2 prompts. If provided, runs Molmo2
            once per prompt and aggregates all detected points. Overrides molmo2_prompt and instruction.

    Returns list of absolute query npz paths written.
    """
    from molmo2_pointing import detect_manipulated_objects_multi_frame

    if video_id is None:
        video_id = Path(video_path).stem
    video_frames, _ = load_video(video_path, backend="opencv")
    if args.max_frames and args.max_frames > 0:
        video_frames = video_frames[: args.max_frames]

    # Subsample to target fps (default 15fps) if source video is higher fps
    _vcap = cv2.VideoCapture(video_path)
    orig_fps = _vcap.get(cv2.CAP_PROP_FPS) or fps
    _vcap.release()
    if orig_fps > fps and fps > 0:
        _step = max(1, round(orig_fps / fps))
        video_frames = video_frames[::_step]
        print(f"  [fps] Subsampled {int(orig_fps)}fps → {fps:.0f}fps (step={_step}): {len(video_frames)} frames")
    else:
        _step = 1
        print(f"  [fps] No subsampling needed ({orig_fps:.1f}fps → {fps:.0f}fps target)")

    total_frames_cv = len(video_frames)
    print(f"Loaded video: {video_path} ({total_frames_cv} frames)")

    import json as _json

    def _clamp_frames(pts):
        """Map VP4B frame indices (original fps space) to subsampled fps space, then clamp."""
        clamped = []
        for f, x, y in pts:
            f = int(f)
            # VP4B reads the original video file, so frame indices are in original fps space.
            # Convert to subsampled fps space by dividing by the subsampling step.
            f_mapped = f // _step
            if f_mapped >= total_frames_cv:
                print(f"  [Molmo2] Clamping frame {f} (orig) → {total_frames_cv - 1} "
                      f"(mapped {f_mapped} out of bounds, total={total_frames_cv})")
                f_mapped = total_frames_cv - 1
            elif _step > 1:
                print(f"  [Molmo2] Frame {f} (orig {int(orig_fps):.0f}fps) → {f_mapped} (subsampled {fps:.0f}fps)")
            clamped.append((f_mapped, x, y))
        return clamped

    # Step 1: Use Molmo2 to detect manipulated objects
    if per_object_pairs:
        # Per-object path: detection happens in Step 2 per object; skip aggregate detection
        point_coords = []
    elif molmo2_per_object_prompts:
        # Run Molmo2 once per object prompt and aggregate all points
        point_coords = []
        for obj_prompt in molmo2_per_object_prompts:
            obj_points = detect_manipulated_objects_multi_frame(
                molmo2_model, molmo2_processor, video_path,
                prompt=obj_prompt, fps=fps,
            )
            point_coords.extend(_clamp_frames(obj_points))
    elif molmo2_prompt:
        # Use explicit prompt
        point_coords = _clamp_frames(detect_manipulated_objects_multi_frame(
            molmo2_model, molmo2_processor, video_path,
            prompt=molmo2_prompt, fps=fps,
        ))
    else:
        # Fall back to instruction-based prompt building
        point_coords = _clamp_frames(detect_manipulated_objects_multi_frame(
            molmo2_model, molmo2_processor, video_path,
            instruction=instruction, fps=fps,
        ))

    args.query_dir.mkdir(parents=True, exist_ok=True)

    if not per_object_pairs:
        print(f"  [Molmo2] Got {len(point_coords)} point(s)")
        # Save Molmo2 detection metadata for visualization
        molmo2_meta = {
            "video_id": video_id,
            "detections": [(int(f), round(float(x), 1), round(float(y), 1)) for f, x, y in point_coords],
            "prompts": molmo2_per_object_prompts if molmo2_per_object_prompts else [molmo2_prompt or instruction or "default"],
            "instruction": instruction,
        }
        meta_path = args.query_dir / f"{video_id}_molmo2_meta.json"
        with open(meta_path, "w") as _f:
            _json.dump(molmo2_meta, _f, indent=2)
        print(f"  [Molmo2] Saved metadata: {meta_path}")

    # Save temporary video for SAM3 session
    downsampled_video_path = args.query_dir / f"{video_id}_temp_video.mp4"

    try:
        h, w = video_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(str(downsampled_video_path), fourcc, fps, (w, h))
        for frame in video_frames:
            out_video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out_video.release()
    except Exception as e:
        print(f"  WARNING: Failed to save temporary video: {e}")
        return []

    # Create SAM3 session
    try:
        response = video_predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=str(downsampled_video_path),
            )
        )
        session_id = response["session_id"]
        print(f"  Created SAM3 session {session_id}")
    except Exception as e:
        print(f"  ERROR: Failed to start SAM3 session: {e}")
        if downsampled_video_path.exists():
            downsampled_video_path.unlink()
        return []

    written: list[str] = []

    # Step 2: Process Molmo2 points through SAM3 (text+point combined, or point-only fallback)
    if per_object_pairs:
        # Per-object: run Molmo2 detection + SAM3 separately for each (molmo2_prompt, object_text)
        all_meta = []
        for obj_prompt, obj_text in per_object_pairs:
            obj_points = _clamp_frames(detect_manipulated_objects_multi_frame(
                molmo2_model, molmo2_processor, video_path,
                prompt=obj_prompt, fps=fps,
            ))
            print(f"  [Molmo2] '{obj_text}': {len(obj_points)} point(s)")
            all_meta.append({
                "prompt": obj_prompt,
                "object": obj_text,
                "points": [(int(f), round(float(x), 1), round(float(y), 1)) for f, x, y in obj_points],
            })
            if obj_points and sam3_tracker is not None:
                obj_results = process_point_prompts(
                    sam3_tracker=sam3_tracker,
                    video_path=str(downsampled_video_path),
                    video_frames=video_frames,
                    point_coords=obj_points,
                    args=args,
                    video_id=video_id,
                    video_predictor=video_predictor,
                    session_id=session_id,
                    object_text=obj_text,
                )
                written.extend(obj_results)
        # Save per-object metadata
        args.query_dir.mkdir(parents=True, exist_ok=True)
        molmo2_meta = {"video_id": video_id, "per_object": all_meta, "instruction": instruction}
        meta_path = args.query_dir / f"{video_id}_molmo2_meta.json"
        with open(meta_path, "w") as _f:
            _json.dump(molmo2_meta, _f, indent=2)
        print(f"  [Molmo2] Saved metadata: {meta_path}")
    elif point_coords and sam3_tracker is not None:
        point_results = process_point_prompts(
            sam3_tracker=sam3_tracker,
            video_path=str(downsampled_video_path),
            video_frames=video_frames,
            point_coords=point_coords,
            args=args,
            video_id=video_id,
            video_predictor=video_predictor,
            session_id=session_id,
            object_text=object_text,
        )
        written.extend(point_results)
    elif point_coords:
        print("  WARNING: sam3_tracker not provided, skipping point prompts")

    # Step 3: Process text prompts if any (e.g. for hands)
    if text_prompts:
        for prompt_idx, prompt in enumerate(text_prompts):
            print(f"  Processing text prompt {prompt_idx + 1}/{len(text_prompts)}: '{prompt}'")
            try:
                video_predictor.handle_request(
                    request=dict(type="reset_session", session_id=session_id)
                )
            except Exception:
                pass

            result = process_one_prompt(
                video_predictor=video_predictor,
                session_id=session_id,
                video_path=video_path,
                video_frames=video_frames,
                prompt=prompt,
                args=args,
                video_id=video_id,
            )
            if result:
                written.append(result)

    # Clean up
    try:
        video_predictor.handle_request(
            request=dict(type="close_session", session_id=session_id)
        )
    except Exception:
        pass

    if downsampled_video_path.exists():
        downsampled_video_path.unlink()

    if written:
        total_points = sum(np.load(p)['query_points'].shape[0] for p in written)
        print(f"Wrote {len(written)} query point file(s) with {total_points} total points")

    return written


def process_one_video_and_extract_qp(
    video_predictor,
    video_path: str,
    prompts: list[str],
    args: argparse.Namespace,
    video_id: Optional[str] = None,
) -> list[str]:
    """
    For each prompt, try detection at frames 0, 10, 20 until first success.
    When detected, save query npz files.

    Returns a list of absolute query npz paths written.
    """
    if video_id is None:
        video_id = Path(video_path).stem
    video_frames, _ = load_video(video_path, backend="opencv")
    if args.max_frames and args.max_frames > 0:
        video_frames = video_frames[: args.max_frames]
    print(f"Loaded video: {video_path} ({len(video_frames)} frames)")

    # Add hand prompts if enabled
    if args.include_hands:
        prompts = prompts + ["left hand", "right hand"]
        print(f"Added hand prompts. Total prompts: {prompts}")

    # Save downsampled video for session
    args.query_dir.mkdir(parents=True, exist_ok=True)
    downsampled_video_path = args.query_dir / f"{video_id}_temp_video.mp4"

    try:
        h, w = video_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(str(downsampled_video_path), fourcc, 15, (w, h))
        for frame in video_frames:
            out_video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out_video.release()
        print(f"  Saved temporary video: {downsampled_video_path.name}")
    except Exception as e:
        print(f"  WARNING: Failed to save temporary video: {e}")
        return []

    # Create ONE session for this video (reuse for all prompts)
    try:
        response = video_predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=str(downsampled_video_path),
            )
        )
        session_id = response["session_id"]
        print(f"  Created session {session_id} with video ({len(video_frames)} frames)")
    except Exception as e:
        print(f"  ERROR: Failed to start session: {e}")
        if downsampled_video_path.exists():
            downsampled_video_path.unlink()
        return []

    written: list[str] = []

    # Process each prompt separately (reusing the same session)
    for prompt_idx, prompt in enumerate(prompts):
        print(f"  Processing prompt {prompt_idx + 1}/{len(prompts)}: '{prompt}'")

        # Reset session between prompts to clear previous detections
        if prompt_idx > 0:
            try:
                video_predictor.handle_request(
                    request=dict(
                        type="reset_session",
                        session_id=session_id,
                    )
                )
            except Exception as e:
                print(f"    WARNING: Failed to reset session: {e}")

        # Process video with this prompt
        result = process_one_prompt(
            video_predictor=video_predictor,
            session_id=session_id,
            video_path=video_path,
            video_frames=video_frames,
            prompt=prompt,
            args=args,
            video_id=video_id,
        )

        if result:
            written.append(result)

    # Clean up: close the session
    try:
        video_predictor.handle_request(
            request=dict(
                type="close_session",
                session_id=session_id,
            )
        )
        print(f"  Closed session {session_id}")
    except Exception as e:
        print(f"  WARNING: Failed to close session: {e}")

    # Clean up temporary video file
    if downsampled_video_path.exists():
        downsampled_video_path.unlink()

    # Print summary
    if len(written) > 0:
        total_points = sum(np.load(p)['query_points'].shape[0] for p in written)
        print(f"Wrote {len(written)} query point file(s) with {total_points} total points")

    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate SAM3 query_points for one video + prompts")
    parser.add_argument("--video_path", required=True, help="Path to input video (any format opencv can read)")
    parser.add_argument(
        "--text_prompt",
        action="append",
        default=[],
        help="Text prompt to detect (repeatable). Example: --text_prompt 'cup' --text_prompt 'pot'",
    )
    parser.add_argument(
        "--prompts",
        default=None,
        help="Comma-separated prompts (alternative to repeating --text_prompt).",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help=(
            "Base output directory. By default outputs will be written to:\n"
            "  <out_dir>/query_points (query npz + manifest)\n"
            "  <out_dir>/masks (mask npz, if enabled)\n"
            "  <out_dir>/debug_png (debug overlays, if --debug)"
        ),
    )
    parser.add_argument(
        "--query_dir",
        default=None,
        help="Where to write query_points .npz files (defaults to <out_dir>/query_points)",
    )
    parser.add_argument(
        "--masks_dir",
        default=None,
        help="Where to write mask .npz files (defaults to <out_dir>/masks)",
    )
    parser.add_argument(
        "--save_masks",
        action="store_true",
        help="If set, also save mask .npz files alongside query points (one per detected prompt).",
    )
    parser.add_argument(
        "--debug_dir",
        default=None,
        help="Where to write debug PNGs (defaults to <out_dir>/debug_png)",
    )
    parser.add_argument("--kmeans_k", type=int, default=100, help="K for KMeans sampling")
    parser.add_argument("--kmeans_seed", type=int, default=0, help="Random seed for KMeans sampling")
    parser.add_argument("--debug", action="store_true", help="Write debug overlay PNGs for detections")
    parser.add_argument("--max_frames", type=int, default=0, help="If >0, cap to first N frames")
    parser.add_argument(
        "--include_hands",
        action="store_true",
        default=False,
        help="Add 'left hand' and 'right hand' as additional prompts to detect hands",
    )
    parser.add_argument(
        "--video_id",
        default=None,
        help="Video ID to use for output filenames (defaults to video filename stem)",
    )

    args = parser.parse_args()
    video_path = str(args.video_path)
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"--video_path does not exist: {video_path}")

    prompts = list(args.text_prompt)
    if args.prompts:
        prompts.extend([p.strip() for p in args.prompts.split(",") if p.strip()])
    # de-dupe, keep order
    seen = set()
    prompts = [p for p in prompts if not (p in seen or seen.add(p))]
    if len(prompts) == 0:
        raise ValueError("No prompts provided. Use --text_prompt ... or --prompts 'a,b,c'.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    args.query_dir = Path(args.query_dir) if args.query_dir else (out_dir / "query_points")
    args.masks_dir = Path(args.masks_dir) if args.masks_dir else (out_dir / "masks")
    args.debug_dir = Path(args.debug_dir) if args.debug_dir else (out_dir / "debug_png")

    manifest_path = args.query_dir / "querypoints_manifest.txt"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        manifest_path.unlink()

    print("Loading SAM3 video predictor...")
    video_predictor = build_sam3_video_predictor()
    print("Model loaded successfully!\n")

    written = process_one_video_and_extract_qp(video_predictor, video_path, prompts, args, video_id=args.video_id)

    with open(manifest_path, "w") as f:
        for p in written:
            f.write(p + "\n")

    print(f"Wrote manifest: {manifest_path} ({len(written)} query files)")
    if len(written) == 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
