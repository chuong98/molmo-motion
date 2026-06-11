#!/usr/bin/env python3
"""
Stage 6 — Video-level motion clipping.

Each long source video is segmented into short, motion-coherent training clips
(paper §4.1, "Video-level clipping"). For every object sub-group we:

  1. Compute a per-frame object motion score  s_t = trimmed-mean over visible
     points of ||p_n(t) - p_n(t-1)||_2  (top-20% of points trimmed as outliers).
  2. Threshold  active = (s_t >= tau)  to obtain contiguous high-motion runs.
  3. Merge runs separated by a gap shorter than `min_gap` frames.
  4. Drop runs shorter than the minimum-clip floor.

Input is the Stage-5 output for each video:
    final_tracks/<vid>_3d.npz         points_3d:(N,T,3)  visibility:(N,T[,1])
    final_tracks/<vid>_filter_meta.npz sub_object_labels:(N,)  keep_mask:(N,)

Output is one JSON with the per-object clip ranges (inclusive on both ends):
    {
      "file": "<vid>", "fps": 15, "num_frames": T,
      "clips_by_object": {"obj0": [[s,e], ...], "obj1": [...]},
      "num_clips_total": K
    }

Algorithm and thresholds match the released MolmoMotion-1M pipeline.
"""
import argparse
import json
from pathlib import Path

import numpy as np

TRIM_PERCENTILE = 80  # keep the bottom 80% of per-point magnitudes


def compute_motion_magnitudes(pts_NT3: np.ndarray, vis_NT: np.ndarray) -> np.ndarray:
    """Per-frame trimmed-mean L2 displacement. Returns (T,), magnitudes[0] = 0."""
    N, T, _ = pts_NT3.shape
    if T < 2:
        return np.zeros(T, dtype=np.float64)

    pts_t = np.transpose(pts_NT3, (1, 0, 2))  # (T, N, 3)
    vis_t = vis_NT.T                           # (T, N)

    deltas = pts_t[1:] - pts_t[:-1]            # (T-1, N, 3)
    mags = np.linalg.norm(deltas, axis=-1)     # (T-1, N)
    both_vis = vis_t[1:] & vis_t[:-1]
    mags = np.where(both_vis & np.isfinite(mags), mags, np.nan)

    valid_counts = np.sum(np.isfinite(mags), axis=-1)
    with np.errstate(all="ignore"):
        p80 = np.nanpercentile(mags, TRIM_PERCENTILE, axis=-1, keepdims=True)
    # Only trim on frames with >= 5 valid points; otherwise keep all.
    cutoff = np.where(valid_counts[:, None] >= 5, p80, np.inf)
    mags_trimmed = np.where(mags <= cutoff, mags, np.nan)

    with np.errstate(all="ignore"):
        mean = np.nanmean(mags_trimmed, axis=-1)
    mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)

    result = np.zeros(T, dtype=np.float64)
    result[1:] = mean
    return result


def segment_high_motion(magnitudes: np.ndarray, fps: int, threshold: float,
                        min_gap: int = 5, min_clip_sec: float = 0.5,
                        min_frames: int | None = None) -> list:
    """Threshold -> merge short gaps -> drop short runs. Returns [[s, e], ...]."""
    T = len(magnitudes)
    active = magnitudes >= threshold

    segments = []
    in_seg, start = False, 0
    for t in range(T):
        if active[t] and not in_seg:
            start, in_seg = t, True
        elif not active[t] and in_seg:
            segments.append((start, t - 1))
            in_seg = False
    if in_seg:
        segments.append((start, T - 1))
    if not segments:
        return []

    merged = [list(segments[0])]
    for s, e in segments[1:]:
        if s - merged[-1][1] - 1 < min_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    min_len = max(1, int(min_frames)) if min_frames is not None \
        else max(3, int(round(min_clip_sec * fps)))
    return [[int(s), int(e)] for s, e in merged if (e - s + 1) >= min_len]


def load_final_tracks(vid: str, final_tracks_dir: Path) -> dict:
    """Load Stage-5 output and split into {obj_name: (pts_NT3, vis_NT)} groups."""
    d3 = np.load(final_tracks_dir / f"{vid}_3d.npz", allow_pickle=True)
    pts = d3["points_3d"]                       # (N, T, 3)
    vis = d3["visibility"]
    if vis.ndim == 3:
        vis = vis[..., 0]                       # (N, T, 1) -> (N, T)
    vis = vis.astype(bool)

    meta_path = final_tracks_dir / f"{vid}_filter_meta.npz"
    if meta_path.exists():
        dm = np.load(meta_path, allow_pickle=True)
        labels = dm["sub_object_labels"]        # (N_orig,)
        keep = dm["keep_mask"]                  # (N_orig,)
        kept_labels = labels[keep]              # aligns with kept tracks in pts
        groups = {}
        for lbl in np.unique(kept_labels[kept_labels >= 0]):
            idx = kept_labels == lbl
            if idx.sum() >= 3:
                groups[f"obj{int(lbl)}"] = (pts[idx], vis[idx])
        if groups:
            return groups

    # No meta (or no labels): treat the whole point set as a single object.
    return {"obj0": (pts, vis)} if pts.shape[0] >= 3 else {}


def clip_one_video(vid: str, final_tracks_dir: Path, fps: int, threshold: float,
                   min_gap: int, min_clip_sec: float,
                   min_frames: int | None) -> dict | None:
    objs = load_final_tracks(vid, final_tracks_dir)
    if not objs:
        return None

    clips_by_object = {}
    num_frames = 0
    for name, (pts, vis) in objs.items():
        T = pts.shape[1]
        if T < 3:
            continue
        num_frames = max(num_frames, T)
        mags = compute_motion_magnitudes(pts, vis)
        segs = segment_high_motion(mags, fps=fps, threshold=threshold,
                                   min_gap=min_gap, min_clip_sec=min_clip_sec,
                                   min_frames=min_frames)
        if segs:
            clips_by_object[name] = segs

    if not clips_by_object:
        return None
    return {
        "file": vid,
        "fps": fps,
        "num_frames": int(num_frames),
        "clips_by_object": clips_by_object,
        "num_clips_total": sum(len(v) for v in clips_by_object.values()),
    }


def main():
    p = argparse.ArgumentParser(description="Stage 6 — video-level motion clipping")
    p.add_argument("--final_tracks_dir", required=True,
                   help="Directory of Stage-5 <vid>_3d.npz / <vid>_filter_meta.npz")
    p.add_argument("--video_ids", nargs="+", required=True,
                   help="Video ids to segment (basename of the *_3d.npz files)")
    p.add_argument("--out_json", required=True, help="Output clips JSON path")
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--threshold", type=float, default=0.02,
                   help="Motion threshold tau in metres/frame (paper default 0.02)")
    p.add_argument("--min_gap", type=int, default=5)
    p.add_argument("--min_clip_sec", type=float, default=0.5)
    p.add_argument("--min_frames", type=int, default=None,
                   help="Override the min-clip floor (e.g. 24 for sim corpora)")
    args = p.parse_args()

    final_tracks_dir = Path(args.final_tracks_dir)
    results = []
    for vid in args.video_ids:
        try:
            entry = clip_one_video(vid, final_tracks_dir, args.fps, args.threshold,
                                   args.min_gap, args.min_clip_sec, args.min_frames)
        except FileNotFoundError as e:
            print(f"  [skip] {vid}: {e}")
            continue
        if entry is None:
            print(f"  [drop] {vid}: no clip survives (static or no valid tracks)")
            continue
        results.append(entry)
        print(f"  [ok]   {vid}: {entry['num_clips_total']} clips across "
              f"{len(entry['clips_by_object'])} object(s)")

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {len(results)} entries -> {out}")


if __name__ == "__main__":
    main()
