#!/usr/bin/env python3
"""
MolmoMotion-1M data-generation pipeline — end-to-end driver.

Turns raw RGB video + an action description into object-grounded 3D point
trajectories and motion-coherent clips, exactly as described in the paper
(§3.1, §4.1) and the appendix.

Stages (run any contiguous subset with --start_stage / --end_stage):
    1  Grounding        Qwen3 + (Molmo2-8B recaption) + MolmoPoint + SAM3 + K-means
    2  Depth + pose     ViPE monocular SLAM  (per-frame metric depth, camera poses)
    3  2D tracking      AllTracker dense point tracks on the query points
    4  3D lift          back-project 2D tracks to a metric world frame at t0
    5  Filter + smooth  consensus-gated trust weighting + ray-only smoothing
    6  Clip             segment each video into motion-coherent clips

Every stage caches its output and is skipped if already present, so the pipeline
is freely resumable.

Inputs
    --tasks   JSON list of {"video_id", "video_path", "action"} objects
    --config  YAML hyperparameters + corpus prompt settings (see configs/)
    --work_dir  all intermediate + final artifacts are written here

This driver hardcodes no machine-specific paths. It uses the active Python
(`sys.executable`) and the installed `vipe` console script; point HF_HOME at your
HuggingFace cache if you do not want the default (~/.cache/huggingface).

Example
    python run_pipeline.py \
        --tasks examples/tasks_example.json \
        --config configs/human_manipulation.yaml \
        --work_dir ./runs/example
"""
import argparse
import gc
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

REPO_ROOT = Path(__file__).resolve().parent
THIRD_PARTY = REPO_ROOT / "third_party"
PY = sys.executable  # run children with the same interpreter / env


# ──────────────────────────────────────────────────────────────────────────────
# Config + paths
# ──────────────────────────────────────────────────────────────────────────────

DEFAULTS = {
    # grounding
    "kmeans_k": 100,
    "max_frames": 0,            # 0 = all frames
    "agent": "hand",            # "hand" | "robot gripper"
    "point_prompt_template": "point to {obj} gripped and picked up by the {agent}",
    "recaption": True,
    "save_masks": False,
    "save_debug": False,
    # video / tracking
    "fps": 15,
    "encode_480p": True,
    "alltracker_max_side": 512,
    "max_frame_groups": 5,
    # filter + smooth (paper / App. A.6 defaults)
    "smooth_steps": 100,
    "smooth_lr": 0.05,
    "alpha": 1000.0,
    "lambda_reg": 1e-4,
    "z_thresh": 1.0,
    "n_anchors": 16,
    "gating_power": 2,
    "min_cluster_size": 10,
    "depth_step": 1,
    # clipping (paper §4.1)
    "clip_threshold": 0.02,
    "clip_min_gap": 5,
    "clip_min_clip_sec": 0.5,
    "clip_min_frames": None,
    # tooling
    "vipe_cmd": "vipe",
    "corpus": "molmomotion",
}


def load_config(path):
    cfg = dict(DEFAULTS)
    if path:
        if yaml is None:
            raise RuntimeError("pyyaml is required to read --config; pip install pyyaml")
        with open(path) as f:
            user = yaml.safe_load(f) or {}
        cfg.update(user)
    return cfg


class Paths:
    """All artifact locations derived from a single work_dir."""
    def __init__(self, work_dir: Path):
        self.work_dir = work_dir
        self.grounding = work_dir / "grounding"          # <vid>/query_points/*.npz
        self.videos_480p = work_dir / "videos_480p"
        self.vipe_results = work_dir / "vipe_results"    # rgb/ depth/ pose/ intrinsics/
        self.tracks_2d = work_dir / "tracks_2d"          # <vid>/<vid>_merged.npz
        self.tracks_3d = work_dir / "tracks_3d"          # <tag>/<vid>_merged_3d_tracks.npz
        self.final_tracks = work_dir / "final_tracks"    # <vid>_{2d,3d}.npz + _filter_meta.npz
        self.clips = work_dir / "clips"
        self.tmp = work_dir / ".tmp"
        for d in (self.grounding, self.videos_480p, self.vipe_results, self.tracks_2d,
                  self.tracks_3d, self.final_tracks, self.clips, self.tmp):
            d.mkdir(parents=True, exist_ok=True)


def sh(cmd, **kw):
    """Run a command list, inheriting env. Returns the CompletedProcess."""
    return subprocess.run(cmd, env=os.environ.copy(), **kw)


def free_gpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1 — grounding
# ──────────────────────────────────────────────────────────────────────────────

def stage1_grounding(tasks, cfg, paths):
    print("=" * 80, "\nSTAGE 1: Grounding + query points\n", "=" * 80, sep="")
    tasks_json = paths.tmp / "grounding_tasks.json"
    prompts_json = paths.tmp / "grounding_prompts.json"
    config_json = paths.tmp / "grounding_config.json"
    tasks_json.write_text(json.dumps(tasks))
    config_json.write_text(json.dumps(cfg))

    worker = REPO_ROOT / "pipeline" / "grounding_worker.py"
    # SAM 3 can rarely SIGSEGV; the worker is resumable, so retry a few times.
    for attempt in range(5):
        remaining = [t for t in tasks
                     if not list((paths.grounding / t["video_id"] / "query_points")
                                 .glob(f"{t['video_id']}_*_f*.npz"))]
        if not remaining:
            print("  All tasks grounded.")
            break
        print(f"  Attempt {attempt + 1}: {len(remaining)} task(s) remaining...")
        ret = sh([PY, str(worker), str(tasks_json), str(prompts_json),
                  str(paths.grounding), str(config_json)])
        if ret.returncode == 0:
            break
        print(f"  worker exited {ret.returncode}; restarting (resumes from cache)")
    free_gpu()


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2 — ViPE depth + pose
# ──────────────────────────────────────────────────────────────────────────────

def _ffmpeg_encode_one(video_id, video_path, out_dir, fps):
    import imageio_ffmpeg
    out = out_dir / f"{video_id}.mp4"
    if out.exists():
        return video_id, "cached"
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    r = subprocess.run(
        [ffmpeg, "-y", "-i", video_path, "-vf", "scale=-2:480", "-r", str(fps),
         "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-an", str(out)],
        capture_output=True)
    if r.returncode != 0 or not out.exists():
        if out.exists():
            out.unlink()
        raise RuntimeError(f"ffmpeg failed for {video_id}: {r.stderr.decode(errors='replace')[-400:]}")
    return video_id, "encoded"


def stage2_vipe(tasks, cfg, paths):
    print("=" * 80, "\nSTAGE 2: ViPE depth + pose\n", "=" * 80, sep="")
    src_dir = paths.videos_480p if cfg["encode_480p"] else None

    if cfg["encode_480p"]:
        todo = [t for t in tasks if not (paths.videos_480p / f"{t['video_id']}.mp4").exists()]
        if todo:
            print(f"  Pre-encoding {len(todo)} video(s) to 480p...")
            with ThreadPoolExecutor(max_workers=min(8, len(todo))) as pool:
                futs = {pool.submit(_ffmpeg_encode_one, t["video_id"], t["video_path"],
                                    paths.videos_480p, cfg["fps"]): t["video_id"] for t in todo}
                for fut in as_completed(futs):
                    try:
                        vid, msg = fut.result()
                        print(f"    {vid}: {msg}")
                    except RuntimeError as e:
                        print(f"    {futs[fut]}: WARN {e}")

    for i, task in enumerate(tasks):
        vid = task["video_id"]
        if (paths.vipe_results / "rgb" / f"{vid}.mp4").exists():
            print(f"[{i+1}/{len(tasks)}] SKIP {vid} — ViPE output exists")
            continue
        inp = (src_dir / f"{vid}.mp4") if src_dir else Path(task["video_path"])
        if not inp.exists():
            print(f"[{i+1}/{len(tasks)}] SKIP {vid} — input video missing ({inp})")
            continue
        print(f"[{i+1}/{len(tasks)}] ViPE {vid}")
        r = sh([cfg["vipe_cmd"], "infer", str(inp), "--output", str(paths.vipe_results)])
        if r.returncode != 0:
            print(f"  ERROR: ViPE failed (code {r.returncode})")
    free_gpu()


# ──────────────────────────────────────────────────────────────────────────────
# Stage 3 — AllTracker 2D tracking
# ──────────────────────────────────────────────────────────────────────────────

def _scale_qps(frame_qps, orig_dim, target_h, target_w):
    H, W = int(orig_dim[0]), int(orig_dim[1])
    if H == target_h and W == target_w:
        return frame_qps, orig_dim
    sx, sy = target_w / W, target_h / H
    scaled = frame_qps.copy()
    scaled[:, 1] = np.round(frame_qps[:, 1] * sx).astype(np.int32)
    scaled[:, 2] = np.round(frame_qps[:, 2] * sy).astype(np.int32)
    return scaled, np.array([target_h, target_w], dtype=np.int32)


def stage3_alltracker(tasks, cfg, paths):
    print("=" * 80, "\nSTAGE 3: AllTracker 2D tracking\n", "=" * 80, sep="")
    alltracker_dir = THIRD_PARTY / "alltracker"

    for i, task in enumerate(tasks):
        vid = task["video_id"]
        video_480p = (paths.videos_480p / f"{vid}.mp4") if cfg["encode_480p"] else Path(task["video_path"])
        if not video_480p.exists():
            print(f"[{i+1}/{len(tasks)}] SKIP {vid} — tracking video missing; run Stage 2")
            continue

        qp_dir = paths.grounding / vid / "query_points"
        qp_files = sorted(qp_dir.glob(f"{vid}_*_f*.npz")) if qp_dir.exists() else []
        if not qp_files:
            print(f"[{i+1}/{len(tasks)}] SKIP {vid} — no query points")
            continue

        frame_groups, orig_dim = {}, None
        for qp_file in qp_files:
            m = re.search(r"_f(\d+)\.npz$", qp_file.name)
            if not m:
                continue
            try:
                qp = np.load(qp_file, allow_pickle=True)
            except (EOFError, ValueError) as e:
                print(f"  WARN corrupted {qp_file.name}: {e}")
                continue
            if orig_dim is None and "dim" in qp.files:
                orig_dim = qp["dim"]
            frame_groups.setdefault(int(m.group(1)), []).append(qp["query_points"])
        if orig_dim is None or not frame_groups:
            print(f"[{i+1}/{len(tasks)}] SKIP {vid} — query points empty/corrupt")
            continue

        H, W = int(orig_dim[0]), int(orig_dim[1])
        H480 = 480 if cfg["encode_480p"] else H
        W480 = (int(round(W * H480 / H / 2)) * 2) if cfg["encode_480p"] else W

        if len(frame_groups) > cfg["max_frame_groups"]:
            print(f"[{i+1}/{len(tasks)}] SKIP {vid} — {len(frame_groups)} frame groups "
                  f"> limit {cfg['max_frame_groups']}")
            continue

        frame_track_files = []
        for fidx in sorted(frame_groups):
            qps_orig = np.concatenate(frame_groups[fidx], axis=0)
            clip_name = f"{vid}_f{fidx}"
            track_file = paths.tracks_2d / vid / f"{clip_name}.npz"
            if track_file.exists():
                frame_track_files.append((fidx, track_file))
                continue
            qps_480, dim_480 = _scale_qps(qps_orig, orig_dim, H480, W480)
            qp_480_file = paths.tracks_2d / vid / f"qp_frame_{fidx}.npz"
            qp_480_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez(qp_480_file, query_points=qps_480, dim=dim_480)
            print(f"[{i+1}/{len(tasks)}] AllTracker {vid} f{fidx}: {len(qps_orig)} pts "
                  f"({W}x{H} -> {W480}x{H480})")
            sh(["python3", "run-query-points.py",
                "--file", vid, "--clip", clip_name,
                "--video_path", str(video_480p), "--query_path", str(qp_480_file.resolve()),
                "--out_root", str(paths.tracks_2d.resolve()),
                "--max_frames", "0", "--max_side", str(cfg["alltracker_max_side"])],
               cwd=str(alltracker_dir))
            if track_file.exists():
                frame_track_files.append((fidx, track_file))
            else:
                print(f"  ERROR: AllTracker failed for f{fidx}")

        merged_file = paths.tracks_2d / vid / f"{vid}_merged.npz"
        if frame_track_files and not merged_file.exists():
            all_t, all_v, dim = [], [], None
            for _, tp in sorted(frame_track_files):
                td = np.load(tp, allow_pickle=True)
                all_t.append(td["tracks"])
                all_v.append(td["visibility"])
                dim = dim if dim is not None else td["dim"]
            np.savez(merged_file,
                     tracks=np.concatenate(all_t, axis=1),
                     visibility=np.concatenate(all_v, axis=1), dim=dim)
            print(f"  merged -> {merged_file.name}")
    free_gpu()


# ──────────────────────────────────────────────────────────────────────────────
# Stage 4 — 3D lift
# ──────────────────────────────────────────────────────────────────────────────

def _lift_one(vid, cfg, paths):
    track_file = paths.tracks_2d / vid / f"{vid}_merged.npz"
    if not track_file.exists():
        return vid, False, "no merged 2D tracks"
    out_3d = paths.tracks_3d / cfg["corpus"] / f"{vid}_merged_3d_tracks.npz"
    out_3d.parent.mkdir(parents=True, exist_ok=True)
    if out_3d.exists():
        return vid, True, "cached"
    script = THIRD_PARTY / "vipe" / "scripts" / "vipe_to_colmap_general.py"
    r = sh(["python3", str(script), str(paths.vipe_results),
            "--sequence", f"{vid}_merged", "--video_stem", vid,
            "--track_path", str(track_file.resolve()),
            "--depth_step", str(cfg["depth_step"]),
            "--dataset_tag", cfg["corpus"], "--output", str(paths.tracks_3d)])
    ok = r.returncode == 0 and out_3d.exists()
    return vid, ok, "done" if ok else f"failed ({r.returncode})"


def stage4_lift(tasks, cfg, paths):
    print("=" * 80, "\nSTAGE 4: 3D lift\n", "=" * 80, sep="")
    with ThreadPoolExecutor(max_workers=min(16, len(tasks))) as pool:
        futs = {pool.submit(_lift_one, t["video_id"], cfg, paths): t["video_id"] for t in tasks}
        for fut in as_completed(futs):
            vid, ok, msg = fut.result()
            print(f"  [{'OK' if ok else 'WARN'}] {vid}: {msg}")


# ──────────────────────────────────────────────────────────────────────────────
# Stage 5 — filter + smooth
# ──────────────────────────────────────────────────────────────────────────────

def _filter_smooth_one(vid, cfg, paths):
    out_3d = paths.tracks_3d / cfg["corpus"] / f"{vid}_merged_3d_tracks.npz"
    track_2d = paths.tracks_2d / vid / f"{vid}_merged.npz"
    final_3d = paths.final_tracks / f"{vid}_3d.npz"
    final_2d = paths.final_tracks / f"{vid}_2d.npz"
    if not out_3d.exists():
        return vid, False, "no 3D tracks"
    if final_3d.exists() and final_2d.exists():
        return vid, True, "cached"

    # object_sizes = number of query points per grounded object (for sub-grouping)
    qp_dir = paths.grounding / vid / "query_points"
    sizes = [len(np.load(f, allow_pickle=True)["query_points"])
             for f in sorted(qp_dir.glob(f"{vid}_*_f*.npz"))] if qp_dir.exists() else []

    pose = paths.vipe_results / "pose" / f"{vid}.npz"
    depth = paths.vipe_results / "depth" / f"{vid}.zip"
    intr = paths.vipe_results / "intrinsics" / f"{vid}.npz"
    script = THIRD_PARTY / "vipe" / "track-filter-smooth.py"

    cmd = ["python3", str(script), "--vid", vid,
           "--src", str(out_3d), "--dst", str(final_3d),
           "--tracks_2d", str(track_2d), "--tracks_2d_out", str(final_2d),
           "--pose_npz", str(pose),
           "--smooth_steps", str(cfg["smooth_steps"]), "--smooth_lr", str(cfg["smooth_lr"]),
           "--alpha", str(cfg["alpha"]), "--lambda_reg", str(cfg["lambda_reg"]),
           "--z_thresh", str(cfg["z_thresh"]), "--n_anchors", str(cfg["n_anchors"]),
           "--gating_power", str(cfg["gating_power"])]
    # NOTE: mean-shift min_cluster_size is fixed to the paper default (10) inside
    # track-filter-smooth.py; it is not a CLI flag.
    if depth.exists() and intr.exists():
        cmd += ["--depth_zip", str(depth), "--intrinsics_npz", str(intr)]
    if sizes:
        cmd += ["--object_sizes", ",".join(map(str, sizes))]
    r = sh(cmd)
    ok = r.returncode == 0 and final_3d.exists()
    return vid, ok, "done" if ok else f"failed ({r.returncode})"


def stage5_filter_smooth(tasks, cfg, paths):
    print("=" * 80, "\nSTAGE 5: Filter + smooth\n", "=" * 80, sep="")
    with ThreadPoolExecutor(max_workers=min(16, len(tasks))) as pool:
        futs = {pool.submit(_filter_smooth_one, t["video_id"], cfg, paths): t["video_id"] for t in tasks}
        for fut in as_completed(futs):
            vid, ok, msg = fut.result()
            print(f"  [{'OK' if ok else 'WARN'}] {vid}: {msg}")


# ──────────────────────────────────────────────────────────────────────────────
# Stage 6 — clipping
# ──────────────────────────────────────────────────────────────────────────────

def stage6_clip(tasks, cfg, paths):
    print("=" * 80, "\nSTAGE 6: Video-level motion clipping\n", "=" * 80, sep="")
    vids = [t["video_id"] for t in tasks
            if (paths.final_tracks / f"{t['video_id']}_3d.npz").exists()]
    if not vids:
        print("  No final tracks to clip.")
        return
    out_json = paths.clips / f"{cfg['corpus']}_clips.json"
    script = REPO_ROOT / "pipeline" / "clip_segments.py"
    cmd = ["python3", str(script),
           "--final_tracks_dir", str(paths.final_tracks),
           "--video_ids", *vids,
           "--out_json", str(out_json),
           "--fps", str(cfg["fps"]),
           "--threshold", str(cfg["clip_threshold"]),
           "--min_gap", str(cfg["clip_min_gap"]),
           "--min_clip_sec", str(cfg["clip_min_clip_sec"])]
    if cfg["clip_min_frames"] is not None:
        cmd += ["--min_frames", str(cfg["clip_min_frames"])]
    sh(cmd)


# ──────────────────────────────────────────────────────────────────────────────

STAGES = {
    1: ("Grounding", stage1_grounding),
    2: ("ViPE depth+pose", stage2_vipe),
    3: ("AllTracker 2D", stage3_alltracker),
    4: ("3D lift", stage4_lift),
    5: ("Filter + smooth", stage5_filter_smooth),
    6: ("Clip", stage6_clip),
}


def main():
    ap = argparse.ArgumentParser(description="MolmoMotion-1M data-generation pipeline")
    ap.add_argument("--tasks", required=True, help="JSON list of {video_id, video_path, action}")
    ap.add_argument("--config", default=None, help="YAML config (see configs/)")
    ap.add_argument("--work_dir", required=True, help="Output directory for all artifacts")
    ap.add_argument("--start_stage", type=int, default=1)
    ap.add_argument("--end_stage", type=int, default=6)
    args = ap.parse_args()

    tasks = json.load(open(args.tasks))
    for t in tasks:
        if "action" not in t and "language_instruction" in t:
            t["action"] = t["language_instruction"]
    cfg = load_config(args.config)
    paths = Paths(Path(args.work_dir).resolve())

    print(f"\nMolmoMotion-1M data-gen: {len(tasks)} task(s), "
          f"stages {args.start_stage}-{args.end_stage}, work_dir={paths.work_dir}\n")

    t0 = time.time()
    for s in range(args.start_stage, args.end_stage + 1):
        name, fn = STAGES[s]
        ts = time.time()
        fn(tasks, cfg, paths)
        print(f"  >> Stage {s} ({name}) finished in {time.time() - ts:.1f}s\n")
    print(f"PIPELINE DONE in {time.time() - t0:.1f}s. Clips -> {paths.clips}\n")


if __name__ == "__main__":
    main()
