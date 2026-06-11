#!/usr/bin/env python3
"""
Stage 1 — Semantic object grounding + query-point sampling.

This worker loads all Stage-1 models ONCE and processes every task in order:

    Qwen3-0.6B      object-phrase extraction from the action description
    Molmo2-8B       (optional) visual re-caption when the description is vague
                    or absent
    MolmoPoint-Vid  localize the object as a 2D point in the anchor frame
    SAM 3           segment the object from that point prompt
    K-means         sample N query points on the object mask

It writes one query-point NPZ per object per detection frame under
    <grounding_dir>/<video_id>/query_points/<video_id>_*_f*.npz
with schema  query_points:(N,3)[frame_idx,x,y]  dim:(2,)[H,W].

The worker is corpus-agnostic: the grounding/pointing prompt and whether to
re-caption are controlled entirely by the JSON config (see configs/*.yaml).

Models are frozen, publicly-released checkpoints pulled from HuggingFace:
    allenai/MolmoPoint-Vid-4B,  allenai/Molmo2-8B,  Qwen/Qwen3-0.6B,  facebook/sam3

Usage (invoked by run_pipeline.py; can also be run directly):
    python pipeline/grounding_worker.py <tasks_json> <prompts_json_out> \
        <grounding_dir> <config_json>

Resumability: tasks whose query-point NPZs already exist are skipped at startup,
so the parent can restart this worker after a crash (SAM 3 can SIGSEGV on rare
inputs) and it resumes from where it left off.
"""
import os

# Must be set before numpy / OpenBLAS import to avoid a thread-explosion crash
# on many-core nodes (OpenBLAS terminates with SIGSEGV when it spawns >128 threads).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import gc
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers.video_utils import VideoMetadata

# Make the vendored grounding code importable: third_party/sam3 holds both the
# top-level helpers (molmo2_pointing.py, querypoints_from_video.py) and the
# `sam3` package.
REPO_ROOT = Path(__file__).resolve().parent.parent
SAM3_DIR = REPO_ROOT / "third_party" / "sam3"
sys.path.insert(0, str(SAM3_DIR))


def atomic_json_save(data, path):
    """Write JSON atomically (temp file + rename) so a kill mid-write can't corrupt it."""
    path = str(path)
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)  # atomic on the same filesystem
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# Verbatim re-caption prompt from the paper (App. C.2). Format-only: the original
# text is never shown to the model, so the description is purely visual.
RECAPTION_PROMPT = (
    "Watch this video carefully. Describe the manipulation action you observe "
    "in exactly this format: [action verb] [specific object with color/material/shape] "
    "[preposition and location if present]. "
    'Examples: "pick up red ceramic coffee mug", '
    '"place blue plastic bottle on table", '
    '"insert metal key into lock". '
    "Be specific about the object — include its color, material, and shape as you see them. "
    "Output only the short description, no extra words."
)


def recaption_with_molmo2_8b_video(model, processor, video_frames: np.ndarray, fps: float) -> str:
    """Generate a fresh visual action caption with Molmo2-8B in video mode."""
    total_frames = len(video_frames)
    h, w = video_frames.shape[1], video_frames.shape[2]
    video_meta = VideoMetadata(
        total_num_frames=total_frames, fps=fps, width=w, height=h,
        duration=total_frames / fps,
    )
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": "placeholder"},
            {"type": "text", "text": RECAPTION_PROMPT},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        videos=[video_frames], text=text, return_tensors="pt",
        videos_kwargs={"video_metadata": video_meta, "return_metadata": True},
    )
    device = next(model.parameters()).device
    inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    out = generated_ids[0, inputs["input_ids"].size(1):]
    result = processor.tokenizer.decode(out, skip_special_tokens=True).strip()
    return result.rstrip(".,;:").replace("[", "").replace("]", "").strip()


def main():
    if len(sys.argv) != 5:
        print("Usage: python grounding_worker.py <tasks_json> <prompts_json_out> "
              "<grounding_dir> <config_json>", file=sys.stderr)
        sys.exit(1)

    tasks_path = sys.argv[1]
    prompts_out_path = Path(sys.argv[2])
    grounding_dir = Path(sys.argv[3])
    config_path = sys.argv[4]

    tasks = json.load(open(tasks_path))
    config = json.load(open(config_path))

    kmeans_k = config["kmeans_k"]              # N query points per object (paper: 100)
    fps = config["fps"]
    max_frames = config["max_frames"]
    agent = config.get("agent", "hand")         # "hand" | "robot gripper"
    # Pointing prompt. {obj} and {agent} are filled per object. In-the-wild corpora
    # use the tracking-mode template "track the {obj}".
    point_prompt_template = config.get(
        "point_prompt_template",
        "point to {obj} gripped and picked up by the {agent}",
    )
    allow_recaption = config.get("recaption", True)

    # Resume: figure out which tasks still need processing before loading any models.
    episode_prompts = json.load(open(prompts_out_path)) if prompts_out_path.exists() else {}
    pending = []
    for task in tasks:
        vid = task["video_id"]
        qp_dir = grounding_dir / vid / "query_points"
        existing = list(qp_dir.glob(f"{vid}_*_f*.npz")) if qp_dir.exists() else []
        if existing:
            print(f"  SKIP {vid} — {len(existing)} query-point NPZ files exist")
        else:
            pending.append(task)

    if not pending:
        print("[grounding] All tasks already grounded.")
        sys.exit(0)

    print(f"\n[grounding] {len(pending)} task(s) to process. Loading models...")
    from molmo2_pointing import (
        load_qwen3, load_molmo2_8b, load_molmo2,
        extract_object_with_qwen3, is_vague_object,
    )
    from sam3.model_builder import build_sam3_video_predictor, build_sam3_video_model
    from querypoints_from_video import process_one_video_with_molmo2

    print("  Loading Qwen3-0.6B...")
    qwen3_model, qwen3_tokenizer = load_qwen3()
    molmo2_8b_model = molmo2_8b_processor = None
    if allow_recaption:
        print("  Loading Molmo2-8B (re-caption)...")
        molmo2_8b_model, molmo2_8b_processor = load_molmo2_8b()
    print("  Loading SAM 3 video predictor + tracker...")
    video_predictor = build_sam3_video_predictor()
    sam3_model = build_sam3_video_model()
    sam3_tracker = sam3_model.tracker
    sam3_tracker.backbone = sam3_model.detector.backbone
    print("  Loading MolmoPoint-Vid-4B...")
    molmo2_vp_model, molmo2_vp_processor = load_molmo2()
    print("  All models loaded.\n")

    for i, task in enumerate(pending):
        vid = task["video_id"]
        video_path = task["video_path"]
        instruction = task.get("action", task.get("language_instruction", ""))
        has_annotation = bool(instruction) and task.get("has_annotation", True)
        print(f"\n[{i+1}/{len(pending)}] {vid}")

        # ── Phase A: object-phrase extraction ────────────────────────────────
        qwen3_obj = None
        molmo2_8b_caption = None
        molmo2_8b_obj = None
        if has_annotation:
            print(f"  Action: {instruction}")
            qwen3_obj = extract_object_with_qwen3(
                qwen3_model, qwen3_tokenizer, instruction, agent=agent)
            print(f"  Qwen3 object: '{qwen3_obj}'")

        need_recaption = allow_recaption and ((not has_annotation) or is_vague_object(qwen3_obj))
        if need_recaption:
            if has_annotation:
                print(f"  '{qwen3_obj}' is vague — re-captioning with Molmo2-8B...")
            cap = cv2.VideoCapture(video_path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            if frames:
                molmo2_8b_caption = recaption_with_molmo2_8b_video(
                    molmo2_8b_model, molmo2_8b_processor,
                    np.array(frames, dtype=np.uint8), fps)
                molmo2_8b_obj = extract_object_with_qwen3(
                    qwen3_model, qwen3_tokenizer, molmo2_8b_caption, agent=agent)
                print(f"  Molmo2-8B → Qwen3: '{molmo2_8b_obj}'")

        final_obj = molmo2_8b_obj or qwen3_obj or ""
        per_object_list = [o.strip() for o in final_obj.split("|") if o.strip()]
        if not per_object_list:
            print(f"  WARNING: could not determine object for {vid} — skipping")
            episode_prompts[vid] = {"skip": True}
            atomic_json_save(episode_prompts, prompts_out_path)
            continue

        per_object_prompts = [
            point_prompt_template.format(obj=obj, agent=agent) for obj in per_object_list
        ]
        print(f"  per_object_list: {per_object_list}")
        print(f"  point prompt[0]: '{per_object_prompts[0]}'")

        # ── Phase B: MolmoPoint + SAM 3 + K-means → query points ─────────────
        qp_root = grounding_dir / vid
        args_ns = argparse.Namespace(
            query_dir=qp_root / "query_points",
            masks_dir=qp_root / "masks",
            debug_dir=qp_root / "debug_png",
            kmeans_k=kmeans_k,
            kmeans_seed=0,
            max_frames=max_frames,
            include_hands=False,
            save_masks=config.get("save_masks", False),
            debug=config.get("save_debug", False),
        )
        qp_root.mkdir(parents=True, exist_ok=True)
        try:
            process_one_video_with_molmo2(
                video_predictor=video_predictor,
                molmo2_model=molmo2_vp_model,
                molmo2_processor=molmo2_vp_processor,
                video_path=video_path,
                args=args_ns,
                video_id=vid,
                text_prompts=None,
                instruction=instruction,
                fps=fps,
                per_object_pairs=list(zip(per_object_prompts, per_object_list)),
                sam3_tracker=sam3_tracker,
            )
        except (ValueError, RuntimeError) as e:
            print(f"  WARNING: skipping {vid} due to video error: {e}")
            episode_prompts[vid] = {"skip": True, "error": str(e)}
            atomic_json_save(episode_prompts, prompts_out_path)
            continue

        print(f"  DONE {vid}")
        episode_prompts[vid] = {
            "skip": False,
            "action": instruction,
            "qwen3_obj": qwen3_obj,
            "molmo2_8b_caption": molmo2_8b_caption,
            "molmo2_8b_obj": molmo2_8b_obj,
            "final_obj": final_obj,
            "per_object_list": per_object_list,
            "per_object_prompts": per_object_prompts,
        }
        atomic_json_save(episode_prompts, prompts_out_path)
        gc.collect()
        torch.cuda.empty_cache()

    print("\n[grounding] Done.")
    sys.exit(0)


if __name__ == "__main__":
    main()
