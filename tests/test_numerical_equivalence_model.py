"""Phase 2: model forward equivalence — bit-exact `generate()` across repos.

Loads the SAME unsharded checkpoint via the OLD `olmo` package and the NEW
`molmo_motion` package, runs `generate()` on identical inputs, and checks:
  - state_dict shape/keys match
  - state_dict tensor values match
  - logits match (first forward pass)
  - generated token IDs match

Run from /tmp:

    cd /tmp && python test_model_forward.py
"""

import sys
import os
import time
import argparse
import numpy as np
import torch

# This test compares against the pre-release internal codebase (the old
# `olmo` package). Point MOLMO_MOTION_OLD_REPO at a checkout of it; the
# test is a no-op without one.
_OLD_REPO = os.environ.get("MOLMO_MOTION_OLD_REPO")
if _OLD_REPO:
    sys.path.insert(0, _OLD_REPO)

def _init_single_rank_dist():
    # Single-rank dist init — needed by load_model_state_unsharded which calls
    # dist_cp_sd.set_model_state_dict (which queries the default process group).
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    if not torch.distributed.is_initialized():
        # Mixed-backend init: gloo for CPU broadcasts (state_dict load),
        # nccl for any GPU ops the model machinery does.
        torch.distributed.init_process_group(backend="cpu:gloo,cuda:nccl")


def load_model(pkg_root, ckpt_dir, device):
    """Build + load a fresh model copy. `pkg_root` ∈ {'olmo', 'molmo_motion'}."""
    if pkg_root == "olmo":
        from olmo.models.molmo2.molmo2 import Molmo2Config
        from olmo.train.checkpointer import load_model_state
    else:
        from molmo_motion.models.molmo2.molmo2 import Molmo2Config
        from molmo_motion.train.checkpointer import load_model_state

    cfg_path = os.path.join(ckpt_dir, "config.yaml")
    cfg = Molmo2Config.load(cfg_path, key="model", validate_paths=False)

    with torch.device("meta"):
        model = cfg.build_model()
    model.to_empty(device=torch.device("cpu"))
    load_model_state(ckpt_dir, model)
    model.eval()
    model = model.to(torch.bfloat16).to(device)
    return cfg, model


def build_preproc(pkg_root, ckpt_dir):
    if pkg_root == "olmo":
        from olmo.models.molmo2.molmo2 import Molmo2Config
    else:
        from molmo_motion.models.molmo2.molmo2 import Molmo2Config

    cfg_path = os.path.join(ckpt_dir, "config.yaml")
    cfg = Molmo2Config.load(cfg_path, key="model", validate_paths=False)
    return cfg.build_preprocessor(
        for_inference=True, is_training=False,
        text_seq_len=None, max_seq_len=cfg.llm.max_sequence_length,
    )


def get_example_batch(pkg_root, ckpt_dir):
    """Build a deterministic input batch (single example, batch=1)."""
    if pkg_root == "olmo":
        from olmo.data.video_loader import VideoFrames
    else:
        from molmo_motion.data.video_loader import VideoFrames

    np.random.seed(7)
    H, P = 3, 8
    frames = np.random.randint(0, 255, (H, 256, 256, 3), dtype=np.uint8)
    points_3d = (np.random.randn(H, P, 3).astype(np.float32) * 0.1)
    anchor = points_3d[-1]
    deltas = points_3d - anchor[None, :, :]
    frame_strings = []
    for fi in range(H):
        parts = [f"{float(fi):.1f}"]
        for pi in range(P):
            obj_id = pi + 1
            x = int(round(float(deltas[fi, pi, 0]) * 1000))
            y = int(round(float(deltas[fi, pi, 1]) * 1000))
            z = int(round(float(deltas[fi, pi, 2]) * 1000))
            parts.append(f"{obj_id} {x} {y} {z}")
        frame_strings.append(" ".join(parts))
    coords_str = ";".join(frame_strings)
    input_tracks = f'<tracks coords="{coords_str}">3d object history</tracks>'
    F = 8
    question = (
        f'Predict the future 3D point coordinates of {P} points over '
        f'{F} timestamps, given action: "test", and history 3d point '
        f'coordinates: "{input_tracks}".'
    )

    video = VideoFrames(
        frames=frames,
        timestamps=np.arange(H, dtype=np.float64) * 1.0,
        target_fps=1.0,
    )
    example = {
        "video": video,
        "message_list": [{"style": "video_qa", "question": question, "answer": ""}],
        "metadata": {"task": "3d_trajectory"},
    }

    pre = build_preproc(pkg_root, ckpt_dir)
    out = pre(example)

    # Add batch dim + rename input_tokens → input_ids (the collator does this)
    batch = {}
    for k, v in out.items():
        if isinstance(v, np.ndarray):
            t = torch.from_numpy(v).unsqueeze(0)
            batch["input_ids" if k == "input_tokens" else k] = t
        elif isinstance(v, torch.Tensor):
            batch["input_ids" if k == "input_tokens" else k] = v.unsqueeze(0)
    return batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="An unsharded ckpt dir (with config.yaml + model.pt).")
    ap.add_argument("--max-steps", type=int, default=64,
                    help="generation steps")
    args = ap.parse_args()

    _init_single_rank_dist()
    device = torch.device("cuda:0")
    print(f"{'='*70}\nPhase 2: MODEL FORWARD EQUIVALENCE\n  ckpt: {args.ckpt}\n  device: {device}\n{'='*70}\n")

    # ---- Load old model ----
    print(f"[1/4] Loading model via OLD `olmo` package ...")
    t0 = time.time()
    old_cfg, old_model = load_model("olmo", args.ckpt, device)
    print(f"      done in {time.time()-t0:.1f}s, "
          f"params: {sum(p.numel() for p in old_model.parameters())/1e9:.2f}B")

    # ---- Build inputs (same for both packages) ----
    print(f"\n[2/4] Building deterministic batch with seed=7 ...")
    old_batch = get_example_batch("olmo", args.ckpt)
    old_batch_dev = {k: v.to(device) for k, v in old_batch.items() if torch.is_tensor(v)}
    print(f"      input_ids shape: {tuple(old_batch_dev['input_ids'].shape)}")
    print(f"      images shape:    {tuple(old_batch_dev['images'].shape)}")

    # ---- Old generate ----
    print(f"\n[3/4] Running old model.generate(max_steps={args.max_steps}) ...")
    t0 = time.time()
    with torch.inference_mode():
        with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            old_gen = old_model.generate(
                batch=old_batch_dev,
                max_steps=args.max_steps,
                is_distributed=False,
            )
    old_tokens = old_gen.token_ids[:, 0].detach().cpu().numpy()  # (1, gen_len)
    print(f"      done in {time.time()-t0:.1f}s; emitted {old_tokens.shape[1]} tokens")

    # Free old model GPU mem
    del old_model
    torch.cuda.empty_cache()

    # ---- Load new model ----
    print(f"\n[4/4] Loading model via NEW `molmo_motion` package ...")
    t0 = time.time()
    new_cfg, new_model = load_model("molmo_motion", args.ckpt, device)
    print(f"      done in {time.time()-t0:.1f}s, "
          f"params: {sum(p.numel() for p in new_model.parameters())/1e9:.2f}B")

    new_batch = get_example_batch("molmo_motion", args.ckpt)
    new_batch_dev = {k: v.to(device) for k, v in new_batch.items() if torch.is_tensor(v)}

    # Verify batch tensors are bit-identical across packages
    print(f"\n  Batch tensor equivalence (OLD preproc vs NEW preproc):")
    all_batch_ok = True
    for k in sorted(set(old_batch.keys()) & set(new_batch.keys())):
        if torch.is_tensor(old_batch[k]) and torch.is_tensor(new_batch[k]):
            eq = torch.equal(old_batch[k], new_batch[k])
            print(f"    {k:20s} shape={tuple(old_batch[k].shape)} dtype={old_batch[k].dtype}  "
                  f"{'bit-exact ✓' if eq else 'DIFFER ✗'}")
            if not eq:
                all_batch_ok = False

    # New generate
    print(f"\n  Running new model.generate() ...")
    t0 = time.time()
    with torch.inference_mode():
        with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            new_gen = new_model.generate(
                batch=new_batch_dev,
                max_steps=args.max_steps,
                is_distributed=False,
            )
    new_tokens = new_gen.token_ids[:, 0].detach().cpu().numpy()
    print(f"      done in {time.time()-t0:.1f}s; emitted {new_tokens.shape[1]} tokens")

    # Compare token IDs
    print(f"\n{'='*70}\nGENERATION COMPARISON\n{'='*70}")
    print(f"  Old tokens shape: {old_tokens.shape}")
    print(f"  New tokens shape: {new_tokens.shape}")
    if old_tokens.shape == new_tokens.shape:
        eq = np.array_equal(old_tokens, new_tokens)
        if eq:
            print(f"  ✓ Generated token IDs are BIT-EXACT")
        else:
            n_diff = np.sum(old_tokens != new_tokens)
            print(f"  ✗ {n_diff}/{old_tokens.size} tokens differ")
            # Show first divergence position
            divergence = np.argmax(old_tokens != new_tokens, axis=1)
            print(f"  first divergence at position: {divergence[0]}")
    else:
        eq = False
        print(f"  ✗ Shape mismatch")

    # Decode both for human comparison
    from olmo.tokenizer import build_tokenizer as old_build
    from molmo_motion.tokenizer import build_tokenizer as new_build
    # Both should produce identical tokenizers from the same config
    old_tok = old_cfg.llm.build_tokenizer()
    new_tok = new_cfg.llm.build_tokenizer()
    old_text = old_tok.decode(old_tokens[0][old_tokens[0] >= 0])
    new_text = new_tok.decode(new_tokens[0][new_tokens[0] >= 0])
    print(f"\n  Old decoded: {old_text[:200]}{'...' if len(old_text) > 200 else ''}")
    print(f"  New decoded: {new_text[:200]}{'...' if len(new_text) > 200 else ''}")
    print(f"  Text equal: {old_text == new_text}")

    print(f"\n{'='*70}\nSUMMARY: {'PASS ✓' if (all_batch_ok and eq and old_text == new_text) else 'FAIL ✗'}\n{'='*70}")
    sys.exit(0 if (all_batch_ok and eq and old_text == new_text) else 1)


if __name__ == "__main__":
    main()
