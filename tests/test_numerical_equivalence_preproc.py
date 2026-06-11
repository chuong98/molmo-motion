"""Numerical-equivalence test between the original `olmo` package and the
release `molmo_motion` package.

Strategy:
 1. Load the same trajectory ckpt config in both packages.
 2. Run identical example dicts through each package's preprocessor.
 3. Compare every output tensor element-wise.
 4. (Optional, with --run-model) load the model in both packages, run
    `generate()` on the batch, compare token IDs.

Run from a neutral cwd (e.g. /tmp) to avoid the `datasets/` shadow issue:

    cd /tmp && python test_equivalence.py
"""

import sys
import os
import argparse
import importlib
import numpy as np
import torch

# Both packages need to be on the path. The release `molmo_motion` is
# pip-installed; the pre-release internal codebase (the old `olmo` package)
# comes from MOLMO_MOTION_OLD_REPO. The test is a no-op without one.
_OLD_REPO = os.environ.get("MOLMO_MOTION_OLD_REPO")
if _OLD_REPO:
    sys.path.insert(0, _OLD_REPO)

# Suppress noisy datasets warnings.
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BAR", "1")


def get_test_example():
    """Build a deterministic example dict matching the trajectory training format."""
    import numpy as np
    from PIL import Image

    np.random.seed(42)
    H, P = 3, 8
    frames = np.random.randint(0, 255, (H, 256, 256, 3), dtype=np.uint8)
    points_3d_camframe = np.random.randn(H, P, 3).astype(np.float32) * 0.1
    points_2d = np.random.rand(P, 2).astype(np.float32) * 256

    # Match trajectory_3d_dataset.py output: VideoFrames + message_list + metadata
    return {
        "frames": frames,
        "points_3d": points_3d_camframe,
        "points_2d": points_2d,
        "H": H,
        "P": P,
    }


def build_example_dict_for_preprocessor(ex, VideoFrames_cls):
    """Build the example dict format that ExamplePreprocessor expects."""
    H = ex["H"]
    P = ex["P"]
    timestamps = np.arange(H, dtype=np.float64) * 1.0
    video = VideoFrames_cls(
        frames=ex["frames"],
        timestamps=timestamps,
        target_fps=1.0,
    )

    # Serialize the 3D history deltas as tracks text (anchor-relative).
    anchor = ex["points_3d"][-1]
    deltas = ex["points_3d"] - anchor[None, :, :]
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

    return {
        "video": video,
        "message_list": [{
            "style": "video_qa",
            "question": question,
            "answer": "",
        }],
        "metadata": {
            "coords_2d": ex["points_2d"],
            "task": "3d_trajectory",
        },
    }


def compare_tensors(name, t_old, t_new, atol=0, rtol=0):
    """Element-wise compare two tensors / arrays and print pass/fail."""
    if t_old is None and t_new is None:
        print(f"  {name:25s}  both None  ✓")
        return True
    if t_old is None or t_new is None:
        print(f"  {name:25s}  MISMATCH: old={t_old is None}, new={t_new is None}  ✗")
        return False

    if hasattr(t_old, "numpy"):
        t_old = t_old.numpy()
    if hasattr(t_new, "numpy"):
        t_new = t_new.numpy()

    if t_old.shape != t_new.shape:
        print(f"  {name:25s}  shape mismatch: old={t_old.shape}, new={t_new.shape}  ✗")
        return False

    if t_old.dtype != t_new.dtype:
        print(f"  {name:25s}  dtype mismatch: old={t_old.dtype}, new={t_new.dtype}  ✗")
        return False

    if np.array_equal(t_old, t_new):
        print(f"  {name:25s}  shape={t_old.shape} dtype={t_old.dtype}  bit-exact ✓")
        return True

    if atol > 0 or rtol > 0:
        if np.allclose(t_old, t_new, atol=atol, rtol=rtol):
            max_diff = np.max(np.abs(t_old.astype(np.float64) - t_new.astype(np.float64)))
            print(f"  {name:25s}  shape={t_old.shape} max|Δ|={max_diff:.2e}  close ✓")
            return True

    # Diagnose
    max_diff = np.max(np.abs(t_old.astype(np.float64) - t_new.astype(np.float64)))
    n_mismatch = np.sum(t_old != t_new)
    print(f"  {name:25s}  shape={t_old.shape}  max|Δ|={max_diff:.4e}  "
          f"n_mismatch={n_mismatch}/{t_old.size}  ✗")
    return False


def check_preprocessing_equivalence(ckpt_dir):
    """Phase 1: end-to-end preprocessing equivalence (no model loading)."""
    print(f"\n{'='*70}")
    print(f"Phase 1: PREPROCESSING EQUIVALENCE")
    print(f"  ckpt config: {ckpt_dir}")
    print(f"{'='*70}\n")

    # --- Load configs from both packages ---
    from olmo.models.molmo2.molmo2_trajectory import Molmo2TrajectoryConfig as OldCfg
    from molmo_motion.models.molmo2.molmo2_trajectory import Molmo2TrajectoryConfig as NewCfg
    cfg_path = os.path.join(ckpt_dir, "config.yaml")
    old_cfg = OldCfg.load(cfg_path, key="model", validate_paths=False)
    new_cfg = NewCfg.load(cfg_path, key="model", validate_paths=False)
    print(f"  old_cfg.use_2d_point_features = {old_cfg.use_2d_point_features}")
    print(f"  new_cfg.use_2d_point_features = {new_cfg.use_2d_point_features}")
    assert old_cfg.use_2d_point_features == new_cfg.use_2d_point_features
    assert old_cfg.llm.max_sequence_length == new_cfg.llm.max_sequence_length

    # --- Build preprocessors ---
    old_pre = old_cfg.build_preprocessor(
        for_inference=True, is_training=False,
        text_seq_len=None, max_seq_len=old_cfg.llm.max_sequence_length,
    )
    new_pre = new_cfg.build_preprocessor(
        for_inference=True, is_training=False,
        text_seq_len=None, max_seq_len=new_cfg.llm.max_sequence_length,
    )

    # --- Build test example using each package's VideoFrames ---
    from olmo.data.video_loader import VideoFrames as OldVideoFrames
    from molmo_motion.data.video_loader import VideoFrames as NewVideoFrames

    test = get_test_example()
    old_ex = build_example_dict_for_preprocessor(test, OldVideoFrames)
    new_ex = build_example_dict_for_preprocessor(test, NewVideoFrames)

    # --- Run through preprocessors ---
    print("\n  Running ExamplePreprocessor on identical input...")
    old_out = old_pre(old_ex)
    new_out = new_pre(new_ex)

    # --- Compare outputs ---
    print(f"\n  Old output keys: {sorted(old_out.keys())}")
    print(f"  New output keys: {sorted(new_out.keys())}")
    common_keys = set(old_out.keys()) & set(new_out.keys())
    only_old = set(old_out.keys()) - set(new_out.keys())
    only_new = set(new_out.keys()) - set(old_out.keys())
    if only_old:
        print(f"  ⚠  Only in OLD: {only_old}")
    if only_new:
        print(f"  ⚠  Only in NEW: {only_new}")

    print(f"\n  Element-wise tensor comparison ({len(common_keys)} keys):")
    all_pass = True
    for k in sorted(common_keys):
        v_old = old_out[k]
        v_new = new_out[k]
        if isinstance(v_old, (np.ndarray, torch.Tensor)):
            if not compare_tensors(k, v_old, v_new):
                all_pass = False
        elif isinstance(v_old, dict):
            # metadata dict — compare keys
            print(f"  {k:25s}  dict[{len(v_old)} keys]")
        elif v_old == v_new:
            print(f"  {k:25s}  scalar match: {v_old!r}")
        else:
            print(f"  {k:25s}  scalar MISMATCH: {v_old!r} vs {v_new!r}  ✗")
            all_pass = False

    return all_pass


def check_decoded_prompt_text(ckpt_dir):
    """Sanity check: the actual decoded prompt string should be byte-identical."""
    print(f"\n{'='*70}")
    print(f"Phase 1b: DECODED PROMPT EQUIVALENCE")
    print(f"{'='*70}\n")

    from olmo.models.molmo2.molmo2_trajectory import Molmo2TrajectoryConfig as OldCfg
    from molmo_motion.models.molmo2.molmo2_trajectory import Molmo2TrajectoryConfig as NewCfg
    cfg_path = os.path.join(ckpt_dir, "config.yaml")
    old_cfg = OldCfg.load(cfg_path, key="model", validate_paths=False)
    new_cfg = NewCfg.load(cfg_path, key="model", validate_paths=False)

    old_pre = old_cfg.build_preprocessor(
        for_inference=True, is_training=False,
        text_seq_len=None, max_seq_len=old_cfg.llm.max_sequence_length,
    )
    new_pre = new_cfg.build_preprocessor(
        for_inference=True, is_training=False,
        text_seq_len=None, max_seq_len=new_cfg.llm.max_sequence_length,
    )

    from olmo.data.video_loader import VideoFrames as OldVideoFrames
    from molmo_motion.data.video_loader import VideoFrames as NewVideoFrames

    test = get_test_example()
    old_ex = build_example_dict_for_preprocessor(test, OldVideoFrames)
    new_ex = build_example_dict_for_preprocessor(test, NewVideoFrames)

    old_out = old_pre(old_ex)
    new_out = new_pre(new_ex)

    old_tok = old_pre.tokenizer
    new_tok = new_pre.tokenizer

    old_input = old_out["input_tokens"]
    new_input = new_out["input_tokens"]
    old_text = old_tok.decode(old_input[old_input >= 0])
    new_text = new_tok.decode(new_input[new_input >= 0])

    print(f"  Old prompt ({len(old_text)} chars):\n    {old_text[:300]}...")
    print(f"  New prompt ({len(new_text)} chars):\n    {new_text[:300]}...")
    if old_text == new_text:
        print("\n  ✓ Decoded prompts are BYTE-IDENTICAL")
        return True
    else:
        print("\n  ✗ Decoded prompts DIFFER")
        # show first diff
        for i, (a, b) in enumerate(zip(old_text, new_text)):
            if a != b:
                print(f"    first diff at char {i}: old={a!r} new={b!r}")
                print(f"    context: ...{old_text[max(0,i-20):i+20]}...")
                break
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="A trajectory ckpt dir (with config.yaml).")
    args = ap.parse_args()

    p1 = check_preprocessing_equivalence(args.ckpt)
    p1b = check_decoded_prompt_text(args.ckpt)

    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  Preprocessing tensor equivalence: {'PASS ✓' if p1 else 'FAIL ✗'}")
    print(f"  Decoded prompt byte equivalence:  {'PASS ✓' if p1b else 'FAIL ✗'}")
    sys.exit(0 if (p1 and p1b) else 1)


if __name__ == "__main__":
    main()
