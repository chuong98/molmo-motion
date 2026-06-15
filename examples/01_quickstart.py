"""Minimal MolmoMotion quickstart — predict the next 30 frames of a tracked
object given:
  - H history RGB frames (H=3 for the H3 model, H=1 for the H1 model)
  - the same P points' 2D pixel coords at the query frame t_0
  - the same P points' 3D camera-frame XYZ across all H history frames
  - a one-sentence action caption

The bundled example clip in `examples/data/molmospaces_pick_place/` comes
from the MolmoSpaces subset of MolmoMotion-1M (sim pick-and-place; see
meta.json for provenance) and includes the ground-truth future trajectory
for comparison.

Run from a clean conda env after `pip install -e .`:

    python examples/01_quickstart.py

Output: an (8, 30, 3) tensor of camera-frame XYZ coordinates per
(point, future frame). XYZ in meters; origin = camera position at t_0;
+z = camera forward.
"""

import json
from pathlib import Path

import torch
from PIL import Image

from molmo_motion import MolmoMotion, MolmoMotionProcessor


# ────────────────────────────────────────────────────────────────────────
# 1. Load the pretrained model + processor (one-line, HF-style).
#    Two model variants: H=3 (typical video) and H=1 (single-keyframe).
# ────────────────────────────────────────────────────────────────────────

MODEL_NAME = "allenai/MolmoMotion-4B-H3-F30"   # or "allenai/MolmoMotion-4B-H1-F32"

processor = MolmoMotionProcessor.from_pretrained(MODEL_NAME)
model = MolmoMotion.from_pretrained(MODEL_NAME)
model._internal = model._internal.to(torch.bfloat16).cuda()  # 4B params
H = processor.config.history_size   # 3 for the H3 model; 1 for the H1 model


# ────────────────────────────────────────────────────────────────────────
# 2. Prepare an example input. The processor needs:
#    - history_frames: H PIL.Images, ordered earliest → query (t_0).
#    - points_2d_at_t0: (P, 2) pixel coords at t_0 — drive pointfeat
#      injection into SigLIP2 patches.
#    - points_3d_history: (H, P, 3) — same P points' 3D positions in
#      CAMERA-FRAME-AT-T_0 across all H history frames. If you have
#      world-frame coords + a camera-to-world matrix, pass `c2w_at_t0`
#      to let the processor do the transform.
#    - action: short caption.
#    - future_horizon: number of future frames to predict.
# ────────────────────────────────────────────────────────────────────────

EXAMPLE_DIR = Path(__file__).parent / "data" / "molmospaces_pick_place"
meta = json.loads((EXAMPLE_DIR / "meta.json").read_text())

history_frames = [
    Image.open(EXAMPLE_DIR / f"frame_t{i:+d}.jpg").convert("RGB")
    for i in range(-(H - 1), 1)            # for H=3: t-2, t-1, t_0; for H=1: t_0 only
]

points_2d_at_t0 = torch.load(EXAMPLE_DIR / "points_2d_at_t0.pt")
# shape: (P=8, 2) — pixel coords at t_0

# In this example the camera-frame coords are sim ground truth. They could
# equally come from a depth-sensor back-projection, MoCap, or any other source.
points_3d_history = torch.load(EXAMPLE_DIR / "points_3d_history.pt")
# shape: (H, P, 3) — meters, camera-frame at t_0

action = meta["action"]
future_horizon = 30


# ────────────────────────────────────────────────────────────────────────
# 3. Forward pass.
# ────────────────────────────────────────────────────────────────────────

inputs = processor(
    history_frames=history_frames,
    points_2d_at_t0=points_2d_at_t0,
    points_3d_history=points_3d_history,
    action=action,
    future_horizon=future_horizon,
)
# Keep non-tensor entries (e.g. `future_horizon`, `history_size`, `metadata`):
# they're consumed by `predict_trajectory` and dropping them silently falls
# back to defaults that mismatch the requested horizon.
inputs = {k: v.cuda() if torch.is_tensor(v) else v for k, v in inputs.items()}

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    out = model.predict_trajectory(**inputs)


# ────────────────────────────────────────────────────────────────────────
# 4. Inspect output.
# ────────────────────────────────────────────────────────────────────────

print(f"Predicted shape: {tuple(out.future_3d.shape)}  (expect (8, 30, 3))")
print(f"First point's predicted trajectory (m, camera-frame):")
for f in range(0, future_horizon, 4):
    x, y, z = out.future_3d[0, f].tolist()
    print(f"  t={f+1:2d}: ({x:+.3f}, {y:+.3f}, {z:+.3f})")

# The bundled example ships the sim ground-truth future for comparison.
gt_future = torch.load(EXAMPLE_DIR / "points_3d_future_gt.pt")  # (F, P, 3)
ade = (out.future_3d.cpu() - gt_future.permute(1, 0, 2)).norm(dim=-1).mean()
print(f"\nADE vs ground truth on this clip: {ade:.3f} m")

# Save predictions for downstream tools (e.g. visualizer):
torch.save(out.future_3d.cpu(), "trajectory_prediction.pt")
print("Saved (8, 30, 3) predictions to trajectory_prediction.pt")
print("Visualize with: python scripts/visualize_trajectory.py "
      "--frames examples/data/molmospaces_pick_place --pred trajectory_prediction.pt")
