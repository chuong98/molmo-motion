# MolmoMotion B-spline Control-Point Trajectory Plan

## Context

MolmoMotion currently predicts the **full future trajectory** as text: for `F=32`
frames × `P=8` points it emits a `<tracks coords="...">` answer of **~1000–1200
tokens** (one row per frame: `TS OBJ_ID DX DY DZ ...`, quantized ×1000). This is
expensive in memory/sequence length and hard to learn — the model must emit every
frame explicitly and stay self-consistent across ~1000 autoregressive steps.

**Objective:** add an opt-in mode where the answer encodes only **`D=10` cubic
B-spline control points per point** (mu0-style) instead of `F=32` frames. Ten
control points reconstruct the full smooth trajectory via a fixed basis, cutting
the answer to **~330 tokens (~70% fewer)** and giving the model a smoother,
lower-dimensional, easier target. The frame-based format stays the default; the
B-spline format is selected by a dataset-name suffix `_ck{D}` + a config flag.

**Decisions (locked):**
1. Implement in **`molmo-motion/` directly** (dataset + eval + config + public API).
2. Keep the **autoregressive text format** — emit control points as `<tracks>` rows;
   tokenizer / loss (next-token CE) / generation are unchanged.
3. **Opt-in variant** — `bspline_n_ctrl` config + `_ck{D}` dataset-name token; the
   32-frame format remains the default for A/B comparison.

## Key existing code (from exploration)

| What | File | Anchor |
|---|---|---|
| Trajectory serialization | `src/molmo_motion/data/trajectory_3d_dataset.py` | `_format_tracks_text()` (~L1839–1863), `_quantize()` (~L1835), `_build_example()` (~L1867–2139) |
| Anchor + delta encode | same | anchor = point 0 at frame `t`=H−1 (~L1914–1919); `traj_delta = traj_clean - anchor` |
| Dataset-name grammar (P/H/F regex) | `src/molmo_motion/data/get_dataset.py` | `_p _h _f` regex (~L60–65), `Trajectory3DDataset(num_future_frames=...)` (~L53–93) |
| Loss (next-token CE on answer) | `src/molmo_motion/train/trainer.py` + `preprocessing/text_preprocessor.py` | answer tokens `is_model=True` (~L82–109) |
| Decode | `src/molmo_motion/eval/egodex_3d_evaluator.py` | `parse_tracks_text()` (~L23–67), `tracks_to_array()` (~L70–98), add anchor (~L186) |
| Rollout driver | `src/molmo_motion/eval/full_rollout.py` | builds prompt, calls generate, decodes |
| Model config | `src/molmo_motion/models/molmo2/molmo2_trajectory.py` | `Molmo2TrajectoryConfig` (~L32–43) |
| Public inference | `src/molmo_motion/modeling.py`, `processor.py`, `public_config.py` | `predict_trajectory`, prompt build |

## B-spline math (port from mu0, no lerobot dependency)

Reference: `mu0/src/lerobot/datasets/bspline_basis.py` and `trace_dataset.py:932–984`.

- **Basis:** cubic, `n_ctrl ∈ {4,7,10}`, precomputed knot vectors, partition-of-unity.
  `build_bspline_basis(n_ctrl, horizon, include_anchor)` → `(H+1, D)` (fit) or `(H, D)` (render).
- **Fit (per point), in MolmoMotion's shared-anchor delta space:**
  target = `[t0_delta ; F future deltas]` `(F+1, 3)`; validity mask `M (F+1,)` with the
  t=0 anchor row always valid; weighted lstsq `A_w = basis * M[:,None]`, `B_w = target * M[:,None]`;
  optional Tikhonov (`λ·Γ`, order-1/2) and hard clip → `P_ctrl (D, 3)`.
  A point is **valid** iff it has `≥ D` valid future frames (else dropped from the answer).
- **Render (eval/inference):** `traj = einsum("hd,pdc->phc", basis_render, P_ctrl)` with
  `basis_render = build_bspline_basis(D, F, include_anchor=False)` → `(F, 3)` per point,
  then add the single shared `gt_anchor` (unchanged decode step).

Fit uses `include_anchor=True` (spline passes through the point's `t0` position);
render uses `include_anchor=False` (exactly the `F` future frames). Same control points.

## Text format for control-point mode

Reuse `_format_tracks_text(timestamps, points_delta, visibility, label)` unchanged:
- `timestamps` → control-point indices `[0.0, 1.0, …, D-1.0]`
- `points_delta` → `(P, D, 3)` control points (quantized ×1000, shared-anchor space)
- `visibility` → `(P, D)` (all-D-valid or all-invalid per point)
- `label` → **`"3d object control points"`** (distinguishes the two modes at decode)

Example answer (`D=10`): `<tracks coords="0.0 1 DX DY DZ 2 …;1.0 …;…;9.0 …">3d object control points</tracks>`

Prompt wording (bspline branch): *"Predict the {D} B-spline control points of {P}
points over a {F}-frame horizon, given action: "…", and history 3d point
coordinates: "…"."*

---

## Tasks

### Task 1 — B-spline utility module (standalone)
**File:** create `src/molmo_motion/data/bspline.py`
- Port `build_bspline_basis(n_ctrl, horizon, *, include_anchor=True)` and the
  finite-difference regularization matrix builder from mu0 (numpy or torch; numpy is
  fine since the dataset is numpy-side).
- Add `fit_control_points(future_delta, valid, n_ctrl, reg_lambda=0.0, reg_order=1, clip=None)`
  → `(ctrl (P,D,3), valid_kp (P,))` implementing the weighted lstsq + anchor row +
  Tikhonov + clip + `≥D valid frames` rule.
- Add `render_control_points(ctrl, horizon)` → `(P, F, 3)`.
- **Test:** `tests/test_bspline.py` — basis rows sum to 1; fit→render of a known smooth
  cubic recovers it within ~1e-3; a straight line is reconstructed near-exactly.

### Task 2 — Dataset: emit control points (opt-in)
**File:** `src/molmo_motion/data/trajectory_3d_dataset.py`
- `__init__`: accept `bspline_n_ctrl:int=0`, `bspline_reg_lambda`, `bspline_reg_order`,
  `bspline_ctrl_clip`. Cache the fit/render bases when `bspline_n_ctrl>0`.
- In `_build_example`, when `bspline_n_ctrl>0`: after computing `traj_delta` and the
  future validity mask, call `fit_control_points` on the future block
  `traj_delta[:, H:H+F]`, build `(P,D)` visibility from `valid_kp`, and pass indices
  `0..D-1` + `(P,D,3)` control points to `_format_tracks_text` with label
  `"3d object control points"`. Use the bspline prompt wording.
- Store in `metadata`: `bspline_n_ctrl=D`, `future_horizon=F`, keep `gt_anchor` as today.
- History/prompt side stays frame-based (unchanged).
- **Test:** build one example in bspline mode; assert answer has `D` rows, label matches,
  and `parse→render+anchor` reconstructs the GT future within (quantization + fit) tol.

### Task 3 — Dataset-name grammar
**File:** `src/molmo_motion/data/get_dataset.py`
- Parse `_ck(\d+)` → `bspline_n_ctrl` (0 if absent). Optionally `_creg…` knobs, else defaults.
- Thread the new kwargs into `Trajectory3DDataset(...)`. `F` still comes from `_f{F}`
  (it is the render horizon). Example: `trajectory_3d_human_p8_h3_f32_ck10`.
- **Test:** parsing `..._f32_ck10` yields `future=32, bspline_n_ctrl=10`; no `_ck` → 0.

### Task 4 — Eval decode: control points → (P, F, 3)
**File:** `src/molmo_motion/eval/egodex_3d_evaluator.py`
- Add `tracks_to_control_points(parsed, P, D)` → `(P, D, 3)` + `(P, D)` valid (rows keyed by
  index `0..D-1`, not frame timestamps).
- Add `control_points_to_traj(ctrl, F)` using `bspline.render_control_points`.
- Branch on mode (label `"3d object control points"` OR `metadata["bspline_n_ctrl"]>0`):
  parse `D` control points → render to `(P, F, 3)` → add anchor. Frame mode path unchanged.
  Metrics (ADE/FDE/PWT) are computed on `(P, F, 3)` and need **no change**.
- **Test:** feed a synthetic control-point `<tracks>` string; assert rendered `(P,F,3)`
  matches `render_control_points` + anchor.

### Task 5 — Rollout driver wiring
**File:** `src/molmo_motion/eval/full_rollout.py`
- Read `bspline_n_ctrl` / `future_horizon` from the model/config or eval args; select the
  control-point decode path; leading numbers are indices `0..D-1` (not `H..H+F−1`).
- Reduce `max_new_tokens` for the (much shorter) control-point answer.
- One-shot only (no multi-rollout chunking needed at F=32 since one answer covers the horizon).

### Task 6 — Config persistence + public inference
**Files:** `models/molmo2/molmo2_trajectory.py`, `public_config.py`, `processor.py`, `modeling.py`
- Add `bspline_n_ctrl` (and reg/clip) to `Molmo2TrajectoryConfig` and the public
  `MolmoMotionConfig` so a trained checkpoint records the mode + `D` + `F`.
- `processor.py`: in bspline mode build the bspline prompt wording; still emit frame-based
  **history** in the prompt.
- `modeling.py` `predict_trajectory`: when the config says bspline, parse `D` control points
  from the generated text and render → `future_3d (P, F, 3)` (+ anchor). Return same
  `MolmoMotionOutput` shape as today (`.future_3d` is `(P, F, 3)`), so downstream/viz is unchanged.
- **Test:** `save_pretrained`→`from_pretrained` round-trips `bspline_n_ctrl`.

### Task 7 — Training recipe + docs
**Files:** `launch_scripts/sft.py` usage docs, `README.md`, this plan
- Document the new recipe name `trajectory_3d_human_p8_h3_f32_ck10`; recommend finetuning
  from the existing Stage-1 checkpoint (the numeric semantics change, so re-train the head/LLM
  on the new answer distribution rather than expecting zero-shot transfer).

---

## Verification

1. **Reconstruction ceiling (no model):** for a sample of GT trajectories, fit `D=10`
   control points and render back; report ADE/FDE of the *fit itself* vs GT. This is the
   best achievable error in bspline mode — sanity-check it is well below current model ADE.
2. **Round-trip test:** dataset `_build_example` (bspline) → `parse_tracks_text` →
   `control_points_to_traj` + anchor ≈ GT future (quantization + fit tol).
3. **Token budget:** log answer token count for `_f32` vs `_f32_ck10` (expect ~1000 → ~330).
4. **A/B train:** short runs of `trajectory_3d_human_p8_h3_f32` vs `..._ck10` from the same
   Stage-1 init; compare PointMotionBench ADE/FDE/PWT, peak memory, and tokens/step.
5. **Unit tests** from Tasks 1–4, 6 all pass: `python -m pytest tests/ -k "bspline or tracks"`.

## Out of scope (v1)

- Continuous regression head (kept AR text per decision 2).
- Per-axis `delta_scale` normalization (mu0 uses it for FM; here ×1000 int quantization on
  clipped meter-space control points is sufficient — revisit only if tokens get too large).
- Changing the history/prompt encoding (stays frame-based).
