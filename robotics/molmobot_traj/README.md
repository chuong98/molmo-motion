# MolmoBot trajectory-conditioning extension

Finetuning a MolmoBot policy from a MolmoMotion checkpoint needs
trajectory-conditioning extensions (3D-track loading + prompt-encoder mode) that
are **not** in public [`allenai/MolmoBot`](https://github.com/allenai/MolmoBot).
This directory carries those changes as a patch plus a small overlay, applied on
top of current MolmoBot `main`.

## Contents

- **`trajectory_conditioning.patch`** — edits 10 existing MolmoBot files
  (`launch_scripts/train_molmobot.py`, `olmo/data/synthmanip_{config,dataset}.py`,
  `olmo/models/molmobot/{molmobot,inference_wrapper}.py`,
  `olmo/models/video_olmo/video_preprocessor.py`, `olmo/nn/llm.py`,
  `olmo/train/trainer.py`, `olmo/train_init_utils.py`,
  `olmo/eval/configure_molmo_spaces.py`). Adds the `--load_3d_tracks`,
  `--prompt_style`, and `--prompt_encoder_mode` flags, the synthmanip 3D-track
  dataset path, and `can_predict_extra_tokens` schema-compat for MolmoMotion
  checkpoints.
- **`overlay/`** — 3 new modules, laid out under `olmo/` to mirror the MolmoBot
  package so they drop straight in: `olmo/data/droid_dataset.py`,
  `olmo/data/synthmanip_tracks_dataset.py`, `olmo/eval/mesh_surface_sampler.py`.

Both derive from `allenai/MolmoBot` (Apache-2.0).

## Apply

```bash
git clone https://github.com/allenai/MolmoBot
cd MolmoBot
git apply /path/to/robotics/molmobot_traj/trajectory_conditioning.patch
cp -r /path/to/robotics/molmobot_traj/overlay/olmo/* MolmoBot/olmo/
export MOLMOBOT_REPO="$PWD/MolmoBot"
```

## Environment

MolmoBot uses `uv`:

```bash
uv sync --extra train --python <python3.11>   # set UV_PYTHON_DOWNLOADS=never if the standalone CPython download is blocked
uv pip install "torchcodec==0.4.*"            # matches torch 2.7; not in the train extra
```

**Known issue — torchcodec / FFmpeg:** MolmoBot's video loader uses
`loading_method="torchcodec_exact"`. torchcodec 0.4 needs **FFmpeg ≤ 7** (not 8)
plus torch/NPP/NVRTC libs on the loader path. Working recipe:

```bash
conda install -c conda-forge 'ffmpeg=7'
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:\
$VENV/lib/python3.11/site-packages/torch/lib:\
$VENV/lib/python3.11/site-packages/nvidia/npp/lib:\
$VENV/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib
```

## Run (from `robotics/`)

```bash
# 1. unshard the MolmoMotion init checkpoint
python scripts/unshard_pretrained.py <molmomotion_run>/step<N> <unsharded_dst>

# 2. build the training view from the molmo-motion-1m molmospaces release
python scripts/prepare_training_data.py \
    --release_root /path/to/molmo-motion-1m/molmospaces --dst_root <train_view>

# 3. finetune
bash launch_scripts/train.sh configs/molmobot_pickplace.yaml
```

## Validation

- **2026-06-15** — re-verified against current MolmoBot `main`: the patch applies
  cleanly, the overlay modules drop in, and all 13 patched + overlay Python files
  byte-compile. The 10 patched files are unchanged on `main` since the original
  end-to-end run, so no rebase was needed.
- Original end-to-end smoke test (1×L40S): `prepare_training_data.py
  --release_root` builds the training view; `train_molmobot.py` builds the
  598M-param policy (578M VLM + 19.3M ActionExpert), loads the MolmoMotion init
  (reinitializing the action expert), and runs training steps with real
  flow-matching loss (`train/flow_loss_*`).
