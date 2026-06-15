# MolmoBot trajectory-conditioning patch (for robotics finetuning)

MolmoBot finetuning from a MolmoMotion checkpoint needs trajectory-conditioning
extensions (3D-track loading + prompt-encoder mode) that are **not** in public
[`allenai/MolmoBot`](https://github.com/allenai/MolmoBot). This directory carries
those modifications as a patch against a pinned MolmoBot commit, so the path is
reproducible from this release.

## Contents

- `molmobot_traj_d0d71e28.patch` — diff of 10 MolmoBot files (+~1560 lines):
  `launch_scripts/train_molmobot.py`, `olmo/data/synthmanip_{config,dataset}.py`,
  `olmo/models/molmobot/{molmobot,inference_wrapper}.py`,
  `olmo/models/video_olmo/video_preprocessor.py`, `olmo/nn/llm.py`,
  `olmo/train/trainer.py`, `olmo/train_init_utils.py`,
  `olmo/eval/configure_molmo_spaces.py`. Adds the `--load_3d_tracks`,
  `--prompt_style`, `--prompt_encoder_mode` flags + the synthmanip 3D-track
  dataset path + `can_predict_extra_tokens` schema-compat for MolmoMotion ckpts.
- `new_files/` — 3 new MolmoBot files not in upstream:
  `olmo/data/{droid_dataset,synthmanip_tracks_dataset}.py`,
  `olmo/eval/mesh_surface_sampler.py`.

## Pinned upstream commits (the tested base)

| Repo | Commit |
|---|---|
| `github.com/allenai/MolmoBot` | `d0d71e28` |
| `github.com/allenai/molmospaces` (sim, eval only) | `a3cd202e` |

The patch is generated against `d0d71e28`; it will **not** apply cleanly to
MolmoBot HEAD (which dropped these areas). Use `d0d71e28`.

## Apply

```bash
git clone https://github.com/allenai/MolmoBot
cd MolmoBot && git checkout d0d71e28
git apply /path/to/robotics/molmobot_traj/molmobot_traj_d0d71e28.patch
cp -r /path/to/robotics/molmobot_traj/new_files/olmo/* MolmoBot/olmo/   # 3 new files
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
    --release_root /path/to/molmomotion-1m/molmospaces --dst_root <train_view>

# 3. finetune
bash launch_scripts/train.sh configs/molmobot_pickplace.yaml
```

## Validation (2026-06-14)

Smoke-verified end-to-end on 1×L40S: patch applies cleanly to `d0d71e28`;
`prepare_training_data.py --release_root` builds the view; `train_molmobot.py`
builds the 598M-param policy (578M VLM + 19.3M ActionExpert), loads the
MolmoMotion init (reinitializing the action expert), and runs training steps with
real flow-matching loss (`train/flow_loss_*`).
