# Robotics: MolmoBot finetuning from MolmoMotion

Finetune a [MolmoBot](https://github.com/allenai/molmobot) policy from a
MolmoMotion pretrained checkpoint and evaluate it on the
[MolmoSpaces](https://github.com/allenai/molmospaces) Franka pick-and-
place benchmark. This subdirectory provides the recipe (config, data
prep, checkpoint conversion, evaluation driver); the model and trainer
code live upstream.

## Installation

MolmoBot finetuning here needs trajectory-conditioning extensions
(`--load_3d_tracks` / `--prompt_encoder_mode` / `--prompt_style`) that are not in
public MolmoBot. Clone current `main` and apply the patch + overlay shipped in
[`molmobot_traj/`](molmobot_traj/) — see [molmobot_traj/README.md](molmobot_traj/README.md)
for the full recipe (env + the torchcodec/FFmpeg note):

```bash
git clone https://github.com/allenai/MolmoBot
cd MolmoBot
git apply /path/to/robotics/molmobot_traj/trajectory_conditioning.patch
cp -r /path/to/robotics/molmobot_traj/overlay/olmo/* MolmoBot/olmo/
export MOLMOBOT_REPO="$PWD/MolmoBot"
uv sync --extra train && uv pip install "torchcodec==0.4.*"
```

## Data

We use the MolmoSpaces pick-and-place 2-camera dataset for training and
the upstream Franka pick-and-place benchmark for evaluation. See the
[MolmoSpaces release](https://github.com/allenai/molmospaces) for
download instructions and benchmark paths.

Convert the downloaded training data into the layout the MolmoBot loader
expects:

```bash
python scripts/prepare_training_data.py \
    --src_root /path/to/molmospaces_data \
    --dst_root /path/to/train_view
```

The robot trajectories also ship in the **molmo-motion-1m** dataset under
`molmospaces/robot_trajectories/` (per-house h5; the episode mp4s are the same
`molmospaces/videos/` shards). To build the training view straight from an
unpacked release, use `--release_root` instead of `--src_root`:

```bash
python scripts/prepare_training_data.py \
    --release_root /path/to/molmo-motion-1m/molmospaces \
    --dst_root /path/to/train_view
```

This walks each task directory under `--src_root`, splits houses 95/5
train/val, symlinks the videos, and writes a `valid_trajectory_index.json`
+ a copy of each `trajectories_batch_*.h5` with mp4 filenames injected
under `obs/sensor_data/<camera>`.

## Initialization

Download a MolmoMotion checkpoint (see the
[released models table](../README.md#downloading-released-models) in the
top-level README):

| Name | History H |
|---|---:|
| [MolmoMotion-4B-H3-Pretrain](https://huggingface.co/allenai/MolmoMotion-4B-H3-Pretrain) | 3 |
| [MolmoMotion-4B-H1-F32](https://huggingface.co/allenai/MolmoMotion-4B-H1-F32) | 1 |

The HF releases ship unsharded. If you are starting from your own
MolmoMotion training run (e.g. a custom H=1 pretrain), convert the FSDP-2
sharded checkpoint to a single `model.pt + config.yaml` first:

```bash
python scripts/unshard_pretrained.py \
    /path/to/molmomotion_run/step100000 \
    /path/to/molmo-motion-unsharded
```

## Training

Edit [`configs/molmobot_pickplace.yaml`](configs/molmobot_pickplace.yaml)
to point at your checkpoint and data, then:

```bash
bash launch_scripts/train.sh configs/molmobot_pickplace.yaml
```

The script reads every hyperparameter from the YAML and runs
`torchrun --nproc-per-node=$NPROC_PER_NODE` (default 8) against
`$MOLMOBOT_REPO/launch_scripts/train_molmobot.py`.

## Evaluation

The evaluator polls a training run's checkpoint folder, runs each new
`step<N>/` against a MolmoSpaces benchmark, and appends per-step
success rates to `summary.json`:

```bash
bash launch_scripts/eval.sh \
    --save_folder      /path/to/train_run \
    --eval_out         /path/to/eval_output \
    --benchmark_dir    /path/to/benchmark \
    --eval_every_n_steps 10000 \
    --stop_after_step 100000 \
    --eval_mode hybrid \
    --pretrained_ckpt_path /path/to/MolmoBot-DROID
```

`--eval_mode hybrid` uses the released
[MolmoBot-DROID](https://huggingface.co/allenai/MolmoBot-DROID) policy
to drive the simulator until the gripper closes on the pickup object,
then hands control to the finetuned policy. Set `--eval_mode standalone`
to run the finetuned policy from `t=0` instead.

## Advanced: bucket-balanced eval subset

For fine-grained analysis on `(seen-house × seen-object)` slices of a
benchmark, build a smaller bucket-balanced subset:

```bash
# 1. Index the seen entities in the training view
python scripts/scan_train_signatures.py \
    --src_root /path/to/train_view \
    --output   train_signatures.json

# 2. Select 50 ss + 25 su + 25 us episodes deterministically
python scripts/build_eval_subset.py \
    --benchmark_dir /path/to/upstream_benchmark \
    --train_sig     train_signatures.json \
    --dst_root      /path/to/eval_subset \
    --manifest      /path/to/eval_subset/manifest.json
```

Point `--benchmark_dir` of `launch_scripts/eval.sh` at the resulting
`<dst_root>` to use this subset for monitored evaluation.

## File layout

```
robotics/
├── configs/molmobot_pickplace.yaml    canonical training hyperparameters
├── launch_scripts/
│   ├── train.sh                       torchrun wrapper around MolmoBot's trainer
│   └── eval.sh                        per-checkpoint evaluation driver
└── scripts/
    ├── prepare_training_data.py       MolmoSpaces data → MolmoBot loader format
    ├── unshard_pretrained.py          FSDP-2 distcp → model.pt
    ├── eval_sidecar.py                eval engine called by eval.sh
    ├── scan_train_signatures.py       (advanced) train signature for bucketing
    └── build_eval_subset.py           (advanced) bucket-balanced eval subset
```
