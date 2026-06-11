#!/usr/bin/env bash
# Launch a MolmoBot pick-and-place training run from a MolmoMotion init.
#
# Reads canonical hyperparameters from configs/molmobot_pickplace.yaml.
# This script is intentionally a thin torchrun wrapper — no scheduler, no
# environment management; you bring your own conda env + your own
# multi-GPU launcher.
#
# Prerequisites (one-time):
#   1. Clone MolmoBot to ${MOLMOBOT_REPO} and run `uv sync --extra train`.
#   2. Run scripts/prepare_training_data.py to produce the training view.
#   3. Run scripts/unshard_pretrained.py to produce an unsharded MolmoMotion
#      checkpoint to initialize from.
#
# Usage:
#   bash launch_scripts/train.sh <config.yaml>
#
# Default config: configs/molmobot_pickplace.yaml relative to this repo.
set -euo pipefail

CONFIG="${1:-$(dirname "$0")/../configs/molmobot_pickplace.yaml}"
if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: config not found: $CONFIG" >&2
    exit 2
fi
: "${MOLMOBOT_REPO:?set MOLMOBOT_REPO to the MolmoBot clone root}"
: "${NPROC_PER_NODE:=8}"

# Parse the YAML with a tiny python helper so we don't depend on `yq`.
read_yaml() {
    python3 -c "
import yaml, sys
cfg = yaml.safe_load(open('$CONFIG'))
def get(path):
    cur = cfg
    for k in path.split('.'):
        if isinstance(cur, list) or cur is None: return ''
        cur = cur.get(k)
        if cur is None: return ''
    return cur
def fmt(v):
    if isinstance(v, list): return ' '.join(str(x) for x in v)
    if isinstance(v, bool): return str(v)
    return str(v)
print(fmt(get(sys.argv[1])))
" "$1"
}

INIT_CKPT=$(read_yaml init_ckpt)
SAVE_DIR=$(read_yaml save_folder)
STATS_PATH=$(read_yaml stats_path)
DATA_PATHS=$(read_yaml data_paths)
SAMPLE_RATES=$(read_yaml dataset_sample_rates)
NUM_POINTS=$(read_yaml num_points)
HISTORY_SIZE=$(read_yaml history_size)
NUM_FUTURE=$(read_yaml num_future_frames)
HISTORY_STRIDE=$(read_yaml history_stride)
FUTURE_STRIDE=$(read_yaml future_stride)
SAMPLE_PHASES=$(read_yaml sample_phases | tr ' ' ',')
N_OBS_STEPS=$(read_yaml n_obs_steps)
OBS_STEP_DELTA=$(read_yaml obs_step_delta)
SEQ_LEN=$(read_yaml seq_len)
MAX_DUR=$(read_yaml max_duration)
DEV_BS=$(read_yaml device_batch_size)
GLOB_BS=$(read_yaml global_batch_size)
LOG_INT=$(read_yaml log_interval)
VAL_INT=$(read_yaml val_interval)
VAL_MAX=$(read_yaml val_max_examples)
SAVE_INT=$(read_yaml save_interval)
KEEP=$(read_yaml save_num_checkpoints_to_keep)
ACTION_PRESET=$(read_yaml action_preset)
CAMERA_PRESET=$(read_yaml camera_preset)
NFT=$(read_yaml num_flow_timestamps)
PROMPT_STYLE=$(read_yaml prompt_style)
PROMPT_ENC=$(read_yaml prompt_encoder_mode)
FT_LLM=$(read_yaml ft_llm)
FT_EMB=$(read_yaml ft_embedding)
IMG_AUG=$(read_yaml img_aug)
FURTHEST=$(read_yaml furthest_camera_prob)
CROP_MODE=$(read_yaml model_overrides.mm_preprocessor.image.crop_mode)
MAX_IMAGES=$(read_yaml model_overrides.mm_preprocessor.image.max_images)
COMPILE_LOSS=$(read_yaml compile_loss)
WEIGHTED_SAMP=$(read_yaml weighted_sampling)
RUN_NAME=$(basename "$SAVE_DIR")

# Sanity
if [[ ! -f "${INIT_CKPT}/model.pt" ]]; then
    echo "ERROR: ${INIT_CKPT}/model.pt missing. Run scripts/unshard_pretrained.py first." >&2
    exit 2
fi
mkdir -p "$SAVE_DIR" "$(dirname "$STATS_PATH")"

echo "[train] init_ckpt = $INIT_CKPT"
echo "[train] save_dir  = $SAVE_DIR"
echo "[train] data_paths = $DATA_PATHS"
echo "[train] sample_rates = $SAMPLE_RATES"
echo "[train] N=$NUM_POINTS H=$HISTORY_SIZE F=$NUM_FUTURE  prompt_encoder=$PROMPT_ENC"

CMD=(torchrun
    --nnodes=1 --nproc-per-node="$NPROC_PER_NODE"
    "$MOLMOBOT_REPO/launch_scripts/train_molmobot.py" "$INIT_CKPT"
    --load_3d_tracks
    --prompt_style "$PROMPT_STYLE"
    --num_points="$NUM_POINTS"
    --history_size="$HISTORY_SIZE"
    --num_future_frames="$NUM_FUTURE"
    --history_stride="$HISTORY_STRIDE" --future_stride="$FUTURE_STRIDE"
    --sample_phases "$SAMPLE_PHASES"
    --n_obs_steps="$N_OBS_STEPS" --obs_step_delta="$OBS_STEP_DELTA"
    --seq_len="$SEQ_LEN"
    --max_duration="$MAX_DUR"
    --save_interval="$SAVE_INT"
    --save_num_checkpoints_to_keep="$KEEP"
    --device_batch_size="$DEV_BS" --global_batch_size="$GLOB_BS"
    --log_interval="$LOG_INT"
    --val_interval="$VAL_INT" --val_max_examples="$VAL_MAX"
    --action_preset "$ACTION_PRESET"
    --camera_preset "$CAMERA_PRESET"
    --ft_embedding="$FT_EMB" --ft_llm="$FT_LLM"
    --furthest_camera_prob="$FURTHEST"
    --model.mm_preprocessor.image.crop_mode="$CROP_MODE"
    --model.mm_preprocessor.image.max_images="$MAX_IMAGES"
    --model.num_flow_timestamps="$NFT"
    --compile_loss="$COMPILE_LOSS"
    --stats_path="$STATS_PATH"
    --save_folder="$SAVE_DIR"
    --wandb.name="$RUN_NAME"
)
[[ "$PROMPT_ENC" == "True" || "$PROMPT_ENC" == "true" ]] && CMD+=(--prompt_encoder_mode)
[[ "$IMG_AUG" == "True" || "$IMG_AUG" == "true" ]] && CMD+=(--img_aug)
[[ "$WEIGHTED_SAMP" == "True" || "$WEIGHTED_SAMP" == "true" ]] && CMD+=(--weighted_sampling)

CMD+=(--data_paths)
for p in $DATA_PATHS; do CMD+=("$p"); done
CMD+=(--dataset_sample_rates)
for r in $SAMPLE_RATES; do CMD+=("$r"); done

echo "[train] launching:"
echo "${CMD[@]}"
exec "${CMD[@]}"
