#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "Usage: bash train.sh MEMVLA_REPO DATA_ROOT INITIAL_PT RUN_ROOT RUN_ID MAX_STEPS" >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo="$(realpath -- "$1")"
data="$(realpath -- "$2")"
checkpoint="$(realpath -- "$3")"
run_root="$(realpath -m -- "$4")"
run_id="$5"
max_steps="$6"
gpus="${GPUS:-4}"
frames="${GROUP_SIZE:-16}"
memory="${MEM_LENGTH:-16}"
save_interval="${SAVE_INTERVAL:-1000}"
dataloader="${DATALOADER_TYPE:-group}"

if [[ ! "$gpus" =~ ^[1-9][0-9]*$ || ! "$frames" =~ ^[1-9][0-9]*$ ]]; then
  echo "GPUS and GROUP_SIZE must be positive integers." >&2
  exit 2
fi
if [[ "$dataloader" != group && "$dataloader" != stream ]]; then
  echo "DATALOADER_TYPE must be group or stream." >&2
  exit 2
fi

if [[ ! -f "$data/custom_finetuning/1.0.0/dataset_info.json" ||
      ! -f "$checkpoint" ||
      ! -f "$(dirname "$(dirname "$checkpoint")")/config.json" ||
      ! -f "$(dirname "$(dirname "$checkpoint")")/dataset_statistics.json" ]]; then
  echo "Expected a built RLDS dataset and a complete MemoryVLA checkpoint directory." >&2
  exit 2
fi
if [[ -e "$run_root/$run_id" ]]; then
  echo "Run already exists: $run_root/$run_id. Choose a new run ID." >&2
  exit 2
fi

cd -- "$repo"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
# Group mode gives each rank one group per batch. Stream mode keeps the
# same per-rank batch size and reads successive frames from each episode.
torchrun --standalone --nproc_per_node="$gpus" "$script_dir/train_entrypoint.py" \
  --pretrained_checkpoint "$checkpoint" \
  --is_resume False --resume_step 0 --resume_epoch 0 \
  --vla.type prism-dinosiglip-224px+oxe+diffusion \
  --vla.data_mix custom_finetuning --data_root_dir "$data" \
  --vla.expected_world_size "$gpus" \
  --vla.per_device_batch_size "$frames" --vla.global_batch_size "$((gpus * frames))" \
  --vla.freeze_vision_backbone True --vla.freeze_llm_backbone True \
  --vla.unfreeze_last_llm_layer False \
  --vla.learning_rate 2e-5 --vla.max_steps "$max_steps" \
  --vla.shuffle_buffer_size 128 --dataloader_type "$dataloader" --group_size "$frames" \
  --mem_length "$memory" --future_action_window_size 15 \
  --action_model_type DiT-L --repeated_diffusion_steps 4 --use_ema False \
  --image_aug False --trackers '["jsonl"]' \
  --run_root_dir "$run_root" --run_id "$run_id" --save_interval "$save_interval"
