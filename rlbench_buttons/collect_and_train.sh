#!/usr/bin/env bash
set -euo pipefail

if (( $# < 3 || $# > 4 )); then
    printf 'Usage: bash collect_and_train.sh WORK_DIR INITIAL_PT LOG_DIR [TRAIN_STEPS]\n' >&2
    exit 2
fi

stage=preflight
trap 'status=$?; if (( status != 0 )); then printf "Failed during %s (exit %d).\n" "$stage" "$status" >&2; fi' EXIT
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
work="$(realpath -- "$1")"
initial_pt="$(realpath -- "$2")"
log_dir="$(realpath -- "$3")"
steps="${4:-5000}"
base="$(dirname -- "$work")"
sim_python="${SIM_PYTHON:-python}"
model_python="${MEMVLA_PYTHON:-$base/envs/memvla/bin/python}"
model_repo="${MEMVLA_REPO:-$base/MemoryVLA}"
train_gpus="${BUTTONS_TRAIN_GPUS:-0,1,2}"
data="$work/data/center50"
run_id="$(basename -- "$log_dir")"
rlds="$work/rlds/$run_id"

if [[ ! "$steps" =~ ^[1-9][0-9]*$ || ! "$train_gpus" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    printf 'Expected a positive training step count and comma-separated GPU indices.\n' >&2
    exit 2
fi
read -r -a gpu_ids <<< "${train_gpus//,/ }"
for required in "$initial_pt" "$(dirname -- "$(dirname -- "$initial_pt")")/config.json" \
    "$(dirname -- "$(dirname -- "$initial_pt")")/dataset_statistics.json" "$model_repo/train.py"; do
    if [[ ! -r "$required" ]]; then
        printf 'Missing or unreadable file: %s\n' "$required" >&2
        exit 2
    fi
done
if [[ ! -d "$log_dir" || ! -x "$model_python" || ! -x "$(dirname -- "$model_python")/torchrun" ]]; then
    printf 'Expected a log directory and an existing MemoryVLA Python environment with torchrun.\n' >&2
    exit 2
fi
if [[ -e "$rlds" || -e "$work/runs/$run_id" ]]; then
    printf 'Run directory already in use. Start with a fresh log directory.\n' >&2
    exit 2
fi
"$sim_python" -c 'import importlib.util as u, sys; sys.exit(not all(u.find_spec(m) for m in ("pyrep", "rlbench")))'
CUDA_VISIBLE_DEVICES="" "$model_python" -c 'import tensorflow_datasets'

mkdir -p -- "$data"
exec 9>>"$data/.collect-and-train.lock"
if ! flock -n 9; then
    printf 'Another collect-and-train job is using %s\n' "$data" >&2
    exit 2
fi

shopt -s nullglob
if [[ -n "${BUTTONS_WAIT_LOG_DIR:-}" ]]; then
    stage='waiting for existing collection'
    previous_pids=("$BUTTONS_WAIT_LOG_DIR"/egl[456].pid)
    if (( ${#previous_pids[@]} == 0 )); then
        printf 'No collection PID files found in %s\n' "$BUTTONS_WAIT_LOG_DIR" >&2
        exit 2
    fi
    for pid_file in "${previous_pids[@]}"; do
        read -r pid < "$pid_file"
        if [[ ! "$pid" =~ ^[1-9][0-9]*$ ]]; then
            printf 'Invalid collection PID in %s\n' "$pid_file" >&2
            exit 2
        fi
        printf 'Waiting for prior collection PID %s\n' "$pid"
        tail --pid="$pid" -f /dev/null
    done
fi

stage=collection
printf 'Collecting or resuming center50: %s\n' "$data"
devices=(egl4 egl5 egl6)
orders=("0 1" "2 3" "4 5")
pids=()
for index in "${!devices[@]}"; do
    device="${devices[index]}"
    SIM_PYTHON="$sim_python" BUTTONS_OUTPUT_DIR="$data" BUTTONS_LAYOUT_SCALE=0.5 BUTTONS_SEED=20261127 \
        BUTTONS_RENDER_EGL="$device" BUTTONS_ORDERS="${orders[index]}" \
        bash "$script_dir/collect_batch.sh" "$work" 300 20 20 > "$log_dir/$device.log" 2>&1 &
    pids+=("$!")
    printf '%s\n' "$!" > "$log_dir/$device.pid"
done

collection_status=0
for index in "${!pids[@]}"; do
    device="${devices[index]}"
    if wait "${pids[index]}"; then
        printf 'Collection complete on %s\n' "$device"
    else
        collection_status=$?
        printf 'Collection failed on %s (exit %d). See %s/%s.log\n' \
            "$device" "$collection_status" "$log_dir" "$device" >&2
    fi
done
if (( collection_status != 0 )); then
    exit "$collection_status"
fi

stage='RLDS export'
printf 'Exporting training episodes to %s\n' "$rlds"
CUDA_VISIBLE_DEVICES="" "$model_python" -u "$script_dir/export_rlds.py" \
    --source "$data" --data-root "$rlds" > "$log_dir/export.log" 2>&1

stage=training
printf 'Training %s for %s steps on GPUs %s\n' "$run_id" "$steps" "$train_gpus"
PATH="$(dirname -- "$model_python"):$PATH" PYTHONUNBUFFERED=1 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$train_gpus" \
    GPUS="${#gpu_ids[@]}" DATALOADER_TYPE=stream GROUP_SIZE=16 MEM_LENGTH=256 SAVE_INTERVAL=1000 \
    bash "$script_dir/train.sh" "$model_repo" "$rlds" "$initial_pt" "$work/runs" "$run_id" "$steps" \
    > "$log_dir/train.log" 2>&1

printf -v final_step '%06d' "$steps"
for checkpoint in "$work/runs/$run_id/checkpoints"/step-"$final_step"-*.pt; do
    if [[ -s "$checkpoint" ]]; then
        printf 'Training complete. Checkpoint=%s\n' "$checkpoint"
        exit 0
    fi
done
printf 'No final checkpoint for step %s in %s/runs/%s/checkpoints\n' "$final_step" "$work" "$run_id" >&2
exit 1
