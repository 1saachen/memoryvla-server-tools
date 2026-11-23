#!/usr/bin/env bash
set -euo pipefail

usage='Usage: bash run_sgt.sh MODEL_GPU [RENDER_GPU] [EPISODES] [--paired-memory-ablation]'
model_gpu=${1:?$usage}
render_gpu=${2:-5}
episodes=${3:-100}
eval_flags=()
if [[ ${4:-} == --paired-memory-ablation && $# == 4 ]]; then
    eval_flags+=(--paired-memory-ablation)
elif (( $# > 3 )); then
    echo "$usage" >&2
    exit 2
fi
base=/data/embodied-ai/yaolin
root="$base/sgt-repro"
repo="$base/MemoryVLA"
model_python="$base/envs/memvla/bin/python"
eval_python="$root/venv-sgt/bin/python"
checkpoint="$root/models/memvla-mikasa/checkpoints/memvla-mikasa.pt"
tools_dir="$(dirname "$(realpath "$0")")"

[[ -s "$checkpoint" && -r "$repo/deploy.py" && -x "$model_python" && -x "$eval_python" ]]
[[ -s "$root/maniskill-data/assets/mani_skill2_ycb/info_pick_v0.json" ]]
if curl -sS --max-time 2 -o /dev/null http://127.0.0.1:8052/ 2>/dev/null; then
    echo "Port 8052 is already serving HTTP." >&2
    exit 1
fi
model_uuid=$(nvidia-smi -i "$model_gpu" --query-gpu=uuid --format=csv,noheader)
render_uuid=$(nvidia-smi -i "$render_gpu" --query-gpu=uuid --format=csv,noheader)
[[ "$model_uuid" == GPU-* && "$model_uuid" != *$'\n'* ]]
[[ "$render_uuid" == GPU-* && "$render_uuid" != *$'\n'* ]]
run=$(mktemp -d "$root/sgt_gpu_XXXXXX")
cd "$run"

CUDA_VISIBLE_DEVICES="$model_uuid" TMPDIR="$run" "$model_python" -u "$tools_dir/sgt_server.py" \
    "$repo/deploy.py" --saved_model_path "$checkpoint" --unnorm_key mikasa_dataset \
    --action_chunking --action_chunking_window 4 --cfg_scale 1.5 --seed 7 --port 8052 \
    > "$run/server.log" 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true' EXIT
trap 'exit 143' TERM
printf 'RUN_DIR=%s\nSERVER_PID=%s\n' "$run" "$server_pid"

for ((attempt=0; attempt<300; attempt++)); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
        tail -n 60 "$run/server.log" >&2
        exit 1
    fi
    if [[ $(curl -fsS --max-time 2 http://127.0.0.1:8052/sgt-ready 2>/dev/null || true) == SGT_READY ]]; then
        break
    fi
    sleep 2
done
if (( attempt == 300 )); then
    echo "Model readiness timed out. See $run/server.log" >&2
    exit 1
fi

source "$base/nvidia_vulkan/570.148.08/env.sh"
export MS_ASSET_DIR="$root/maniskill-data"
export CUDA_VISIBLE_DEVICES="$render_uuid"
printf 'SERVER_READY\nEVAL_LOG=%s/eval.log\n' "$run"
if ! "$eval_python" -u "$tools_dir/eval_sgt.py" --render-backend cuda:0 --episodes "$episodes" \
    --seed 7 --out "$run" "${eval_flags[@]}" > "$run/eval.log" 2>&1; then
    tail -n 30 "$run/eval.log" >&2
    tail -n 60 "$run/server.log" >&2
    exit 1
fi
cat "$run/summary.json"
