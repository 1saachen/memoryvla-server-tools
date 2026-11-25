#!/usr/bin/env bash
set -euo pipefail

if (( $# < 4 || $# > 5 )); then
    echo "Usage: bash run_eval.sh WORK_DIR CHECKPOINT MODEL_GPU RENDER_EGL [PER_ORDER]" >&2
    exit 2
fi

work="$(realpath -- "$1")"
checkpoint="$(realpath -- "$2")"
model_gpu="$3"
render_egl="$4"
per_order="${5:-20}"
chunk_steps="${BUTTONS_CHUNK_STEPS:-4}"
if [[ ! "$chunk_steps" =~ ^[1-4]$ ]]; then
    echo 'BUTTONS_CHUNK_STEPS must be between 1 and 4.' >&2
    exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
model_repo="${MEMVLA_REPO:-$(dirname "$work")/MemoryVLA}"
model_python="${MEMVLA_PYTHON:-$(dirname "$work")/envs/memvla/bin/python}"
sim_python="${SIM_PYTHON:-$(command -v python || true)}"

if [[ ! -x "$sim_python" ]] || ! "$sim_python" -c 'import importlib.util, sys; sys.exit(not all(importlib.util.find_spec(m) for m in ("pyrep", "rlbench")))'; then
    echo 'Activate buttons-sim first or set SIM_PYTHON to its Python executable.' >&2
    exit 2
fi
if [[ ! -f "$checkpoint" || ! -f "$model_repo/train.py" || ! -x "$model_python" ]]; then
    echo 'Checkpoint, MemoryVLA repository or inference Python is missing.' >&2
    exit 2
fi

model_uuid="$(nvidia-smi -i "$model_gpu" --query-gpu=uuid --format=csv,noheader)"
driver="$(nvidia-smi -i "$model_gpu" --query-gpu=driver_version --format=csv,noheader)"
graphics="${NVIDIA_GRAPHICS_ROOT:-$(dirname "$work")/nvidia_vulkan/$driver/extracted/usr}"
export COPPELIASIM_ROOT="${COPPELIASIM_ROOT:-$work/coppeliasim}"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
export LD_LIBRARY_PATH="$graphics/lib/x86_64-linux-gnu:$COPPELIASIM_ROOT${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export __EGL_VENDOR_LIBRARY_FILENAMES="$graphics/share/glvnd/egl_vendor.d/10_nvidia.json"
unset LIBGL_ALWAYS_SOFTWARE LIBGL_ALWAYS_INDIRECT
if [[ ! -r "$__EGL_VENDOR_LIBRARY_FILENAMES" ]]; then
    printf 'NVIDIA EGL manifest is missing: %s\n' "$__EGL_VENDOR_LIBRARY_FILENAMES" >&2
    exit 2
fi

vgl=/opt/VirtualGL/bin/vglrun
renderer="$(xvfb-run -a "$vgl" -d "$render_egl" glxinfo -B)"
if ! grep -i '^OpenGL renderer string:.*NVIDIA' <<< "$renderer"; then
    echo 'VirtualGL did not report an NVIDIA renderer.' >&2
    exit 2
fi

mkdir -p "$work/evals"
output="$(mktemp -d "$work/evals/buttons-$(date +%Y%m%d-%H%M%S).XXXXXX")"
nohup xvfb-run -a -s '-screen 0 1024x768x24' "$vgl" -d "$render_egl" \
    "$sim_python" -u "$script_dir/eval_buttons.py" --data "$work/data/full" \
    --checkpoint "$checkpoint" --output "$output" --model-repo "$model_repo" \
    --model-python "$model_python" --model-gpu "$model_uuid" --per-order "$per_order" \
    --chunk-steps "$chunk_steps" \
    > "$output/eval.log" 2>&1 < /dev/null &
eval_pid=$!
printf '%s\n' "$eval_pid" > "$output/eval.pid"
printf '%s\n' "$output" > "$work/evals/latest-buttons-eval.txt"
printf 'PID=%s\nLog=%s/eval.log\nResults=%s\n' "$eval_pid" "$output" "$output"
tail --pid="$eval_pid" -n 20 -f "$output/eval.log"
