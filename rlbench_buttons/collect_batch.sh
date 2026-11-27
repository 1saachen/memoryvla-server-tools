#!/usr/bin/env bash
set -euo pipefail

usage="Usage: bash collect_batch.sh WORK_DIR [TRAIN_PER_ORDER [VAL_PER_ORDER [TEST_PER_ORDER]]]"
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  printf '%s\n' "$usage" \
    "Default per-order counts: 200 train, 20 validation, 20 test." \
    "Set BUTTONS_OUTPUT_DIR to change the dataset directory." \
    "Set BUTTONS_LAYOUT_SCALE to sample from the centered fraction of the original width and depth." \
    "Set BUTTONS_ORDERS to a space-separated subset of order IDs (0 through 5)." \
    "Set BUTTONS_SEED to change the base layout seed." \
    "Set SIM_PYTHON to the simulation Python and BUTTONS_RENDER_EGL for NVIDIA rendering." \
    "Set BUTTONS_ATTEMPTS to change the maximum attempts per episode (default: 12)."
  exit 0
fi
if (( $# < 1 || $# > 4 )); then
  printf '%s\n' "$usage" >&2
  exit 2
fi

splits=(train val test)
per_order=("${2:-200}" "${3:-20}" "${4:-20}")
attempts="${BUTTONS_ATTEMPTS:-12}"
layout_scale="${BUTTONS_LAYOUT_SCALE:-1.0}"
layout_seed="${BUTTONS_SEED:-20261120}"
sim_python="${SIM_PYTHON:-python}"
read -r -a orders <<< "${BUTTONS_ORDERS:-0 1 2 3 4 5}"
if (( ${#orders[@]} == 0 )); then
  printf 'BUTTONS_ORDERS must contain at least one order.\n' >&2
  exit 2
fi
seen_orders=()
for order in "${orders[@]}"; do
  if [[ ! "$order" =~ ^[0-5]$ || -n "${seen_orders[order]:-}" ]]; then
    printf 'BUTTONS_ORDERS must contain distinct order IDs from 0 to 5.\n' >&2
    exit 2
  fi
  seen_orders[order]=1
done
for count in "${per_order[@]}"; do
  if [[ ! "$count" =~ ^[1-9][0-9]*$ ]]; then
    printf 'Episode counts per order must be positive integers.\n' >&2
    exit 2
  fi
done
if [[ ! "$attempts" =~ ^[1-9][0-9]*$ ]]; then
  printf 'BUTTONS_ATTEMPTS must be a positive integer.\n' >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
work_dir="$(realpath -m -- "$1")"
output_dir="$(realpath -m -- "${BUTTONS_OUTPUT_DIR:-$work_dir/data/full}")"
log_file="$work_dir/logs/collect-$(basename -- "$output_dir").log"
if [[ -n "${BUTTONS_ORDERS:-}" ]]; then
  order_label="$(IFS=-; printf '%s' "${orders[*]}")"
  log_file="${log_file%.log}-orders-$order_label.log"
fi

export COPPELIASIM_ROOT="${COPPELIASIM_ROOT:-$work_dir/coppeliasim}"
if [[ ! -d "$COPPELIASIM_ROOT" ]]; then
  printf 'CoppeliaSim directory not found: %s\n' "$COPPELIASIM_ROOT" >&2
  exit 2
fi
export LD_LIBRARY_PATH="$COPPELIASIM_ROOT${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
sim_command=(xvfb-run -a -s '-screen 0 1024x768x24')
if [[ -n "${BUTTONS_RENDER_EGL:-}" ]]; then
  driver="$(nvidia-smi -i 0 --query-gpu=driver_version --format=csv,noheader)"
  graphics="${NVIDIA_GRAPHICS_ROOT:-$(dirname "$work_dir")/nvidia_vulkan/$driver/extracted/usr}"
  export LD_LIBRARY_PATH="$graphics/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
  export __EGL_VENDOR_LIBRARY_FILENAMES="$graphics/share/glvnd/egl_vendor.d/10_nvidia.json"
  unset LIBGL_ALWAYS_SOFTWARE LIBGL_ALWAYS_INDIRECT
  if [[ ! -r "$__EGL_VENDOR_LIBRARY_FILENAMES" ]]; then
    printf 'NVIDIA EGL manifest is missing: %s\n' "$__EGL_VENDOR_LIBRARY_FILENAMES" >&2
    exit 2
  fi
  sim_command+=(/opt/VirtualGL/bin/vglrun -d "$BUTTONS_RENDER_EGL")
  renderer="$("${sim_command[@]}" glxinfo -B)"
  if ! grep -i '^OpenGL renderer string:.*NVIDIA' <<< "$renderer"; then
    printf 'VirtualGL did not report an NVIDIA renderer.\n' >&2
    exit 2
  fi
else
  export LIBGL_ALWAYS_SOFTWARE=1
fi

mkdir -p -- "$(dirname -- "$log_file")"
shopt -s nullglob
for i in "${!splits[@]}"; do
  split="${splits[i]}"
  count="${per_order[i]}"
  expected=$((${#orders[@]} * count))
  printf '\nCollecting %s: %d episodes (%d per order) into %s/%s\n' \
    "$split" "$expected" "$count" "$output_dir" "$split" | tee -a "$log_file"
  "${sim_command[@]}" "$sim_python" -u "$script_dir/collect.py" --output "$output_dir" \
    --split "$split" --per-order "$count" --attempts "$attempts" --resume \
    --layout-scale "$layout_scale" --seed "$layout_seed" --orders "${orders[@]}" 2>&1 | tee -a "$log_file"

  for order in "${orders[@]}"; do
    episodes=("$output_dir/$split"/order"${order}"_episode*.npz)
    if (( ${#episodes[@]} != count )); then
      printf '%s order %s has %d episode files, expected %d. Check the output directory.\n' \
        "$split" "$order" "${#episodes[@]}" "$count" >&2
      exit 1
    fi
  done
  printf '%s complete: %d episodes\n' "$split" "$expected" | tee -a "$log_file"
done
