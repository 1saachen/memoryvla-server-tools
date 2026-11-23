#!/usr/bin/env bash
set -euo pipefail

usage="Usage: bash collect_batch.sh WORK_DIR [TRAIN_PER_ORDER [VAL_PER_ORDER [TEST_PER_ORDER]]]"
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  printf '%s\n' "$usage" \
    "Default per-order counts: 200 train, 20 validation, 20 test." \
    "Set BUTTONS_OUTPUT_DIR to change the dataset directory." \
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
log_file="$work_dir/logs/collect-full.log"

export COPPELIASIM_ROOT="${COPPELIASIM_ROOT:-$work_dir/coppeliasim}"
if [[ ! -d "$COPPELIASIM_ROOT" ]]; then
  printf 'CoppeliaSim directory not found: %s\n' "$COPPELIASIM_ROOT" >&2
  exit 2
fi
export LD_LIBRARY_PATH="$COPPELIASIM_ROOT${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
export LIBGL_ALWAYS_SOFTWARE=1

mkdir -p -- "$(dirname -- "$log_file")"
shopt -s nullglob
for i in "${!splits[@]}"; do
  split="${splits[i]}"
  count="${per_order[i]}"
  expected=$((6 * count))
  printf '\nCollecting %s: %d episodes (%d per order) into %s/%s\n' \
    "$split" "$expected" "$count" "$output_dir" "$split" | tee -a "$log_file"
  xvfb-run -a -s '-screen 0 1024x768x24' \
    python "$script_dir/collect.py" --output "$output_dir" \
    --split "$split" --per-order "$count" --attempts "$attempts" --resume 2>&1 | tee -a "$log_file"

  episodes=("$output_dir/$split"/order*_episode*.npz)
  if (( ${#episodes[@]} != expected )); then
    printf '%s has %d episode files, expected %d. Check the output directory.\n' \
      "$split" "${#episodes[@]}" "$expected" >&2
    exit 1
  fi
  printf '%s complete: %d episodes\n' "$split" "$expected" | tee -a "$log_file"
done
