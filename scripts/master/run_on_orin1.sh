#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'MSG'
Usage:
  ./scripts/master/run_on_orin1.sh <target>

Targets:
  check-env          Run Orin1 environment checks.
  ch3-isolated      Chapter 3 isolated baseline, 100 images.
  ch3-concurrent    Chapter 3 TS/MPS motivation, 100 images.
  ch3-sweep         Chapter 3 isolated memory sweep, 100 images.
  ch5-concurrent    Chapter 5 TS/MPS/PRIMA-TS/PRIMA-MPS, 100 images.
  ch5-mps           Chapter 5 MPS partitioning, 100 images.
  ch5-goodput       Chapter 5 goodput/latency percentile.
  ch5-clip          Chapter 5 CLIP generalization.
  ch5-overhead      Chapter 5 PRIMA predictor/calculator overhead.

Environment:
  ORIN_HOST          SSH host alias or host name. Default: orin1
  ORIN_PORT          Optional SSH port when ORIN_HOST is not an alias.
  ORIN_REPO          PRIMA repo path on Orin1. Default: /home/orin1/woosy/PRIMA
  ORIN_TTY           Use SSH tty for sudo prompts. Default: 1
  PULL               Run git pull --ff-only on Orin1 before execution. Default: 0
  DRY_RUN            Print the SSH command without executing it. Default: 0

Experiment overrides are passed through when set:
  NUM_IMAGES REPEATS REPEAT_START REPEAT_END MODE_LIST MPS_LEVELS
  UNIQUE_IMAGES LOOPS REQUESTS MODELS PRIMA_ARTIFACT_ROOT EXPERIMENT_ROOT
  YOLO_ROOT DATASET_ROOT CLIP_ROOT GOODPUT_ROOT GOODPUT_DATASET_ROOT
  GOODPUT_RESULT_OVERRIDE OUTPUT_DIR MODEL_CSV
MSG
}

target="${1:-}"
if [[ -z "$target" || "$target" == "-h" || "$target" == "--help" ]]; then
  usage
  exit 0
fi

ORIN_HOST="${ORIN_HOST:-orin1}"
ORIN_REPO="${ORIN_REPO:-/home/orin1/woosy/PRIMA}"
ORIN_TTY="${ORIN_TTY:-1}"
PULL="${PULL:-0}"
DRY_RUN="${DRY_RUN:-0}"

ssh_args=()
if [[ "$ORIN_TTY" != "0" ]]; then
  ssh_args+=("-tt")
fi
if [[ -n "${ORIN_PORT:-}" ]]; then
  ssh_args+=("-p" "$ORIN_PORT")
fi

env_args=()
append_env() {
  local name="$1"
  if [[ -n "${!name+x}" ]]; then
    env_args+=("$name=${!name}")
  fi
}

for name in \
  NUM_IMAGES REPEATS REPEAT_START REPEAT_END MODE_LIST MPS_LEVELS \
  UNIQUE_IMAGES UNIQUE_IMAGES_OBB LOOPS REQUESTS MODELS PRIMA_ARTIFACT_ROOT EXPERIMENT_ROOT \
  YOLO_ROOT DATASET_ROOT CLIP_ROOT GOODPUT_ROOT GOODPUT_DATASET_ROOT \
  GOODPUT_RESULT_OVERRIDE OUTPUT_DIR MODEL_CSV
do
  append_env "$name"
done

case "$target" in
  check-env)
    remote_cmd="./scripts/setup/check_orin_env.sh"
    ;;
  ch3-isolated)
    remote_cmd="./scripts/experiments/run_isolated_baseline.sh"
    ;;
  ch3-concurrent)
    env_args+=("MODE_LIST=ts_unlimited mps20_unlimited")
    remote_cmd="./scripts/experiments/run_concurrent_comparison.sh"
    ;;
  ch3-sweep)
    remote_cmd="./scripts/experiments/run_memory_sweep.sh"
    ;;
  ch5-concurrent)
    remote_cmd="./scripts/experiments/run_concurrent_comparison.sh"
    ;;
  ch5-mps)
    remote_cmd="./scripts/experiments/run_mps_partitioning.sh"
    ;;
  ch5-goodput)
    remote_cmd="./scripts/experiments/run_goodput_latency.sh"
    ;;
  ch5-clip)
    remote_cmd="./scripts/experiments/run_clip_generalization.sh"
    ;;
  ch5-overhead)
    remote_cmd="./scripts/experiments/run_prima_overhead.sh"
    ;;
  *)
    echo "[ERROR] Unknown target: $target" >&2
    usage >&2
    exit 2
    ;;
esac

printf -v quoted_repo '%q' "$ORIN_REPO"
env_prefix=""
if [[ "${#env_args[@]}" -gt 0 ]]; then
  printf -v env_prefix ' %q' "${env_args[@]}"
  env_prefix="env${env_prefix}"
fi

remote_script="set -euo pipefail; cd ${quoted_repo};"
if [[ "$PULL" == "1" ]]; then
  remote_script+=" git pull --ff-only;"
fi
if [[ -n "$env_prefix" ]]; then
  remote_script+=" ${env_prefix} ${remote_cmd};"
else
  remote_script+=" ${remote_cmd};"
fi

echo "[INFO] ${ORIN_HOST}: ${target}"
if [[ "$DRY_RUN" == "1" ]]; then
  printf '[DRY_RUN] ssh'
  printf ' %q' "${ssh_args[@]}" "$ORIN_HOST" "$remote_script"
  printf '\n'
  exit 0
fi

ssh "${ssh_args[@]}" "$ORIN_HOST" "$remote_script"
