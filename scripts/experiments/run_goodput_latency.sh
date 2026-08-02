#!/usr/bin/env bash
set -euo pipefail

PRIMA_ARTIFACT_ROOT="${PRIMA_ARTIFACT_ROOT:-/home/orin1/prima_artifacts}"
GOODPUT_ROOT="${GOODPUT_ROOT:-${PRIMA_ARTIFACT_ROOT}/goodput}"
UNIQUE_IMAGES="${UNIQUE_IMAGES:-5000}"
UNIQUE_IMAGES_OBB="${UNIQUE_IMAGES_OBB:-auto}"
LOOPS="${LOOPS:-1}"
REQUESTS="${REQUESTS:-5000}"
GOODPUT_DATASET_ROOT="${GOODPUT_DATASET_ROOT:-/mnt/prima_usb/prima_goodput_5000}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ -s /proc/swaps ]] && [[ "$(wc -l < /proc/swaps)" -gt 1 ]]; then
  echo "[ERROR] Host swap is enabled. Disable swap before running Chapter 5 goodput." >&2
  cat /proc/swaps >&2
  exit 1
fi

GOODPUT_ROOT="$GOODPUT_ROOT" \
GOODPUT_DATASET_ROOT="$GOODPUT_DATASET_ROOT" \
UNIQUE_IMAGES="$UNIQUE_IMAGES" \
UNIQUE_IMAGES_OBB="$UNIQUE_IMAGES_OBB" \
LOOPS="$LOOPS" \
REQUESTS="$REQUESTS" \
  "${REPO_ROOT}/scripts/runners/goodput/run_goodput_concurrent.sh"

organize_root="${GOODPUT_RESULT_OVERRIDE:-${GOODPUT_ROOT}/results/goodput_concurrent_${UNIQUE_IMAGES}x${LOOPS}_15W_1}"
python3 "${REPO_ROOT}/scripts/runners/goodput/organize_goodput.py" "$organize_root"
