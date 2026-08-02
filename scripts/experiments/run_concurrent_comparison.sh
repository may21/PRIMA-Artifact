#!/usr/bin/env bash
set -euo pipefail

PRIMA_ARTIFACT_ROOT="${PRIMA_ARTIFACT_ROOT:-/home/orin1/prima_artifacts}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${PRIMA_ARTIFACT_ROOT}/yolo}"
NUM_IMAGES="${NUM_IMAGES:-100}"
REPEATS="${REPEATS:-3}"
MODE_LIST="${MODE_LIST:-ts_unlimited mps20_unlimited ts_memlimit mps20_memlimit}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

sudo -n true || {
  echo "[ERROR] sudo credential is not cached. Run 'sudo -v' before this script." >&2
  exit 1
}

IMAGE_LIMIT="$NUM_IMAGES" \
MODE_LIST="$MODE_LIST" \
EXPERIMENT_ROOT="$EXPERIMENT_ROOT" \
  "${REPO_ROOT}/scripts/runners/yolo/run_concurrent_4mode.sh" "$REPEATS"
