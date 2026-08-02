#!/usr/bin/env bash
set -euo pipefail

PRIMA_ARTIFACT_ROOT="${PRIMA_ARTIFACT_ROOT:-/home/orin1/prima_artifacts}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${PRIMA_ARTIFACT_ROOT}/yolo}"
NUM_IMAGES="${NUM_IMAGES:-100}"
REPEAT_START="${REPEAT_START:-1}"
REPEAT_END="${REPEAT_END:-3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

sudo -n true || {
  echo "[ERROR] sudo credential is not cached. Run 'sudo -v' before this script." >&2
  exit 1
}

IMAGE_LIMIT="$NUM_IMAGES" \
REPEAT_START="$REPEAT_START" \
REPEAT_END="$REPEAT_END" \
EXPERIMENT_ROOT="$EXPERIMENT_ROOT" \
  "${REPO_ROOT}/scripts/runners/yolo/run_isolated_baseline.sh"
