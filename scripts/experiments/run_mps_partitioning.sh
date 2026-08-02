#!/usr/bin/env bash
set -euo pipefail

PRIMA_ARTIFACT_ROOT="${PRIMA_ARTIFACT_ROOT:-/home/orin1/prima_artifacts}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${PRIMA_ARTIFACT_ROOT}/yolo}"
NUM_IMAGES="${NUM_IMAGES:-100}"
REPEATS="${REPEATS:-3}"
MPS_LEVELS="${MPS_LEVELS:-20,50}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

sudo -n true || {
  echo "[ERROR] sudo credential is not cached. Run 'sudo -v' before this script." >&2
  exit 1
}

IFS=',' read -r -a levels <<< "$MPS_LEVELS"
for level in "${levels[@]}"; do
  case "$level" in
    20)
      echo "[INFO] Running MPS 20% partitioning conditions with $NUM_IMAGES images."
      IMAGE_LIMIT="$NUM_IMAGES" \
      MODE_LIST="mps20_unlimited mps20_memlimit" \
      EXPERIMENT_ROOT="$EXPERIMENT_ROOT" \
        "${REPO_ROOT}/scripts/runners/yolo/run_concurrent_4mode.sh" "$REPEATS"
      ;;
    50)
      echo "[INFO] Running MPS 50% partitioning conditions with $NUM_IMAGES images."
      IMAGE_LIMIT="$NUM_IMAGES" \
      MODE_LIST="mps50_unlimited mps50_memlimit" \
      EXPERIMENT_ROOT="$EXPERIMENT_ROOT" \
        "${REPO_ROOT}/scripts/runners/yolo/run_mps50_2mode.sh" "$REPEATS"
      ;;
    100)
      cat >&2 <<'MSG'
[ERROR] MPS 100% / 8-SM-equivalent runner has not been found on Orin1 yet.
Provide the Orin1 or master-node script path, then wire it here.
MSG
      exit 3
      ;;
    *)
      echo "[ERROR] Unsupported MPS level: $level. Use comma-separated values from 20,50,100." >&2
      exit 2
      ;;
  esac
done
