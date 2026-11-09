#!/usr/bin/env bash
set -euo pipefail

# Allocation-policy baselines for the second major revision (reviewer comment C).
#
# Runs the five YOLO workloads concurrently under time slicing with four
# memory-allocation policies that share the same total budget:
#   ts_unlimited  None   : no container memory limit (paper TS baseline)
#   ts_equal      Equal  : AMU / 5 for every container
#   ts_oracle     Oracle : AMU split in proportion to MEASURED isolated usage
#   ts_memlimit   PRIMA  : AMU split in proportion to PREDICTED usage (paper)
#
# Oracle budgets need the isolated-baseline results (run_isolated_baseline.sh)
# or explicit measured values:
#   ISOLATED_RESULT_ROOT=/home/orin1/prima_artifacts/yolo/results/isolated_no_memlimit_100_1 \
#     ./scripts/experiments/run_allocation_baselines.sh
#   ORACLE_MEASURED_MIB="classify=600 detect=800 pose=800 segment=900 obb=2000" \
#     ./scripts/experiments/run_allocation_baselines.sh
# A previously generated BUDGET_FILE can be reused instead.
# MPS counterparts (mps20_equal, mps20_oracle) are supported through MODE_LIST
# once the TS results have been examined.

PRIMA_ARTIFACT_ROOT="${PRIMA_ARTIFACT_ROOT:-/home/orin1/prima_artifacts}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${PRIMA_ARTIFACT_ROOT}/yolo}"
NUM_IMAGES="${NUM_IMAGES:-100}"
REPEATS="${REPEATS:-3}"
MODE_LIST="${MODE_LIST:-ts_unlimited ts_equal ts_oracle ts_memlimit}"
TOTAL_BUDGET_MIB="${TOTAL_BUDGET_MIB:-6842}"
ORACLE_STATISTIC="${ORACLE_STATISTIC:-max}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUNNER_DIR="${REPO_ROOT}/scripts/runners/yolo"

sudo -n true || {
  echo "[ERROR] sudo credential is not cached. Run 'sudo -v' before this script." >&2
  exit 1
}

RESULT_BASE="${RESULT_BASE:-${EXPERIMENT_ROOT}/results}"
if [[ -n "${RESULT_ROOT_OVERRIDE:-}" ]]; then
  RESULT_ROOT="$RESULT_ROOT_OVERRIDE"
else
  RUN_INDEX=1
  while [[ -e "${RESULT_BASE}/allocation_baselines_${RUN_INDEX}" ]]; do
    RUN_INDEX=$((RUN_INDEX + 1))
  done
  RESULT_ROOT="${RESULT_BASE}/allocation_baselines_${RUN_INDEX}"
fi
mkdir -p "$RESULT_ROOT"

needs_oracle=0
[[ " $MODE_LIST " == *_oracle* ]] && needs_oracle=1

BUDGET_FILE="${BUDGET_FILE:-${RESULT_ROOT}/allocation_budgets.env}"
if [[ "$needs_oracle" -eq 1 && ! -f "$BUDGET_FILE" ]]; then
  compute_args=(--total-budget-mib "$TOTAL_BUDGET_MIB" --env-output "$BUDGET_FILE"
                --csv-output "${RESULT_ROOT}/allocation_budgets.csv")
  if [[ -n "${ORACLE_MEASURED_MIB:-}" ]]; then
    compute_args+=(--measured "$ORACLE_MEASURED_MIB")
  elif [[ -n "${ISOLATED_RESULT_ROOT:-}" ]]; then
    compute_args+=(--isolated-root "$ISOLATED_RESULT_ROOT" --statistic "$ORACLE_STATISTIC")
  else
    latest="$(ls -d "${RESULT_BASE}"/isolated_no_memlimit_100_* 2>/dev/null | sort -V | tail -1 || true)"
    if [[ -n "$latest" && -f "${latest}/organized/isolated_no_memlimit_pidstat_memory_peak_summary.csv" ]]; then
      echo "[INFO] using isolated baseline results from ${latest}"
      compute_args+=(--isolated-root "$latest" --statistic "$ORACLE_STATISTIC")
    else
      echo "[ERROR] oracle budgets need ISOLATED_RESULT_ROOT, ORACLE_MEASURED_MIB, or BUDGET_FILE." >&2
      exit 2
    fi
  fi
  python3 "${RUNNER_DIR}/compute_allocation_baselines.py" "${compute_args[@]}"
fi

if [[ -f "$BUDGET_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$BUDGET_FILE"
  cp -f "$BUDGET_FILE" "${RESULT_ROOT}/allocation_budgets.env" 2>/dev/null || true
fi
export TOTAL_BUDGET_MIB EQUAL_BUDGET ORACLE_BUDGETS

echo "[INFO] modes: $MODE_LIST"
echo "[INFO] total budget: ${TOTAL_BUDGET_MIB} MiB, equal: ${EQUAL_BUDGET:-$((TOTAL_BUDGET_MIB / 5))m}, oracle: ${ORACLE_BUDGETS:-unset}"

IMAGE_LIMIT="$NUM_IMAGES" \
MODE_LIST="$MODE_LIST" \
EXPERIMENT_ROOT="$EXPERIMENT_ROOT" \
RESULT_ROOT_OVERRIDE="$RESULT_ROOT" \
  "${RUNNER_DIR}/run_concurrent_4mode.sh" "$REPEATS"

python3 "${RUNNER_DIR}/summarize_allocation_baselines.py" "$RESULT_ROOT"
echo "[DONE] allocation baselines: $RESULT_ROOT"
