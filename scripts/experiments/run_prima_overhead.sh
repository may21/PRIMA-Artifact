#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/prima_common.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/prima_overhead}"
REPEATS="${REPEATS:-30}"
# onnx_path entries may use ${PRIMA_ARTIFACT_ROOT}; files missing from
# onnx_cache are also looked up in ${PRIMA_ARTIFACT_ROOT}/yolo/workload.
MODEL_CSV="${MODEL_CSV:-${REPO_ROOT}/configs/workloads/overhead_models.csv}"

cd "$REPO_ROOT"
python3 "${REPO_ROOT}/scripts/runners/overhead/measure_prima_overhead.py" \
  --model-csv "$MODEL_CSV" \
  --output-dir "$OUTPUT_DIR" \
  --repeats "$REPEATS" \
  --model "${REPO_ROOT}/prima/predictor/rf_model.pkl" \
  --feature-order "${REPO_ROOT}/prima/predictor/feature_order.json"
