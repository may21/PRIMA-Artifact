#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-results/prima_overhead}"
REPEATS="${REPEATS:-30}"
MODEL_CSV="${MODEL_CSV:-configs/workloads/overhead_models.csv}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

python3 "${REPO_ROOT}/scripts/runners/overhead/measure_prima_overhead.py" \
  --model-csv "$MODEL_CSV" \
  --output-dir "$OUTPUT_DIR" \
  --repeats "$REPEATS"
