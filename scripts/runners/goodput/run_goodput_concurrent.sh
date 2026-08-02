#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRIMA_ARTIFACT_ROOT="${PRIMA_ARTIFACT_ROOT:-/home/orin1/prima_artifacts}"
BASE="${GOODPUT_ROOT:-${PRIMA_ARTIFACT_ROOT}/goodput}"
YOLO="${YOLO_ROOT:-${PRIMA_ARTIFACT_ROOT}/yolo_runtime}"
GOODPUT_DATASET_ROOT="${GOODPUT_DATASET_ROOT:-$YOLO}"
PY="${SCRIPT_DIR}/repeat_goodput.py"
IMAGE="${IMAGE:-ultralytics/ultralytics:8.3.102-jetson-jetpack6}"
UNIQUE_IMAGES="${UNIQUE_IMAGES:-5000}"
UNIQUE_IMAGES_OBB="${UNIQUE_IMAGES_OBB:-auto}"
LOOPS="${LOOPS:-1}"
REQUESTS="${REQUESTS:-5000}"
RESULT="${GOODPUT_RESULT_OVERRIDE:-${BASE}/results/goodput_concurrent_${UNIQUE_IMAGES}x${LOOPS}_15W_1}"
STATE="${BASE}/goodput_concurrent_${UNIQUE_IMAGES}x${LOOPS}_15W.state"
SUMMARY="${RESULT}/summary.csv"
WORKLOADS=(classify detect pose segment obb)

mkdir -p "$RESULT"
echo "workload,exit_status,completed_requests,elapsed_sec,csv" > "$SUMMARY"

if [[ -s /proc/swaps ]] && [[ "$(wc -l < /proc/swaps)" -gt 1 ]]; then
  echo "failed_swap_enabled $(date --iso-8601=seconds)" > "$STATE"
  exit 1
fi

START_AT_NS=$(($(date +%s%N) + 30000000000))
echo "warming_up start_at_ns=${START_AT_NS} $(date --iso-8601=seconds)" > "$STATE"

declare -A PIDS
for workload in "${WORKLOADS[@]}"; do
  out="${RESULT}/${workload}"
  mkdir -p "$out"
  cname="prima-goodput-${workload}"
  docker rm -f "$cname" >/dev/null 2>&1 || true

  docker run --rm \
    --name "$cname" \
    --runtime=nvidia \
    --ipc=host \
    -e HOME=/tmp \
    -e YOLO_CONFIG_DIR=/tmp/ultralytics \
    -e START_AT_NS="$START_AT_NS" \
    -e UNIQUE_IMAGES="$UNIQUE_IMAGES" \
    -e UNIQUE_IMAGES_OBB="$UNIQUE_IMAGES_OBB" \
    -e TOTAL_REQUESTS="$REQUESTS" \
    -v "${YOLO}:/usr/src/ultralytics/yolo_new:ro" \
    -v "${GOODPUT_DATASET_ROOT}/expanded_imagenet_images:/usr/src/ultralytics/yolo_new/expanded_imagenet_images:ro" \
    -v "${GOODPUT_DATASET_ROOT}/expanded_coco_images:/usr/src/ultralytics/yolo_new/expanded_coco_images:ro" \
    -v "${GOODPUT_DATASET_ROOT}/expanded_dota_images:/usr/src/ultralytics/yolo_new/expanded_dota_images:ro" \
    -v "${PY}:/test/repeat_goodput.py:ro" \
    -v "${out}:/results" \
    "$IMAGE" \
    bash -lc "cd /usr/src/ultralytics/yolo_new && \
      python3 /test/repeat_goodput.py '${workload}' /results/inference_latency.csv" \
    > "${out}/run.log" 2>&1 &
  PIDS["$workload"]=$!
done

echo "running_all_5 start_at_ns=${START_AT_NS} $(date --iso-8601=seconds)" > "$STATE"

for workload in "${WORKLOADS[@]}"; do
  rc=0
  wait "${PIDS[$workload]}" || rc=$?
  out="${RESULT}/${workload}"
  count=0
  elapsed=""
  if [ -f "${out}/inference_latency.csv" ]; then
    count=$(awk 'END{print NR-1}' "${out}/inference_latency.csv")
  fi
  if [ -f "${out}/run.log" ]; then
    elapsed=$(awk -F= '/^elapsed_sec=/{print $2}' "${out}/run.log")
  fi
  echo "${workload},${rc},${count},${elapsed},${out}/inference_latency.csv" >> "$SUMMARY"
done

{
  echo "experiment=goodput_concurrent"
  echo "unique_images_per_dataset=${UNIQUE_IMAGES}"
  echo "unique_images_obb=${UNIQUE_IMAGES_OBB}"
  echo "repeat_count=${LOOPS}"
  echo "requests_per_workload=${REQUESTS}"
  echo "total_requests=$((REQUESTS * ${#WORKLOADS[@]}))"
  echo "workloads=classify,detect,pose,segment,obb"
  echo "execution=all_5_concurrent"
  echo "common_start_at_ns=${START_AT_NS}"
  echo "warmup_requests_per_workload=5"
  echo "save_results=false"
  echo "measurement=result.speed[inference]"
  echo "goodput_dataset_root=${GOODPUT_DATASET_ROOT}"
  echo "power_mode=$(nvpmodel -q | tr '\n' ' ')"
  echo "swap=off"
  echo "completed_at=$(date --iso-8601=seconds)"
} > "${RESULT}/experiment_config.txt"

echo "completed $(date --iso-8601=seconds)" > "$STATE"
