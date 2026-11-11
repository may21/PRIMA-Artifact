#!/usr/bin/env bash
# Export the five YOLOv8n TensorRT engines (and ONNX graphs for the Predictor)
# used by every YOLO experiment: 640x640, batch 1, FP32, static shape.
#
#   PRIMA_ARTIFACT_ROOT=/path/to/prima_artifacts ./scripts/setup/export_yolo_engines.sh
#
# Output: $PRIMA_ARTIFACT_ROOT/yolo/workload/yolov8n{-cls,,-pose,-seg,-obb}.{pt,onnx,engine}
# and a copy of the ONNX files in $PRIMA_ARTIFACT_ROOT/onnx_cache.
# Engines are device-specific: export them on the Jetson that runs the experiments.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/prima_common.sh"

IMAGE="${IMAGE:-ultralytics/ultralytics:8.3.102-jetson-jetpack6}"
WORKLOAD_DIR="${WORKLOAD_DIR:-${PRIMA_ARTIFACT_ROOT}/yolo/workload}"
ONNX_CACHE="${ONNX_CACHE:-${PRIMA_ARTIFACT_ROOT}/onnx_cache}"
MODELS="${MODELS:-yolov8n-cls yolov8n yolov8n-pose yolov8n-seg yolov8n-obb}"
FORCE="${FORCE:-0}"

mkdir -p "$WORKLOAD_DIR" "$ONNX_CACHE"
docker image inspect "$IMAGE" >/dev/null 2>&1 || docker pull "$IMAGE"

for model in $MODELS; do
  if [[ "$FORCE" != "1" && -f "${WORKLOAD_DIR}/${model}.engine" && -f "${WORKLOAD_DIR}/${model}.onnx" ]]; then
    echo "[SKIP] ${model}: engine and onnx exist (FORCE=1 to rebuild)"
  else
    echo "[EXPORT] ${model}"
    docker run --rm --runtime=nvidia --ipc=host \
      --user "$(id -u):$(id -g)" -e HOME=/tmp -e YOLO_CONFIG_DIR=/tmp/ultralytics \
      -v "${WORKLOAD_DIR}:/workspace" -w /workspace "$IMAGE" \
      bash -lc "yolo export model=${model}.pt format=onnx imgsz=640 batch=1 opset=17 simplify=False && \
                yolo export model=${model}.pt format=engine imgsz=640 batch=1 half=False device=0"
  fi
  cp -f "${WORKLOAD_DIR}/${model}.onnx" "${ONNX_CACHE}/${model}.onnx"
done

ls -lh "${WORKLOAD_DIR}"/*.engine "${ONNX_CACHE}"/*.onnx
