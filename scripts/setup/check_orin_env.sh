#!/usr/bin/env bash
# Check the edge node before running PRIMA experiments. Prints every problem
# found and exits non-zero if any required item is missing.
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/prima_common.sh"

IMAGE="${IMAGE:-ultralytics/ultralytics:8.3.102-jetson-jetpack6}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${PRIMA_ARTIFACT_ROOT}/yolo}"
failures=0
ok()   { echo "[ OK ] $*"; }
warn() { echo "[WARN] $*"; }
fail() { echo "[FAIL] $*"; failures=$((failures + 1)); }

echo "[info] $(uname -r) | $(tr -d '\0' </proc/device-tree/model 2>/dev/null || echo unknown device)"
echo "[info] PRIMA_ARTIFACT_ROOT=${PRIMA_ARTIFACT_ROOT}  PRIMA_SSD_ROOT=${PRIMA_SSD_ROOT}"
echo "[info] power mode: $(nvpmodel -q 2>/dev/null | tr '\n' ' ' || echo unknown)"

for cmd in python3 docker pidstat mpstat vmstat top tegrastats nvidia-cuda-mps-control; do
  command -v "$cmd" >/dev/null 2>&1 && ok "$cmd" || fail "missing command: $cmd"
done
[[ -x /usr/bin/time ]] && ok "/usr/bin/time" || warn "/usr/bin/time missing (CLIP runner): sudo apt install time"
perf_bin="$(prima_resolve_perf)" && ok "perf: ${perf_bin}" || fail "no working perf for $(uname -r) (install linux-nvidia-tegra-tools-* or set PERF_BIN)"
docker info 2>/dev/null | grep -qi nvidia && ok "docker nvidia runtime" || fail "docker nvidia runtime not found"
docker image inspect "$IMAGE" >/dev/null 2>&1 && ok "image ${IMAGE}" || fail "missing image ${IMAGE} (docker pull ${IMAGE})"
[[ "$(stat -fc %T /sys/fs/cgroup)" == "cgroup2fs" ]] && ok "cgroup v2" || fail "cgroup v2 required"
if [[ -s /proc/swaps ]] && [[ "$(wc -l < /proc/swaps)" -gt 1 ]]; then warn "swap is on (runners call swapoff -a)"; else ok "swap off"; fi

workload_dir="${EXPERIMENT_ROOT}/workload"
for f in classify.py detect.py pose.py segment.py obb.py \
         yolov8n-cls.engine yolov8n.engine yolov8n-pose.engine yolov8n-seg.engine yolov8n-obb.engine; do
  [[ -f "${workload_dir}/${f}" ]] && ok "workload/${f}" || fail "missing ${workload_dir}/${f}"
done
for list in imagenet_100 coco_100 dota_100; do
  [[ -f "${workload_dir}/image_lists/${list}.txt" ]] && ok "image_lists/${list}.txt" \
    || fail "missing ${workload_dir}/image_lists/${list}.txt (scripts/setup/prepare_yolo_inputs.sh)"
done
dataset_root="$(prima_resolve_dataset_root "$EXPERIMENT_ROOT")"
for d in expanded_imagenet_images expanded_coco_images expanded_dota_images; do
  n="$(find "${dataset_root}/${d}" -maxdepth 1 -type f 2>/dev/null | wc -l)"
  (( n > 0 )) && ok "${dataset_root}/${d}: ${n} files" || fail "missing images in ${dataset_root}/${d}"
done
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
for f in classify.py detect.py pose.py segment.py obb.py; do
  [[ -f "${workload_dir}/${f}" ]] && ! cmp -s "${workload_dir}/${f}" "${repo_root}/workloads/yolo/${f}" \
    && warn "workload/${f} differs from the paper version in workloads/yolo (scripts/setup/install_workloads.sh)"
done
for ds in imagenet coco dota; do
  list="${repo_root}/workloads/yolo/image_lists/${ds}_100.txt"
  [[ -f "$list" ]] || continue
  missing=0
  while IFS= read -r entry; do [[ -f "${dataset_root}/expanded_${ds}_images/$(basename "$entry")" ]] || missing=$((missing + 1)); done < "$list"
  (( missing == 0 )) && ok "paper ${ds} image set present" || warn "${missing}/100 paper ${ds} images missing from ${dataset_root}/expanded_${ds}_images"
done
for f in "${PRIMA_ARTIFACT_ROOT}"/onnx_cache/yolov8n{-cls,,-pose,-seg,-obb}.onnx; do
  [[ -f "$f" ]] && ok "$(basename "$f")" || warn "missing ${f} (overhead experiment; export_yolo_engines.sh copies it)"
done

if (( failures > 0 )); then echo "[done] ${failures} problem(s) found"; exit 1; fi
echo "[done] environment ready"
