#!/usr/bin/env bash
# Install the original paper workload scripts and image lists into the
# artifact workload directory used by every YOLO runner.
#
#   PRIMA_ARTIFACT_ROOT=... ./scripts/setup/install_workloads.sh   [FORCE=1 to overwrite]
#
# The scripts come from orin1, the node that produced the revised paper's
# results (prima_artifacts/yolo/workload). Keep them as they are: their
# behaviour determines the memory footprint the paper measures
# (see docs/workload_interface.md).
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/prima_common.sh"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="${REPO_ROOT}/workloads/yolo"
DEST="${WORKLOAD_DIR:-${PRIMA_ARTIFACT_ROOT}/yolo/workload}"
FORCE="${FORCE:-0}"

mkdir -p "${DEST}/image_lists"
for f in classify.py detect.py pose.py segment.py obb.py image_lists/imagenet_100.txt image_lists/coco_100.txt image_lists/dota_100.txt; do
  if [[ -e "${DEST}/${f}" && "$FORCE" != "1" ]] && ! cmp -s "${SRC}/${f}" "${DEST}/${f}"; then
    echo "[KEEP] ${DEST}/${f} differs from the original (FORCE=1 to replace; a copy is kept as .bak)"
    continue
  fi
  [[ -e "${DEST}/${f}" ]] && ! cmp -s "${SRC}/${f}" "${DEST}/${f}" && cp -p "${DEST}/${f}" "${DEST}/${f}.bak"
  cp -p "${SRC}/${f}" "${DEST}/${f}"
  echo "[OK] ${DEST}/${f}"
done
