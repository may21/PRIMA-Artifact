#!/usr/bin/env bash
# Prepare the 100-image inputs and image lists used by the isolated, memory
# sweep, concurrent, MPS and allocation-baseline runners.
#
#   PRIMA_ARTIFACT_ROOT=... [PRIMA_SSD_ROOT=...] ./scripts/setup/prepare_yolo_inputs.sh
#
# Source images: $SOURCE_ROOT/expanded_{imagenet,coco,dota}_images
#   (default: $PRIMA_SSD_ROOT/prima_goodput_5000, see prepare_goodput_datasets.sh)
# Destination:   $DATASET_ROOT (default: the root the runners resolve, i.e.
#   $PRIMA_ARTIFACT_ROOT/yolo_runtime if it exists, else .../yolo/workload)
# Image lists:   $PRIMA_ARTIFACT_ROOT/yolo/workload/image_lists/{imagenet,coco,dota}_<N>.txt
#
# By default the paper's own image lists (workloads/yolo/image_lists) are used
# and every listed image must already be in the destination: the paper set is
# the orin1 "expanded_*" folders (full-size DOTA originals, ~4.4 MPix on
# average, named NNN_<original>). USE_ORIGINAL_LISTS=0 instead copies the first
# N images from SOURCE_ROOT and lists them, which does NOT reproduce the paper's
# memory footprint.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/prima_common.sh"

NUM_IMAGES="${NUM_IMAGES:-100}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${PRIMA_ARTIFACT_ROOT}/yolo}"
SOURCE_ROOT="${SOURCE_ROOT:-${PRIMA_SSD_ROOT}/prima_goodput_5000}"
DEST_ROOT="${DATASET_ROOT:-$(prima_resolve_dataset_root "$EXPERIMENT_ROOT")}"
LIST_DIR="${EXPERIMENT_ROOT}/workload/image_lists"
CONTAINER_DIR=/usr/src/ultralytics/yolo_new
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

mkdir -p "$LIST_DIR"
declare -A LIST_NAME=([expanded_imagenet_images]=imagenet [expanded_coco_images]=coco [expanded_dota_images]=dota)

for dataset in expanded_imagenet_images expanded_coco_images expanded_dota_images; do
  dest="${DEST_ROOT}/${dataset}"
  mkdir -p "$dest"
  have="$(find "$dest" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)"
  original="${REPO_ROOT}/workloads/yolo/image_lists/${LIST_NAME[$dataset]}_${NUM_IMAGES}.txt"
  use_original=0
  [[ -f "$original" && "${USE_ORIGINAL_LISTS:-1}" == "1" ]] && use_original=1
  if (( use_original == 0 && have < NUM_IMAGES )); then
    [[ -d "${SOURCE_ROOT}/${dataset}" ]] || {
      echo "[ERROR] ${dest} has ${have} images and no source ${SOURCE_ROOT}/${dataset}" >&2
      exit 1
    }
    find "${SOURCE_ROOT}/${dataset}" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) \
      | sort | head -n "$NUM_IMAGES" | while IFS= read -r file; do cp -n "$file" "$dest/"; done
  fi
  list="${LIST_DIR}/${LIST_NAME[$dataset]}_${NUM_IMAGES}.txt"
  if (( use_original == 1 )); then
    # Paper image set: every listed file must be present in the dataset root.
    cp -f "$original" "$list"
    missing=0
    while IFS= read -r entry; do
      [[ -f "${dest}/$(basename "$entry")" ]] || missing=$((missing + 1))
    done < "$list"
    if (( missing > 0 )); then
      echo "[ERROR] ${missing} of the paper's ${LIST_NAME[$dataset]} images are missing from ${dest}." >&2
      echo "        Copy the original expanded_* folders (see README), or set USE_ORIGINAL_LISTS=0 to list whatever is there." >&2
      exit 1
    fi
  else
    # Ultralytics reads a .txt source as one path per line, resolved in the container.
    find "$dest" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) -printf '%f\n' \
      | sort | head -n "$NUM_IMAGES" | sed "s#^#${CONTAINER_DIR}/${dataset}/#" > "$list"
  fi
  count="$(wc -l < "$list")"
  echo "[OK] ${dataset}: $(find "$dest" -maxdepth 1 -type f | wc -l) files in ${dest}; ${count} entries in ${list}"
  (( count == NUM_IMAGES )) || { echo "[ERROR] ${list} has ${count} entries, expected ${NUM_IMAGES}" >&2; exit 1; }
done

# Runners whose DATASET_ROOT differs from the workload directory still mount
# the lists from yolo/workload; keep a copy next to the images for clarity.
if [[ "$DEST_ROOT" != "${EXPERIMENT_ROOT}/workload" ]]; then
  mkdir -p "${DEST_ROOT}/image_lists" && cp -f "${LIST_DIR}"/*_"${NUM_IMAGES}".txt "${DEST_ROOT}/image_lists/"
fi
