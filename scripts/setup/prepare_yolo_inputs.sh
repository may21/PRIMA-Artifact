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
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/prima_common.sh"

NUM_IMAGES="${NUM_IMAGES:-100}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${PRIMA_ARTIFACT_ROOT}/yolo}"
SOURCE_ROOT="${SOURCE_ROOT:-${PRIMA_SSD_ROOT}/prima_goodput_5000}"
DEST_ROOT="${DATASET_ROOT:-$(prima_resolve_dataset_root "$EXPERIMENT_ROOT")}"
LIST_DIR="${EXPERIMENT_ROOT}/workload/image_lists"

mkdir -p "$LIST_DIR"
declare -A LIST_NAME=([expanded_imagenet_images]=imagenet [expanded_coco_images]=coco [expanded_dota_images]=dota)

for dataset in expanded_imagenet_images expanded_coco_images expanded_dota_images; do
  dest="${DEST_ROOT}/${dataset}"
  mkdir -p "$dest"
  have="$(find "$dest" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)"
  if (( have < NUM_IMAGES )); then
    [[ -d "${SOURCE_ROOT}/${dataset}" ]] || {
      echo "[ERROR] ${dest} has ${have} images and no source ${SOURCE_ROOT}/${dataset}" >&2
      exit 1
    }
    find "${SOURCE_ROOT}/${dataset}" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) \
      | sort | head -n "$NUM_IMAGES" | while IFS= read -r file; do cp -n "$file" "$dest/"; done
  fi
  list="${LIST_DIR}/${LIST_NAME[$dataset]}_${NUM_IMAGES}.txt"
  find "$dest" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) -printf '%f\n' \
    | sort | head -n "$NUM_IMAGES" > "$list"
  count="$(wc -l < "$list")"
  echo "[OK] ${dataset}: $(find "$dest" -maxdepth 1 -type f | wc -l) files in ${dest}; ${count} entries in ${list}"
  (( count == NUM_IMAGES )) || { echo "[ERROR] ${list} has ${count} entries, expected ${NUM_IMAGES}" >&2; exit 1; }
done

# Runners whose DATASET_ROOT differs from the workload directory still mount
# the lists from yolo/workload; keep a copy next to the images for clarity.
if [[ "$DEST_ROOT" != "${EXPERIMENT_ROOT}/workload" ]]; then
  mkdir -p "${DEST_ROOT}/image_lists" && cp -f "${LIST_DIR}"/*_"${NUM_IMAGES}".txt "${DEST_ROOT}/image_lists/"
fi
