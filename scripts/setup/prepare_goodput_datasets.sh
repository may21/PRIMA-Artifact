#!/usr/bin/env bash
set -euo pipefail

ROOT="${GOODPUT_DATASET_ROOT:-/mnt/prima_usb/prima_goodput_5000}"
DOWNLOAD_DIR="${GOODPUT_DOWNLOAD_DIR:-${ROOT}/_downloads}"
COCO_URL="${COCO_URL:-http://images.cocodataset.org/zips/val2017.zip}"
IMAGENET_URL="${IMAGENET_URL:-https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar}"
DOTA_VAL_URL="${DOTA_VAL_URL:-https://drive.google.com/drive/folders/1n5w45suVOyaqY84hltJhIZdtVFD9B224?usp=sharing}"
DOTA_TEST_URL="${DOTA_TEST_URL:-https://drive.google.com/drive/folders/1mYOf5USMGNcJRPcvRVJVV1uHEalG5RPl?usp=sharing}"

COCO_DIR="${ROOT}/expanded_coco_images"
IMAGENET_DIR="${ROOT}/expanded_imagenet_images"
DOTA_DIR="${ROOT}/expanded_dota_images"

mkdir -p "$COCO_DIR" "$IMAGENET_DIR" "$DOTA_DIR" "$DOWNLOAD_DIR"

count_images() {
  find "$1" -maxdepth 1 -type f \
    \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" \) \
    2>/dev/null | wc -l
}

copy_first_n_flat() {
  local src="$1"
  local dst="$2"
  local limit="$3"
  mkdir -p "$dst"
  find "$src" -type f \
    \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" \) \
    | sort | head -n "$limit" | while IFS= read -r file_path; do
      cp -n "$file_path" "$dst/$(basename "$file_path")"
    done
}

require_at_least() {
  local name="$1"
  local dir="$2"
  local expected="$3"
  local actual
  actual="$(count_images "$dir")"
  if [ "$actual" -lt "$expected" ]; then
    echo "[ERROR] ${name}: expected at least ${expected}, found ${actual} in ${dir}" >&2
    exit 1
  fi
}

echo "[START] $(date --iso-8601=seconds)"
df -h "$ROOT" || true

if [ "$(count_images "$COCO_DIR")" -lt 5000 ]; then
  echo "[COCO] downloading val2017.zip"
  wget -nv -c -O "${DOWNLOAD_DIR}/val2017.zip" "$COCO_URL"
  mkdir -p "${DOWNLOAD_DIR}/coco_val2017"
  echo "[COCO] extracting"
  unzip -q -n "${DOWNLOAD_DIR}/val2017.zip" -d "${DOWNLOAD_DIR}/coco_val2017"
  echo "[COCO] copying 5000 images"
  copy_first_n_flat "${DOWNLOAD_DIR}/coco_val2017/val2017" "$COCO_DIR" 5000
fi
require_at_least "COCO" "$COCO_DIR" 5000
echo "[COCO] count=$(count_images "$COCO_DIR")"

if [ "$(count_images "$IMAGENET_DIR")" -lt 5000 ]; then
  echo "[IMAGENET] downloading validation tar"
  wget -nv -c -O "${DOWNLOAD_DIR}/ILSVRC2012_img_val.tar" "$IMAGENET_URL"
  echo "[IMAGENET] extracting first 5000 validation images"
  tar -tf "${DOWNLOAD_DIR}/ILSVRC2012_img_val.tar" \
    | grep -Ei '\.(jpg|jpeg|png|bmp)$' \
    | sort | head -n 5000 > "${DOWNLOAD_DIR}/imagenet_val_5000.txt"
  tar -xf "${DOWNLOAD_DIR}/ILSVRC2012_img_val.tar" \
    -C "$IMAGENET_DIR" \
    -T "${DOWNLOAD_DIR}/imagenet_val_5000.txt"
fi
require_at_least "ImageNet" "$IMAGENET_DIR" 5000
echo "[IMAGENET] count=$(count_images "$IMAGENET_DIR")"

if [ ! -f "${DOTA_DIR}/.dota_complete" ]; then
  if ! python3 - <<'PY' >/dev/null 2>&1
import gdown
PY
  then
    echo "[DOTA] installing gdown"
    python3 -m pip install --user gdown
  fi

  export PATH="${HOME}/.local/bin:${PATH}"
  rm -rf "${DOWNLOAD_DIR}/dota_val" "${DOWNLOAD_DIR}/dota_test" "${DOWNLOAD_DIR}/dota_extract"
  mkdir -p "${DOWNLOAD_DIR}/dota_val" "${DOWNLOAD_DIR}/dota_test" "${DOWNLOAD_DIR}/dota_extract"

  echo "[DOTA] downloading official validation folder"
  python3 -m gdown --folder "$DOTA_VAL_URL" -O "${DOWNLOAD_DIR}/dota_val"
  echo "[DOTA] downloading official testing-images folder"
  python3 -m gdown --folder "$DOTA_TEST_URL" -O "${DOWNLOAD_DIR}/dota_test"

  echo "[DOTA] extracting archives"
  find "${DOWNLOAD_DIR}/dota_val" "${DOWNLOAD_DIR}/dota_test" -type f -iname "*.zip" \
    | sort | while IFS= read -r archive_path; do
      unzip -q -n "$archive_path" -d "${DOWNLOAD_DIR}/dota_extract"
    done

  echo "[DOTA] copying val/test original images"
  find "$DOTA_DIR" -maxdepth 1 -type f \
    \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" \) \
    -delete
  copy_first_n_flat "${DOWNLOAD_DIR}/dota_extract" "$DOTA_DIR" 999999
  copy_first_n_flat "${DOWNLOAD_DIR}/dota_val" "$DOTA_DIR" 999999
  copy_first_n_flat "${DOWNLOAD_DIR}/dota_test" "$DOTA_DIR" 999999
  touch "${DOTA_DIR}/.dota_complete"
fi

if [ "$(count_images "$DOTA_DIR")" -eq 0 ]; then
  echo "[ERROR] DOTA: no images prepared in ${DOTA_DIR}" >&2
  exit 1
fi
echo "[DOTA] count=$(count_images "$DOTA_DIR")"

echo "[DONE] $(date --iso-8601=seconds)"
du -sh "$ROOT" "$DOWNLOAD_DIR" 2>/dev/null || true
df -h "$ROOT" || true
