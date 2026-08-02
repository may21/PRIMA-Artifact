#!/usr/bin/env bash
set -euo pipefail

PRIMA_ARTIFACT_ROOT="${PRIMA_ARTIFACT_ROOT:-/home/orin1/prima_artifacts}"
CLIP_ROOT="${CLIP_ROOT:-${PRIMA_ARTIFACT_ROOT}/clip}"
REPEATS="${REPEATS:-3}"
MODELS="${MODELS:-vitb32,rn50}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

IFS=',' read -r -a models <<< "$MODELS"
for model in "${models[@]}"; do
  for repeat in $(seq 1 "$REPEATS"); do
    case "$model" in
      vitb32|ViT-B-32|vit-b-32)
        echo "[INFO] CLIP ViT-B/32 repeat ${repeat}/${REPEATS}"
        CLIP_ROOT="$CLIP_ROOT" "${REPO_ROOT}/scripts/runners/clip/run_clip_vitb32_docker.sh"
        ;;
      rn50|RN50)
        echo "[INFO] CLIP RN50 repeat ${repeat}/${REPEATS}"
        CLIP_ROOT="$CLIP_ROOT" "${REPO_ROOT}/scripts/runners/clip/run_clip_rn50_docker.sh"
        ;;
      *)
        echo "[ERROR] Unsupported CLIP model: $model. Use MODELS=vitb32,rn50." >&2
        exit 2
        ;;
    esac
  done
done

CLIP_ROOT="$CLIP_ROOT" python3 "${REPO_ROOT}/scripts/runners/clip/summarize_clip_memory.py"
