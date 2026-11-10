#!/usr/bin/env bash
# Build the clip-full:jetson image and pre-fetch CIFAR-100 and the CLIP weights
# into $CLIP_ROOT, so run_clip_generalization.sh runs offline (the inference
# scripts use download=False).
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/prima_common.sh"

BASE_IMAGE="${BASE_IMAGE:-ultralytics/ultralytics:8.3.102-jetson-jetpack6}"
CLIP_IMAGE="${CLIP_IMAGE:-clip-full:jetson}"
CLIP_ROOT="${CLIP_ROOT:-${PRIMA_ARTIFACT_ROOT}/clip}"
mkdir -p "$CLIP_ROOT"

if ! docker image inspect "$CLIP_IMAGE" >/dev/null 2>&1; then
  docker build -t "$CLIP_IMAGE" - <<DOCKERFILE
FROM ${BASE_IMAGE}
RUN pip3 install --no-cache-dir git+https://github.com/openai/CLIP.git
DOCKERFILE
fi

docker run --rm -v "${CLIP_ROOT}:/workspace" -w /workspace "$CLIP_IMAGE" \
  python3 -c 'from torchvision.datasets import CIFAR100; CIFAR100(root=".", download=True, train=False)'
for model in ViT-B/32 RN50; do
  docker run --rm --runtime=nvidia -v "${CLIP_ROOT}:/workspace" -w /workspace "$CLIP_IMAGE" \
    python3 -c "import clip; clip.load('${model}', device='cpu', download_root='.')"
done
command -v /usr/bin/time >/dev/null 2>&1 || echo "[WARN] /usr/bin/time missing: sudo apt install time"
ls -lh "$CLIP_ROOT"
