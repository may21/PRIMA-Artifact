#!/usr/bin/env bash
set -euo pipefail

echo "[check] uname"
uname -a || true

echo "[check] python"
python3 --version

echo "[check] docker"
docker --version

echo "[check] nvidia runtime"
docker info 2>/dev/null | grep -i nvidia || true

echo "[check] tegrastats"
command -v tegrastats

echo "[check] done"

