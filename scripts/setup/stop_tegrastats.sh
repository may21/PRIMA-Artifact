#!/usr/bin/env bash
set -euo pipefail

if [ -f /tmp/prima_tegrastats.pid ]; then
  kill "$(cat /tmp/prima_tegrastats.pid)" 2>/dev/null || true
  rm -f /tmp/prima_tegrastats.pid
fi

pkill -f "tegrastats.*prima_tegrastats" 2>/dev/null || true
echo "tegrastats stopped"

