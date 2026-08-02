#!/usr/bin/env bash
set -euo pipefail

LOG_PATH="${1:-/tmp/prima_tegrastats.log}"
INTERVAL_MS="${2:-1000}"

mkdir -p "$(dirname "$LOG_PATH")"
tegrastats --interval "$INTERVAL_MS" --logfile "$LOG_PATH" &
echo $! > /tmp/prima_tegrastats.pid
echo "tegrastats started: pid=$(cat /tmp/prima_tegrastats.pid), log=$LOG_PATH"

