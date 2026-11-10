#!/bin/bash

OUTPUT_DIR="gpu_five_result"
GPU_FILE="${OUTPUT_DIR}/gpu_usage.txt"

if [ -d "$OUTPUT_DIR" ]; then
    rm -rf "$OUTPUT_DIR"
fi
mkdir "$OUTPUT_DIR"

if [ -f "$GPU_FILE" ]; then
    rm "$GPU_FILE"
fi

tegrastats --interval 1000 > tegrastats_output.log &
TEGRA_PID=$!

cleanup() {
    echo "Cleaning up..."
    if ps -p $TEGRA_PID > /dev/null; then
        kill -9 $TEGRA_PID
        wait $TEGRA_PID 2>/dev/null
    fi
    rm -f tegrastats_output.log
    exit
}
trap cleanup SIGINT SIGTERM EXIT

INTERVAL=1

while true; do
    START_TIME=$(date +%s.%N)

    GPU_USAGE=$(tail -n 1 tegrastats_output.log | grep -oP "GR3D_FREQ \K[0-9]+")
    GPU_USAGE=${GPU_USAGE:-0}

    TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
    echo "$TIMESTAMP, GPU_USAGE=$GPU_USAGE%" >> "$GPU_FILE"

    END_TIME=$(date +%s.%N)
    ELAPSED_TIME=$(awk "BEGIN {print $END_TIME - $START_TIME}")
    SLEEP_TIME=$(awk "BEGIN {print $INTERVAL - $ELAPSED_TIME}")
    if (( $(echo "$SLEEP_TIME > 0" | bc -l) )); then
        sleep "$SLEEP_TIME"
    fi
done

