#!/bin/bash

OUTPUT_DIR="gpu_five_result"
GPU_FILE="${OUTPUT_DIR}/gpu_usage.txt"
MEMORY_FILE="${OUTPUT_DIR}/memory_usage.txt"

if [ -e "$OUTPUT_DIR" ]; then
    if [ -f "$OUTPUT_DIR" ]; then
        rm "$OUTPUT_DIR"
    fi
fi

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir "$OUTPUT_DIR"
fi

if [ -f "$GPU_FILE" ]; then
    rm "$GPU_FILE"
fi

if [ -f "$MEMORY_FILE" ]; then
    rm "$MEMORY_FILE"
fi

TARGET_PROCESS="python3"

tegrastats --interval 1000 > tegrastats_output.log &
TEGRA_PID=$!

get_res_memory_per_pid() {
    pgrep -x "$TARGET_PROCESS" | while read -r PID; do
        RES=$(ps -o rss= -p "$PID")
        CMD=$(ps -o args= -p "$PID" | awk '{print $2}')
        if [ -n "$RES" ]; then
            RES_MB=$(awk "BEGIN {print $RES / 1024}")
            echo "PID=$PID, RES_MEMORY=${RES_MB}MB, FILE=$CMD"
        else
            echo "PID=$PID, RES_MEMORY=0MB, FILE=Unknown"
        fi
    done
}

INTERVAL=1

while true; do
    START_TIME=$(date +%s.%N)

    if ! pgrep -x "$TARGET_PROCESS" > /dev/null; then
        echo "Target process '$TARGET_PROCESS' not running. Exiting script."
        break
    fi

    GPU_USAGE=$(tail -n 1 tegrastats_output.log | grep -oP "GR3D_FREQ \K[0-9]+")
    GPU_USAGE=${GPU_USAGE:-0}

    RES_MEMORY_OUTPUT=$(get_res_memory_per_pid)

    TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
    echo "$TIMESTAMP, GPU_USAGE=$GPU_USAGE%" >> "$GPU_FILE"
    echo "$TIMESTAMP" >> "$MEMORY_FILE"
    echo "$RES_MEMORY_OUTPUT" >> "$MEMORY_FILE"

    END_TIME=$(date +%s.%N)
    ELAPSED_TIME=$(awk "BEGIN {print $END_TIME - $START_TIME}")
    SLEEP_TIME=$(awk "BEGIN {print $INTERVAL - $ELAPSED_TIME}")
    if (( $(echo "$SLEEP_TIME > 0" | bc -l) )); then
        sleep "$SLEEP_TIME"
    fi
done

kill $TEGRA_PID

