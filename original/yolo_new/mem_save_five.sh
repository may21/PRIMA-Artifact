#!/bin/bash

OUTPUT_DIR="mem_five_result"
MEMORY_LOG_FILE="${OUTPUT_DIR}/memory_res.txt"

if [ -d "$OUTPUT_DIR" ]; then
    rm -rf "$OUTPUT_DIR"
fi
mkdir "$OUTPUT_DIR"

TARGET_PROCESS="python3"
INTERVAL=1

while true; do
    TARGET_PIDS=$(pgrep -x "$TARGET_PROCESS")
    if [ -z "$TARGET_PIDS" ]; then
        echo "No process found for $TARGET_PROCESS. Exiting."
        break
    fi

    TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

    TOP_OUTPUT=$(top -b -n 1 | grep -E "$(echo $TARGET_PIDS | sed 's/ /|/g')")

    for PID in $TARGET_PIDS; do
        CMDLINE=$(tr '\0' ' ' < /proc/$PID/cmdline | awk '{print $2}')
        if [ -z "$CMDLINE" ]; then
            CMDLINE="UNKNOWN_SCRIPT"
        fi

        MEMORY_OUTPUT=$(echo "$TOP_OUTPUT" | grep -w "$PID")
        if [ -n "$MEMORY_OUTPUT" ]; then
            RES_MEMORY=$(echo "$MEMORY_OUTPUT" | awk '{print $6}')
            
            if [[ "$RES_MEMORY" == *g ]]; then
                VALUE=${RES_MEMORY%g}
                RES_MEMORY_MB=$(awk "BEGIN {printf \"%.2f\", $VALUE * 1024}")
            else
                VALUE=$RES_MEMORY
                RES_MEMORY_MB=$(awk "BEGIN {printf \"%.2f\", $VALUE / 1024}")
            fi

            echo "$TIMESTAMP, PID=$PID, SCRIPT=$CMDLINE, RES_MEMORY=${RES_MEMORY_MB}MB" >> "$MEMORY_LOG_FILE"
        else
            echo "$TIMESTAMP, PID=$PID, SCRIPT=$CMDLINE, ERROR: Memory usage not found." >> "$MEMORY_LOG_FILE"
        fi
    done

    sleep $INTERVAL
done

