#!/bin/bash

OUTPUT_FILE="obb_pidstat.log"

while true; do
    TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
    TARGET_PIDS=$(pgrep -f "obb.py")

    if [ -z "$TARGET_PIDS" ]; then
        echo "$TIMESTAMP - No process found for obb.py" >> "$OUTPUT_FILE"
    else
        for PID in $TARGET_PIDS; do
            pidstat -u -p "$PID" 1 1 | awk -v timestamp="$TIMESTAMP" -v pid="$PID" '
                NR>3 && $1 != "Average:" {print timestamp ", PID=" pid ", CPU=" $8}' >> "$OUTPUT_FILE"
        done
    fi
    sleep 1
done

