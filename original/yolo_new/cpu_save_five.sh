#!/bin/bash

OUTPUT_DIR="cpu_five_result"
if [ -d "$OUTPUT_DIR" ]; then
    rm -rf "$OUTPUT_DIR"
fi
mkdir "$OUTPUT_DIR"

TARGET_PROCESS="python3"

while true; do
    # 현재 실행 중인 모든 Python 프로세스 ID 가져오기
    TARGET_PIDS=$(pgrep -x "$TARGET_PROCESS")
    if [ -z "$TARGET_PIDS" ]; then
        echo "No process found for $TARGET_PROCESS. Exiting."
        break
    fi

    for PID in $TARGET_PIDS; do
        # 실행 중인 Python 스크립트 파일명 추출
        CMDLINE=$(tr '\0' ' ' < /proc/$PID/cmdline | awk '{for (i=2; i<=NF; i++) printf $i " "; print ""}')
        if [ -z "$CMDLINE" ]; then
            CMDLINE="UNKNOWN_SCRIPT"
        fi

        LOG_FILE="$OUTPUT_DIR/${CMDLINE}_${PID}_pidstat.log"

        # 이미 추적 중인 프로세스는 건너뜀
        if [ -f "$LOG_FILE" ]; then
            continue
        fi

        echo "Monitoring PID=$PID, Command=$CMDLINE" > "$LOG_FILE"
        echo "Time         UID       PID    %usr %system  %guest   %wait    %CPU   CPU  Command" >> "$LOG_FILE"

        {
            while ps -p "$PID" > /dev/null; do
                TIMESTAMP=$(date +"%H:%M:%S")
                # 1초 간격으로 pidstat 실행
                pidstat -u -p "$PID" 1 1 | awk -v timestamp="$TIMESTAMP" -v pid="$PID" -v cmdline="$CMDLINE" '
                    NR > 3 && $1 != "Average:" {
                        print timestamp, $2, $3, $4, $5, $6, $7, $8, $9
                    }' >> "$LOG_FILE"
            done
            echo "$(date +"%H:%M:%S") PID=$PID SCRIPT=$CMDLINE Status=Terminated" >> "$LOG_FILE"
        } &
    done

    sleep 1
done

