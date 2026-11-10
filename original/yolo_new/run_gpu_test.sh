#!/bin/bash

prepare_directory() {
    if [ ! -d "$1" ]; then
        mkdir "$1"
    else
        rm -f "$1"/*.txt
    fi
}

prepare_directory "run_gpu_results"
prepare_directory "results"
prepare_directory "run_mem_results"

convert_to_mb() {
    local value=$1
    if [[ $value == *g ]]; then
        echo $(awk "BEGIN {print ${value%g} * 1024}")
    elif [[ $value == *m ]]; then
        echo ${value%m}
    elif [[ $value == *k ]]; then
        echo $(awk "BEGIN {print ${value%k} / 1024}")
    else
        echo $(awk "BEGIN {print $value / 1024 / 1024}")
    fi
}

run_with_tegrastats() {
    local script_name=$1
    local result_file="results/$2"
    local gpu_usage_file="run_gpu_results/${script_name%.py}_gpu_usage.txt"
    local mem_file="run_mem_results/${script_name%.py}_mem.txt"
    local max_mem_file="run_mem_results/${script_name%.py}_max_mem.txt"

    > "$gpu_usage_file"
    > "$mem_file"
    > "$max_mem_file"

    python3 $script_name > $result_file &
    SCRIPT_PID=$!

    max_mem=0

    monitor_gpu() {
        > "$gpu_usage_file"
        tegrastats --interval 1000 > temp_tegrastats.log &
        TEGRA_PID=$!

        while kill -0 $SCRIPT_PID 2>/dev/null; do
            gpu_usage=$(tail -n 1 temp_tegrastats.log | grep -oP "GR3D_FREQ \K[0-9]+%")
            if [ -n "$gpu_usage" ]; then
                echo "$gpu_usage" >> "$gpu_usage_file"
            fi
            sleep 1
        done
        kill $TEGRA_PID
        rm -f temp_tegrastats.log
    }

    monitor_memory() {
        > "$mem_file"
        while kill -0 $SCRIPT_PID 2>/dev/null; do
            if [ -f "/proc/$SCRIPT_PID/status" ]; then
                current_mem=$(awk '/VmRSS/ {print $2}' /proc/$SCRIPT_PID/status)
                current_mem_mb=$(convert_to_mb "${current_mem}k")
                current_mem_mb=${current_mem_mb:-0}
                echo "$current_mem_mb" >> "$mem_file"
                if (( $(echo "$current_mem_mb > $max_mem" | bc -l) )); then
                    max_mem=$current_mem_mb
                    echo "$max_mem" > "$max_mem_file"
                fi
            fi
            sleep 1
        done
    }

    monitor_gpu &
    GPU_MONITOR_PID=$!
    monitor_memory &
    MEMORY_MONITOR_PID=$!

    wait $SCRIPT_PID
    kill $GPU_MONITOR_PID 2>/dev/null
    kill $MEMORY_MONITOR_PID 2>/dev/null

    echo "Final RES memory usage for $script_name:" >> "$max_mem_file"
}

trap "kill 0" EXIT

run_with_tegrastats "obb.py" "obb_result.txt"
run_with_tegrastats "pose.py" "pose_result.txt"
run_with_tegrastats "detection.py" "detect_result.txt"
run_with_tegrastats "classification.py" "classify_result.txt"
run_with_tegrastats "segment.py" "segment_result.txt"


