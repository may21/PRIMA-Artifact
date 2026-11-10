#!/bin/bash

if [ ! -d "results" ]; then
    mkdir results
else
    rm -f results/*.txt
fi

run_with_tegrastats() {
    local script_name=$1
    local result_dir="results/${script_name%.py}"
    local result_file="${result_dir}/${script_name%.py}_result.txt"
    local gpu_usage_file="${result_dir}/${script_name%.py}_gpu_usage.txt"
    local mem_file="${result_dir}/${script_name%.py}_mem.txt"

    if [ ! -d "$result_dir" ]; then
        mkdir -p "$result_dir"
    else
        rm -f "$result_dir"/*.txt
    fi

    python3 $script_name > $result_file &
    SCRIPT_PID=$!

    while kill -0 $SCRIPT_PID 2>/dev/null; do
        # GPU 사용량 추출
        gpu_usage=$(tegrastats --interval 1000 | grep -m 1 -oP "GR3D_FREQ \K[0-9]+")
        gpu_usage=${gpu_usage:-0}
        echo "$gpu_usage%" >> $gpu_usage_file

        # 메모리 사용량 추출
        current_mem=$(top -b -n 1 -p $SCRIPT_PID | awk -v pid=$SCRIPT_PID '$1 == pid {print $6}')
        current_mem=${current_mem:-0}
        if [[ "$current_mem" == *g ]]; then
            current_mem=$(echo $current_mem | sed 's/g//')
            current_mem=$(echo "$current_mem * 1024" | bc)
        elif [[ "$current_mem" == *m ]]; then
            current_mem=$(echo $current_mem | sed 's/m//')
        else
            current_mem=$(echo "$current_mem / 1024" | bc)
        fi
        echo "$current_mem MB" >> $mem_file
        sleep 1
    done
}

scripts=("segment.py")

for script in "${scripts[@]}"; do
    run_with_tegrastats "$script"
done

