#!/bin/bash

if [ ! -d "results" ]; then
    mkdir results
else
    rm -f results/*.txt
fi

run_with_pidstat() {
    local script_name=$1
    local result_dir="results/${script_name%.py}"
    local result_file="${result_dir}/${script_name%.py}_result.txt"
    local pidstat_file="${result_dir}/${script_name%.py}_pidstat.txt"
    local mpstat_file="${result_dir}/${script_name%.py}_mpstat.txt"
    local mem_file="${result_dir}/${script_name%.py}_mem.txt"
    local max_mem_file="${result_dir}/${script_name%.py}_max_mem.txt"

    if [ ! -d "$result_dir" ]; then
        mkdir -p "$result_dir"
    else
        rm -f "$result_dir"/*.txt
    fi

    python3 $script_name > $result_file &
    SCRIPT_PID=$!

    pidstat -h -r -u -I -t -p $SCRIPT_PID 2 > $pidstat_file &
    PIDSTAT_PID=$!

    mpstat -P ALL 2 > $mpstat_file &
    MPSTAT_PID=$!

    echo "Monitoring memory usage for $script_name" > $max_mem_file
    echo "Monitoring memory usage for $script_name" > $mem_file
    max_mem=0

    monitor_memory() {
        while kill -0 $SCRIPT_PID 2> /dev/null; do
            current_mem=$(top -b -n 1 -p $SCRIPT_PID | awk '/'"$SCRIPT_PID"'/{print $6}' | sed 's/K//')
            echo "Current RES memory usage: ${current_mem}K" >> $mem_file
            free -h >> $mem_file

            if [ "$current_mem" -gt "$max_mem" ]; then
                max_mem=$current_mem
                echo "Current max RES memory usage: ${max_mem}K" > $max_mem_file
                free -h >> $max_mem_file
            fi
            sleep 1
        done
    }

    monitor_memory &

    wait $SCRIPT_PID

    kill $PIDSTAT_PID
    kill $MPSTAT_PID

    echo "Final RES memory usage for $script_name:" >> $max_mem_file
    free -h >> $max_mem_file
}

scripts=("classification.py")

for script in "${scripts[@]}"; do
    run_with_pidstat "$script"
done

