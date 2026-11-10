#!/bin/bash

if [ ! -d "pid_results" ]; then
    mkdir pid_results
else
    rm -f pid_results/*.txt
fi

if [ ! -d "results" ]; then
    mkdir results
else
    rm -f results/*.txt
fi

if [ ! -d "mp_results" ]; then
    mkdir mp_results
else
    rm -f mp_results/*.txt
fi

if [ ! -d "mem_results" ]; then
    mkdir mem_results
else
    rm -f mem_results/*.txt
fi

run_with_pidstat() {
    local script_name=$1
    local result_file="results/$2"
    local pidstat_file="pid_results/${script_name%.py}_pidstat.txt"
    local mpstat_file="mp_results/${script_name%.py}_mpstat.txt"
    local mem_file="mem_results/${script_name%.py}_mem.txt"
    local max_mem_file="mem_results/${script_name%.py}_max_mem.txt"

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
            current_mem=$(top -b -n 1 -p $SCRIPT_PID | awk 'NR>7 {print $6}' | sed 's/K//')
            current_mem=${current_mem:-0} 
            echo "Current RES memory usage: ${current_mem}K" >> $mem_file

            if [ "$current_mem" -gt "$max_mem" ]; then
                max_mem=$current_mem
                echo "Current max RES memory usage: ${max_mem}K" > $max_mem_file
            fi
            sleep 1
        done
    }

    monitor_memory &

    wait $SCRIPT_PID

    kill $PIDSTAT_PID
    kill $MPSTAT_PID

    echo "Final RES memory usage for $script_name:" >> $max_mem_file
    top -b -n 1 -p $SCRIPT_PID >> $max_mem_file
}

run_with_pidstat "obb.py" "obb_result.txt"
run_with_pidstat "pose.py" "pose_result.txt"
run_with_pidstat "detection.py" "detect_result.txt"
run_with_pidstat "classification.py" "classify_result.txt"
run_with_pidstat "segment.py" "segment_result.txt"

