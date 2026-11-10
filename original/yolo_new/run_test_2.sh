#!/bin/bash

# Check and create necessary directories if they don't exist
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

# Function to run a script and monitor CPU usage with pidstat, mpstat and memory usage with top
run_with_pidstat() {
    local script_name=$1
    local result_file="results/$2"
    local pidstat_file="pid_results/${script_name%.py}_pidstat.txt"
    local mpstat_file="mp_results/${script_name%.py}_mpstat.txt"
    local mem_file="mem_results/${script_name%.py}_mem.txt"

    # Start the python script and get its PID
    python3 $script_name > $result_file &
    SCRIPT_PID=$!

    # Monitor CPU usage of the script with pidstat including -I and -t options
    pidstat -h -r -u -I -t -p $SCRIPT_PID 2 > $pidstat_file &
    PIDSTAT_PID=$!

    # Monitor overall CPU usage with mpstat
    mpstat -P ALL 2 > $mpstat_file &
    MPSTAT_PID=$!

    # Initialize memory usage
    echo "Monitoring memory usage for $script_name" > $mem_file

    # Function to monitor memory usage using top
    monitor_memory() {
        while kill -0 $SCRIPT_PID 2> /dev/null; do
            top_output=$(top -b -n 1 -p $SCRIPT_PID | awk '/^ *'$SCRIPT_PID'/')
            virt_usage_kb=$(echo $top_output | awk '{print $5}' | sed 's/[a-zA-Z]//g')
            res_usage_kb=$(echo $top_output | awk '{print $6}' | sed 's/[a-zA-Z]//g')
 
 	    shr_usage_kb=$(echo $top_output | awk '{print $7}' | sed 's/[a-zA-Z]//g')
            total_usage_kb=$(echo "$virt_usage_kb + $res_usage_kb + $shr_usage_kb" | bc)
            virt_usage_gb=$(echo "scale=4; $virt_usage_kb/1024/1024" | bc)
            res_usage_gb=$(echo "scale=4; $res_usage_kb/1024/1024" | bc)
            shr_usage_gb=$(echo "scale=4; $shr_usage_kb/1024/1024" | bc)
            total_usage_gb=$(echo "scale=4; $total_usage_kb/1024/1024" | bc)
            echo "VIRT: ${virt_usage_gb}GB, RES: ${res_usage_gb}GB, SHR: ${shr_usage_gb}GB, TOTAL: ${total_usage_gb}GB" >> $mem_file
            sleep 1
        done
    }

    # Run the monitor_memory function in the background
    monitor_memory &

    # Wait for the python script to finish
    wait $SCRIPT_PID

    # Stop pidstat and mpstat monitoring
    kill $PIDSTAT_PID
    kill $MPSTAT_PID
}

# Run each script with pidstat, mpstat, and memory usage monitoring
#run_with_pidstat "obb.py" "obb_result.txt"
#run_with_pidstat "pose.py" "pose_result.txt"
#run_with_pidstat "detection.py" "detect_result.txt"
#run_with_pidstat "classification.py" "classify_result.txt"
#run_with_pidstat "segment.py" "segment_result.txt"
run_with_pidstat "GIL_4.py" "GIL_4_result.txt"
#run_with_pidstat "multiprocess.py" "multprocess_result.txt"


