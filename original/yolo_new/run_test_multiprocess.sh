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

if [ ! -d "gpu_results" ]; then
    mkdir gpu_results
else
    rm -f gpu_results/*.txt
fi

#timestamp=$(date +"%H:%M:%S")

# Function to monitor CPU usage of a process and its child processes
monitor_cpu_usage() {
    local parent_pid=$1
    local pidstat_file=$2
    
    while kill -0 $parent_pid 2> /dev/null; do
        # Get child process IDs of the parent
        child_pids=$(pgrep -P $parent_pid | tr '\n' ',')
        
        #if [ ! -z "$child_pids" ]; then
            # Monitor CPU usage for parent and child processes, displaying only PID and %CPU
            #pidstat -u -p "${parent_pid},${child_pids%,}" 2 1 | awk '/^[0-9]/ {print "PID:", $3, "CPU Usage:", $8"%"}' >> $pidstat_file
	    #pidstat -I -p "${parent_pid},${child_pids%,}" 1 5  | awk '!/Linux/' >> $pidstat_file  # | awk '/^[0-9]/ {print "PID:", $3, "CPU Usage:", $8"%"}'
        #else
            # If no child processes, monitor CPU usage for the parent only
            #pidstat -I -p $parent_pid 1 5 | awk '!/Linux/' >> $pidstat_file 
	#fi
    done
}

# Function to monitor memory usage of a process and its child processes
monitor_memory_usage() {
    local parent_pid=$1
    local mem_file=$2

    while kill -0 $parent_pid 2> /dev/null; do
        timestamp=$(date +"%H:%M:%S")

	# Get parent and child process IDs
        child_pids=$(pgrep -P $parent_pid)
        all_pids="$parent_pid $child_pids"

        echo "Parent PID: $parent_pid" >> $mem_file

	total_res_usage_gb=0
        # For each process, get memory usage
        for pid in $all_pids; do
            top_output=$(top -b -n 1 -p $pid | awk '/^ *'$pid'/')
            #virt_usage_kb=$(echo $top_output | awk '{print $5}' | sed 's/[a-zA-Z]//g')
	    res_usage_kb=$(echo $top_output | awk '{print $6}' ) # | sed 's/[a-zA-Z]//g')
            #shr_usage_kb=$(echo $top_output | awk '{print $7}' | sed 's/[a-zA-Z]//g')

            # Calculate memory usage GB if it is KB
	    if [[ $res_usage_kb =~ ^[0-9]+$ ]]; then
            	#virt_usage_gb=$(echo "scale=4; $virt_usage_kb/1024/1024" | bc)
            	res_usage_gb=$(echo "scale=4; $res_usage_kb/1024/1024" | bc)
           	#shr_usage_gb=$(echo "scale=4; $shr_usage_kb/1024/1024" | bc)
	    else
        	res_usage_gb=$(echo $res_usage_kb | sed 's/[a-zA-Z]//g')
	    fi

	    total_res_usage_gb=$(echo "scale=4; $total_res_usage_gb + $res_usage_gb" | bc)

            # Log memory usage per process
            #echo "$timestamp PID: $pid, VIRT: ${virt_usage_gb}GB, RES: ${res_usage_gb}GB, SHR: ${shr_usage_gb}GB" >> $mem_file
	    echo "$timestamp $pid RES(GB): ${res_usage_gb}" >> $mem_file

    	done

	echo "$timestamp Total RES(GB): ${total_res_usage_gb}" >> $mem_file	
	
	#total used memory
	#vmstat 1 1 | awk -v timestamp="$timestamp" 'NR==3 {print timestamp " Used Memory(GB):", 61.3 - $4/1048576 - $5/1048576 - $6/1048576}' >> $mpstat_file

        sleep 1
    done
}


monitor_gpu() {
        #while kill -0 $parent_pid 2> /dev/null; do
            tegrastats --interval 1000 | while read -r line; do
            gpu_usage=$(echo "$line" | grep -oP "GR3D_FREQ \K[0-9]+%")
            timestamp=$(date +"%H:%M:%S")
            echo "$timestamp $gpu_usage"  >>  $gpu_usage_file	
	    done
	    
	    pkill -f "tegrastats"

            #gpu_usage=$(tegrastats | head -n 1 | grep -oP "GR3D_FREQ \K[0-9]+%")
            #echo "$timestamp $gpu_usage" # >> gpu_usage_file
        #done
}


# Function to run a script and monitor CPU usage
run_with_pidstat() {
    local script_name=$1
    local pidstat_file="pid_results/${script_name%.py}_pidstat.txt"
    local result_file="results/${script_name%.py}_result.txt"
    local times_file="results/${script_name%.py}_times.txt"
    local mem_file="mem_results/${script_name%.py}_mem.txt"
    local mpstat_file="mp_results/${script_name%.py}_mpstat.txt"
    local gpu_usage_file="gpu_results/${script_name%.py}_gpu_usage.txt"
    local cpu_summary_file="pid_results/${script_name%.py}_cpu_summary.txt"
    
    # Start the python script and get its PID
    python3 $script_name > $result_file &
    SCRIPT_PID=$!

    # Monitor CPU and memory usage
    monitor_cpu_usage $SCRIPT_PID $pidstat_file &
    PIDSTAT_PID=$!
    monitor_memory_usage $SCRIPT_PID $mem_file &
    MEM_PID=$!
    monitor_gpu $SCRIPT_PID $gpu_usage_file &
    #TEGRASTATS_PID=$!
	

    # Wait for the python script to finish
    wait $SCRIPT_PID

    # Stop monitoring
    pkill -P $$
    pkill -f "tegrastats"

    # Extract preprocess, inference, and postprocess times
    grep "Speed:" $result_file | awk '{print NR , "Preprocess: "$2, "Inference: "$4, "Postprocess: "$6}' > $times_file

    grep "Total" $mem_file > mem_results/multiprocess_mem_total.txt

    # Extract lines starting with "Average:" from pidstat output to a single file
    awk '/^Average:/' $pidstat_file > pid_results/${script_name%.py}_average.txt

    # Parse `pidstat_file` to calculate total CPU usage per second   
    awk '/^[0-9]/ {timestamps[$1] += $9} END {for (time in timestamps) print time, "Total CPU Usage(%):", timestamps[time] }' $pidstat_file | sort > $cpu_summary_file

}

# Run each script with CPU usage monitoring
run_with_pidstat "multiprocess.py"

