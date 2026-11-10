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


# Function to run a script and monitor CPU usage with pidstat
run_with_pidstat() {
    local script_name=$1
    local result_file="results/$2"
    local pidstat_file="pid_results/${script_name%.py}_pidstat.txt"
    local mpstat_file="mp_results/${script_name%.py}_mpstat.txt"
    local times_file="results/${script_name%.py}_times.txt"
    local mem_file="mem_results/${script_name%.py}_mem.txt"
    local gpu_usage_file="gpu_results/${script_name%.py}_gpu_usage.txt"

    # Start the python script and get its PID
    #python3 $script_name > $result_file &
    #SCRIPT_PID=$!
    
    TARGET_PROCESS="python3"

    while true; do
    # 현재 실행 중인 모든 Python 프로세스 ID 가져오기
        SCRIPT_PID=$(pgrep -x "$TARGET_PROCESS")
    	if [ -z "$TARGET_PIDS" ]; then
        	echo "No process found for $TARGET_PROCESS. Exiting."
        	break
    	fi 

    # Initialize CPU usage log
    echo "Monitoring CPU usage with pidstat" > $pidstat_file

    # Function to monitor CPU usage using pidstat in a while loop
    monitor_cpu_usage() {
        while kill -0 $SCRIPT_PID 2> /dev/null; do
            pidstat -I -t -p $SCRIPT_PID 1 5  | awk '!/Linux/' >> $pidstat_file  #| awk '{print $9}'
	    #pidstat -I -t -p $SCRIPT_PID 1 5 | awk '/Average:/ {print $0}'
	    #sleep 0.1 
        done
    }

    # Run the monitor_cpu_usage function in the background
    monitor_cpu_usage &
    CPU_PID=$!
	
    # Function to monitor memory usage using top
    monitor_memory() {
        while kill -0 $SCRIPT_PID 2> /dev/null; do
	    timestamp=$(date +"%H:%M:%S")
	    top_output=$(top -b -n 1 -p $SCRIPT_PID | awk '/^ *'$SCRIPT_PID'/')
            #virt_usage_kb=$(echo $top_output | awk '{print $5}' | sed 's/[a-zA-Z]//g')
	    res_usage_kb=$(echo $top_output | awk '{print $6}' ) #| sed 's/[a-zA-Z]//g')
            #shr_usage_kb=$(echo $top_output | awk '{print $7}' | sed 's/[a-zA-Z]//g')
            #total_usage_kb=$(echo "$virt_usage_kb + $res_usage_kb + $shr_usage_kb" | bc)
           
	    total_res_usage_gb=0
	    # Calculate memory usage GB if it is KB
            if [[ $res_usage_kb =~ ^[0-9]+$ ]]; then
 	    	#virt_usage_gb=$(echo "scale=4; $virt_usage_kb/1024/1024" | bc)
            	res_usage_gb=$(echo "scale=4; $res_usage_kb/1024/1024" | bc)
            	#shr_usage_gb=$(echo "scale=4; $shr_usage_kb/1024/1024" | bc)
            	#total_usage_gb=$(echo "scale=4; $total_usage_kb/1024/1024" | bc)
            else
                res_usage_gb=$(echo $res_usage_kb | sed 's/[a-zA-Z]//g')
            fi
		
	    #echo "$timestamp VIRT: ${virt_usage_gb}GB, RES: ${res_usage_gb}GB, SHR: ${shr_usage_gb}GB, TOTAL: ${total_usage_gb}GB" >> $mem_file
	    echo "$timestamp  RES(GB): ${res_usage_gb} " >> $mem_file
	    #total_res_usage_gb=$(echo "scale=4; $total_res_usage_gb + $res_usage_gb" | bc)
    	
	    #total used memory
	    #vmstat 1 1 | awk -v timestamp="$timestamp" 'NR==3 {print timestamp " Used Memory(GB):", 61.3 - $4/1048576 - $5/1048576 - $6/1048576}' >> $mpstat_file

	    sleep 1
        done
    }    

    # Run the monitor_memory function in the background
    monitor_memory &


    monitor_gpu() {
        while kill -0 $SCRIPT_PID 2> /dev/null; do
            tegrastats --interval 1000 | while read -r line; do
            gpu_usage=$(echo "$line" | grep -oP "GR3D_FREQ \K[0-9]+%")
	    timestamp=$(date +"%H:%M:%S")
	    if [ -n "$gpu_usage" ]; then
	   	echo "$timestamp $gpu_usage" >> "$gpu_usage_file"
	    fi
	    done
	
	    #gpu_usage=$(tegrastats | head -n 1 | grep -oP "GR3D_FREQ \K[0-9]+%")
	    #echo "$timestamp $gpu_usage" # >> gpu_usage_file
	done
    }


    # Run the monitor_gpu_usage functuon in the background
    monitor_gpu &
    GPU_PID=$!


    # Wait for the python script to finish
    #wait $SCRIPT_PID

    # Kill background monitoring functions
    pkill -P $$
    pkill -P $$
    kill $CPU_PID
    kill $GPU_PID
    pkill -f "tegrastats"
    
    # Extract preprocess, inference, and postprocess times
    grep "Speed:" $result_file | awk '{print NR , "Preprocess: "$2, "Inference: "$4, "Postprocess: "$6}' > $times_file

    # Extract lines starting with "Average:" from pidstat output to a single file
    awk '/^Average:/' $pidstat_file > pid_results/${script_name%.py}_average.txt



# Run the function with the desired Python script
run_with_pidstat "GIL_4.py" "GIL_4_result.txt"

