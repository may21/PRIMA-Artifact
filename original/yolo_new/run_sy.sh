#!/bin/bash

# Check and create necessary directories for GPU monitoring results
if [ ! -d "gpu_results" ]; then
    mkdir gpu_results
else
    rm -f gpu_results/*.txt
fi

# Function to monitor GPU usage using tegrastats
monitor_gpu() {
    local gpu_usage_file="gpu_results/gpu_usage.txt"
    echo "Monitoring GPU usage. Results will be saved in $gpu_usage_file"

    # Start monitoring GPU usage
    while true; do
        tegrastats --interval 1000 | while read -r line; do
            gpu_usage=$(echo "$line" | grep -oP "GR3D_FREQ \K[0-9]+%")
            gpu_clock=$(echo "$line" | grep -oP "GR3D \K[0-9]+@[0-9]+MHz" | grep -oP "@\K[0-9]+")
	    timestamp=$(date +"%Y-%m-%d %H:%M:%S")
            if [ -n "$gpu_usage" ]; then
                echo "$timestamp GPU Usage: $gpu_usage GPU clock speed :${gpu_clock}MHz" >> "$gpu_usage_file"
            fi
        done
    done
}

# Run the GPU monitoring function
monitor_gpu

