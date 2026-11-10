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
    echo "Monitoring GPU usage and clock speeds. Results will be saved in $gpu_usage_file"

    # Start monitoring GPU usage and clock speeds
    while true; do
        sudo tegrastats --interval 1000 | while read -r line; do
            timestamp=$(date +"%Y-%m-%d %H:%M:%S")

            # Extract GPU utilization and clock speed
            gpu_usage=$(echo "$line" | grep -oP "GR3D_FREQ \K[0-9]+%")
            gpu_clock=$(echo "$line" | grep -oP "GR3D_FREQ .*?\@\[.*?]" | grep -oP "\[\K[0-9,]+(?=\])")

            # Extract CPU clock speeds
            #cpu_clocks=$(echo "$line" | grep -oP "CPU \[.*?\]" | sed 's/CPU \[//;s/\]//')

            # Extract RAM usage
            #ram_usage=$(echo "$line" | grep -oP "RAM \K[0-9]+/[0-9]+MB")

            # Extract other relevant data (if needed)
            #emc_freq=$(echo "$line" | grep -oP "EMC_FREQ \K[0-9]+%@\d+")
            #gpu_temp=$(echo "$line" | grep -oP "GPU@\K[0-9.]+C")

            # Combine and save the extracted information
            echo "$timestamp GPU Usage: $gpu_usage GPU Clock: $gpu_clock " >> "$gpu_usage_file"
        done
    done
}

# Run the GPU monitoring function
monitor_gpu

