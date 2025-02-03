#!/bin/bash

# Function to write zero to all peak files
write_zero_to_peaks() {
    local base_path="/sys/devices/virtual/neuron_device/neuron0/neuron_core0/stats/memory_usage/device_mem"
    local categories=("collectives" "constants" "dma_rings" "driver_memory" "model_code" "model_shared_scratchpad" "nonshared_scratchpad" "notifications" "runtime_memory" "tensors" "uncategorized")

    for category in "${categories[@]}"; do
        sudo bash -c "echo 0 > ${base_path}/${category}/peak"
    done
    sudo bash -c "echo 0 > ${base_path}/peak"
}

# Function to read and print summary of peak memory
read_peak_memory() {
    local base_path="/sys/devices/virtual/neuron_device/neuron0/neuron_core0/stats/memory_usage/device_mem"
    local categories=("collectives" "constants" "dma_rings" "driver_memory" "model_code" "model_shared_scratchpad" "nonshared_scratchpad" "notifications" "runtime_memory" "tensors" "uncategorized")

    # Function to convert bytes to GB with 3 decimal places
    bytes_to_gb() {
        echo "scale=3; $1 / 1024 / 1024 / 1024" | bc
    }

    echo "Peak Memory Summary:"
    for category in "${categories[@]}"; do
        local peak=$(sudo cat "${base_path}/${category}/peak")
        local peak_gb=$(bytes_to_gb $peak)
        printf "%s: %s GB\n" "${category}" "${peak_gb}"
    done

    local total_peak=$(sudo cat "${base_path}/peak")
    local total_peak_gb=$(bytes_to_gb $total_peak)
    printf "Total Peak: %s GB\n" "${total_peak_gb}"
}
