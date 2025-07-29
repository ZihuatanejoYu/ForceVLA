#!/bin/bash

# Array of train names
trains=("insert_USB" "plug_insert" "peel_cucumber")  # 注意: 我将"plug insert"改为"plug_insert"，因为空格在路径中可能有问题

# Array of GPU IDs to use
gpus=(3 4 5)  # 假设你有3个GPU可用

# Check if we have enough GPUs
if [ ${#trains[@]} -gt ${#gpus[@]} ]; then
    echo "Error: Not enough GPUs for all tasks (${#trains[@]} tasks but only ${#gpus[@]} GPUs)"
    exit 1
fi

# Loop through each train with index
for i in "${!trains[@]}"; do
    train="${trains[i]}"
    gpu="${gpus[i]}"
    
    echo "Processing $train on GPU $gpu..."
    CUDA_VISIBLE_DEVICES=$gpu python /home/hairuo/flexiv_pi0/examples/port_datasets/convert_to_video_mode.py \
        --raw_dir "/data/hairuo/forcevla_train_raw/flexiv_${train}_raw/" \
        --repo_id "flexiv_train/flexiv_${train}_inputForce" &
done

# Wait for all background processes to finish
wait
echo "All tasks completed."