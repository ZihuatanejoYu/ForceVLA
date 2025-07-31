#!/bin/bash

# Array of train names
trains=("insert_USB" "plug_insert" "peel_cucumber" "wipe_board" "peel_cucumber")

# Array of GPU IDs to use
gpus=(0 1 2 3 4)  # 假设你有3个GPU可用

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
    # CUDA_VISIBLE_DEVICES=$gpu python /home/hairuo/flexiv_pi0/examples/port_datasets/convert_to_video_mode.py \
    #     --raw_dir "/data/hairuo/forcevla_train_raw/flexiv_${train}_raw/" \
    #     --repo_id "flexiv_train/flexiv_${train}_inputForce" &
    CUDA_VISIBLE_DEVICES=$gpu python /home/hairuo/flexiv_pi0/examples/port_datasets/npz2zarr.py \
        --input_dir "/data/hairuo/forcevla_train_raw/flexiv_${train}_raw/" \
        --output_dir "/home/hairuo/data/forcevla_train_zarr/flexiv_${train}_zarr/" &
done

# Wait for all background processes to finish
wait
echo "All tasks completed."