#!/bin/bash

# Array of test names
tests=("insert_plug" "insert_usb" "peel_cucumber" "pump_bottle")

# Loop through each test
for test in "${tests[@]}"; do
    echo "Processing $test..."
    python flexiv_pi0/examples/port_datasets/convert_to_video_mode.py \
        --raw_dir "/data/hairuo/forcevla_test_raw/flexiv_test_${test}_raw/" \
        --repo_id "flexiv_test/flexiv_${test}_inputForce"
done