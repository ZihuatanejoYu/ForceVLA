#!/usr/bin/bash

source /home/hairuo/VLA/.venv/bin/activate

export CUDA_VISIBLE_DEVICES=6

uv run scripts/compute_norm_stats.py --config-name pi0_fast_flexiv_noforce_lora