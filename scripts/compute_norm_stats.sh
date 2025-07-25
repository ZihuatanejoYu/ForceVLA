#!/usr/bin/bash

source ./.venv/bin/activate

export CUDA_VISIBLE_DEVICES=1

uv run scripts/compute_norm_stats.py --config-name pi0_guidance_lora