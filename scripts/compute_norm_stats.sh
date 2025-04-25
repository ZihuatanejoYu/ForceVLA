#!/usr/bin/bash

cd /home/qiaojun/flexiv_pi0-dev

source ./.venv/bin/activate

export NCCL_P2P_DISABLE=1

export CUDA_VISIBLE_DEVICES=0

uv run scripts/compute_norm_stats.py --config-name pi0_guidance_lora