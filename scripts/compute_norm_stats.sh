#!/usr/bin/bash

source ./.venv/bin/activate

export NCCL_P2P_DISABLE=1

export CUDA_VISIBLE_DEVICES=7

uv run scripts/compute_norm_stats.py --config-name pi0_fast_flexiv_noforce_lora