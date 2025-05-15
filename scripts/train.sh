# export WANDB_API_KEY=3c7198950e2447940a415a6284d547f702387ea7

source ./.venv/bin/activate

export NCCL_P2P_DISABLE=1

export CUDA_VISIBLE_DEVICES=0

XLA_PYTHON_CLIENT_MEM_FRACTION=0.98 uv run scripts/train.py pi0_guidance_lora --exp-name=pi0_guidance_0513 --overwrite