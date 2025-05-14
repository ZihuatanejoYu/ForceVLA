source ./.venv/bin/activate

export NCCL_P2P_DISABLE=0

export CUDA_VISIBLE_DEVICES=0,1

XLA_PYTHON_CLIENT_MEM_FRACTION=0.99 uv run scripts/train.py pi0_guidance_lora --exp-name=pi0_guidance_lora_multitask_0514 --overwrite