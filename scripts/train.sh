# export WANDB_API_KEY=3c7198950e2447940a415a6284d547f702387ea7

source ./.venv/bin/activate

export NCCL_P2P_DISABLE=1

export CUDA_VISIBLE_DEVICES=0
# --exp-name=flexiv_1plug_insert_inputForce_50_0417
# pi0_guidance_debug_0422
XLA_PYTHON_CLIENT_MEM_FRACTION=0.75 uv run scripts/train.py pi0_guidance_lora --exp-name=pi0_guidance_debug_0426 --overwrite