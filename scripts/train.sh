export WANDB_API_KEY=3c7198950e2447940a415a6284d547f702387ea7

source ./.venv/bin/activate

export NCCL_P2P_DISABLE=1

export CUDA_VISIBLE_DEVICES=7

XLA_PYTHON_CLIENT_MEM_FRACTION=0.98 uv run scripts/train.py pi0_fast_flexiv_noforce_lora --exp-name=pi0_fast_flexiv_noforce_lora_flexiv_insert_USB_inputForce_noforce_0502 --overwrite