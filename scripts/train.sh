export WANDB_API_KEY=3c7198950e2447940a415a6284d547f702387ea7

source ./.venv/bin/activate

export NCCL_P2P_DISABLE=1

export CUDA_VISIBLE_DEVICES=6

XLA_PYTHON_CLIENT_MEM_FRACTION=0.99 uv run scripts/train.py pi0_fast_flexiv_input_force_lora --exp-name=pi0_fast_flexiv_input_force_lora_flexiv_wipe_board_inputForce_input_force_0503 --overwrite