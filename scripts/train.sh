source /home/hairuo/VLA/.venv/bin/activate

export CUDA_VISIBLE_DEVICES=6,7

XLA_PYTHON_CLIENT_MEM_FRACTION=0.99 uv run scripts/train.py pi0_flexiv_lora_eef_pos_eef_action --exp-name=pi0_flexiv_lora_eef_pos_eef_action_multi_task_0515 --overwrite