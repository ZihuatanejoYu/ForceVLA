source /home/hairuo/VLA/.venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1

XLA_PYTHON_CLIENT_MEM_FRACTION=0.99 uv run scripts/train.py pi0_fast_flexiv_input_force_lora --exp-name=pi0_fast_flexiv_input_force_lora_multi_task_0515 --overwrite