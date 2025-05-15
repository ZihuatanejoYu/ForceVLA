source ./.venv/bin/activate

export CUDA_VISIBLE_DEVICES=2

XLA_PYTHON_CLIENT_MEM_FRACTION=0.98 uv run scripts/train.py pi0_guidance_lora --exp-name=pi0_guidance_router_vis_0515 --overwrite