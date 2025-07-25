source ../VLA/.venv/bin/activate

export CUDA_VISIBLE_DEVICES=6

uv run scripts/serve_policy.py --port 8000 policy:checkpoint --policy.config=pi0_guidance_lora --policy.dir=/home/hairuo/flexiv_pi0/checkpoints/pi0_guidance_lora/pi0_guidance_lora_multitask_0514/29999