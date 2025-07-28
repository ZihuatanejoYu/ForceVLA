# cd /home/qiaojun/flexiv_manipulation/example_py/flexiv_pi0-dev
source .venv/bin/activate

# export HF_HUB_OFFLINE=1
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_guidance_lora \
    --policy.dir=/data/hairuo/checkpoints/pi0/flexiv/pi0_guidance_lora/pi0_guidance_rebuttle_force_ablation_arch_insert_plug_0725/10000