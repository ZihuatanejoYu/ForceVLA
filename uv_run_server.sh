cd /home/qiaojun/flexiv_manipulation/example_py/flexiv_pi0-dev
source .venv/bin/activate

export HF_HUB_OFFLINE=1
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_fast_flexiv_noforce_lora \
    --policy.dir=/data_1/flexiv_pi0_ckpts/pi0_fast_noforce/insert_plug/10000