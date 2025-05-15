source ../VLA/.venv/bin/activate

export CUDA_VISIBLE_DEVICES=5

uv run scripts/eval_open_loop_router_load.py \
    --repo-id flexiv/flexiv_1plug_insert_inputForce \
    --episode-index 0 \
    --prompt "Insert the power plug into the socket." \
    --policy-host 127.0.0.1 --policy-port 8000 \
    --output-dir results/