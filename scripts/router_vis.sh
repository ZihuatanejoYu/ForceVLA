source ../VLA/.venv/bin/activate

export CUDA_VISIBLE_DEVICES=5

uv run scripts/eval_open_loop_mse.py \
    --repo-id flexiv/flexiv_wipe_board_inputForce \
    --episode-index 0 \
    --prompt "Pick up the eraser and clean off the writing on the whiteboard." \
    --policy-host 127.0.0.1 --policy-port 8000 \
    --output-dir results/