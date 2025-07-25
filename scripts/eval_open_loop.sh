source ../VLA/.venv/bin/activate
export CUDA_VISIBLE_DEVICES=7

declare -A tasks=(
    ["flexiv_test/flexiv_insert_plug_inputForce"]="Insert the power plug into the socket."
    ["flexiv_test/flexiv_insert_usb_inputForce"]="Pick up the USB and plug it into the socket."
    # ["flexiv_test/flexiv_peel_cucumber_inputForce"]="Peel cucumber skin."
    ["flexiv_test/flexiv_pump_bottle_inputForce"]="Press the pump dispenser on the bottle all the way down."
    ["flexiv_test/flexiv_wipe_board_inputForce"]="Pick up the eraser and clean off the writing on the whiteboard."
)

BATCH_SIZE=10
    
for repo_id in "${!tasks[@]}"; do
    for ((run=0; run<BATCH_SIZE; run++)); do
        prompt="${tasks[$repo_id]}"
        echo "Evaluating $repo_id (Run ${run}) with prompt: $prompt"
        
        uv run scripts/eval_open_loop_mse.py \
            --repo-id "$repo_id" \
            --episode-index "${run}" \
            --prompt "$prompt" \
            --policy-host 127.0.0.1 \
            --policy-port 8000
        mv ./router_load/gating_probs.pt ./router_load/gating_probs_${repo_id//\//_}_run${run}.pt

        sleep 2
    done
done
