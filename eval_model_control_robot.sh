# python lerobot/scripts/control_flexiv.py \
#     --robot.type=aloha \
#     --control.type=record \
#     --control.fps=30 \
#     --control.single_task="pick up the box and put it in the basket" \
#     --control.repo_id=Predentsi/flexiv_box_boxpp_scenechange \
#     --control.root=/hdd1/flexiv_teleop_dataset/Predentsi/flexiv_box_boxpp_scenechange \
#     --control.num_episodes=10 \
#     --control.warmup_time_s=2 \
#     --control.episode_time_s=30 \
#     --control.reset_time_s=10 \
#     --control.push_to_hub=false \
#     --control.policy.path=/hdd1/flexiv_act_ckpts/act_flexiv_box_boxpp_scenechange_0403/checkpoints/020000/pretrained_model

python lerobot/scripts/control_flexiv.py \
    --robot.type=aloha \
    --control.type=record \
    --control.fps=30 \
    --control.single_task="pick up the box and put it in the basket" \
    --control.repo_id=Predentsi/flexiv_box_boxpp_scenechange \
    --control.root=/hdd1/flexiv_teleop_dataset/Predentsi/flexiv_box_boxpp_scenechange \
    --control.num_episodes=10 \
    --control.warmup_time_s=2 \
    --control.episode_time_s=30 \
    --control.reset_time_s=10 \
    --control.push_to_hub=false \
    --control.policy.path=/hdd1/flexiv_dp_ckpts/dp_0403/checkpoints/040000/pretrained_model