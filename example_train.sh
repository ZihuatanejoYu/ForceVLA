CUDA_VISIBLE_DEVICES=3 python lerobot/scripts/train.py \
  --dataset.repo_id=Predentsi/flexiv_box_pickplace \
  --dataset.root=/data/qiaojun/6023/vla/teleop_recordings/lerobot/Predentsi/flexiv_box_pickplace \
  --policy.type=act \
  --output_dir=/data/qiaojun/6023/vla_finetune/lerobot/outputs/train/act_flexiv_boxpp_test \
  --job_name=act_flexiv_boxpp_test \
  --policy.device=cuda \
  --wandb.enable=true