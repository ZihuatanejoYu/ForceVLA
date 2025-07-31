#!/bin/bash

# 定义输入输出目录对
declare -A tasks=(
  ["flexiv_insert_USB_raw"]="insert_USB_w_force.zarr"
  ["flexiv_peel_cucumber_raw"]="peel_cucumber_w_force.zarr"
  ["flexiv_plug_insert_raw"]="plug_insert_w_force.zarr"
  ["flexiv_pump_bottle_raw"]="pump_bottle_w_force.zarr"
  ["flexiv_wipe_board_raw"]="wipe_board_w_force.zarr"
)

input_base="/data/hairuo/forcevla_train_raw"
output_base="$HOME/data/flexiv_w_force"    

for dir in "${!tasks[@]}"; do
  input_dir="${input_base}/${dir}"
  output_path="${output_base}/${tasks[$dir]}"
  echo "正在处理 $input_dir -> $output_path"
  python examples/port_datasets/npz2zarr.py "$input_dir" "$output_path"
done