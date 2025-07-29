import numpy as np
import zarr
import os
import json
import argparse
from pathlib import Path

# =================================================================================
# 配置区域: 请根据你的需求修改这里的映射关系
# =================================================================================

# 定义哪些 NPZ 中的键将被合并成 Zarr 中的 /state 数组
# 注意：这里的顺序非常重要，它将决定 state 向量的最终结构！
STATE_KEYS_TO_CONCATENATE = [
    'tcp_pose',
    'gripper_width'
]

# 定义 NPZ 中的图像键如何映射到 Zarr 中的相机名
# Zarr 结构: /<camera_name>/image
IMAGE_KEY_MAPPING = {
    'image': 'camera_front',       # NPZ的 'image' 键 -> Zarr的 'camera_front'
    'wrist_image': 'camera_wrist'  # NPZ的 'wrist_image' 键 -> Zarr的 'camera_wrist'
}

# =================================================================================

def convert_npz_to_zarr(input_dir: Path, output_dir: Path):
    """
    将包含多个npz文件的目录转换为一个Zarr数据集。
    """
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找所有npz文件
    npz_files = sorted(list(input_dir.glob('*/*.npz')))
    if not npz_files:
        print(f"错误: 在目录 {input_dir} 中未找到任何 .npz 文件。")
        return

    # 初始化元数据字典
    meta_data = {
        'episode_meta': {},
        'env_meta': {}
    }

    print(f"找到 {len(npz_files)} 个 episode 文件，开始转换...")

    for i, npz_path in enumerate(npz_files):
        episode_idx = i
        print(f"  - 正在处理 Episode {episode_idx}: {npz_path.name}")

        # 加载 NPZ 文件数据
        try:
            npz_data = np.load(npz_path)
        except Exception as e:
            print(f"    - 警告: 加载文件 {npz_path.name} 失败，已跳过。错误: {e}")
            continue

        # 创建 Zarr episode 存储
        # Zarr 会自动处理目录创建
        zarr_path = output_dir / f'episode_{episode_idx}.zarr'
        store = zarr.DirectoryStore(str(zarr_path))
        root = zarr.group(store=store, overwrite=True)

        # 1. 保存图像数据
        camera_names_in_episode = []
        for npz_key, zarr_camera_name in IMAGE_KEY_MAPPING.items():
            if npz_key in npz_data and npz_data[npz_key].shape[0] > 0:
                root.create_dataset(f'{zarr_camera_name}/image', data=npz_data[npz_key], chunks=(1, None, None, None), compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.BITSHUFFLE))
                camera_names_in_episode.append(zarr_camera_name)
        
        # 2. 保存 action 数据
        if 'action' in npz_data:
            root.create_dataset('action', data=npz_data['action'], chunks=(1, None), compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.BITSHUFFLE))
        else:
             print(f"    - 警告: 在 {npz_path.name} 中未找到 'action' 数据。")
             continue

        # 3. 合并并保存 state 数据
        state_arrays = []
        observation_dims = {}
        for key in STATE_KEYS_TO_CONCATENATE:
            if key in npz_data:
                # 确保数据是二维的 (Time, Dim)
                arr = npz_data[key]
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1) # 如果是1D，则扩展为 (N, 1)
                state_arrays.append(arr)
                observation_dims[key] = arr.shape[1]
            else:
                print(f"    - 警告: 在 {npz_path.name} 中未找到状态键 '{key}'。")
        
        if not state_arrays:
            print(f"    - 错误: 未能从 {npz_path.name} 提取任何状态数据，已跳过此 episode。")
            continue
            
        # 沿维度1（特征维度）拼接
        combined_state = np.concatenate(state_arrays, axis=1)
        root.create_dataset('state', data=combined_state, chunks=(1, None), compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.BITSHUFFLE))

        # 4. 填充元数据
        n_steps = combined_state.shape[0]
        meta_data['episode_meta'][f'episode_{episode_idx}'] = {
            'n_steps': n_steps,
            'camera_names': camera_names_in_episode,
            'instruction': str(npz_data.get('instruction', '')) # 添加指令
        }
        
        # 如果是第一个episode，填充env_meta
        if i == 0:
            meta_data['env_meta'] = {
                'observation_names': STATE_KEYS_TO_CONCATENATE,
                'observation_dim': observation_dims,
                'state_dim': combined_state.shape[1],
                'action_names': ['action'],
                'action_dim': root['action'].shape[1]
            }

    # 5. 保存 meta.json 文件
    meta_json_path = output_dir / 'meta.json'
    with meta_json_path.open('w') as f:
        json.dump(meta_data, f, indent=2)

    print("\n转换完成!")
    print(f"Zarr 数据集已保存至: {output_dir}")
    print(f"元数据文件已保存至: {meta_json_path}")


def main():
    parser = argparse.ArgumentParser(description="将NPZ格式的机器人轨迹数据转换为Zarr数据集格式")
    parser.add_argument('-i', '--input_dir', type=str, default="/data_1/flexiv_teleop_dataset/flexiv_pump_1bottle_raw",
                        help="包含所有 .npz 文件的输入目录。")
    parser.add_argument('-o', '--output_dir', type=str, default="/data_1/flexiv_teleop_dataset/zarr/flexiv_pump_1bottle_noForce",
                        help="用于保存生成的Zarr数据集的输出目录。")
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)

    convert_npz_to_zarr(input_path, output_path)

if __name__ == '__main__':
    main()