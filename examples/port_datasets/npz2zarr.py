import numpy as np
import zarr
import os
import argparse
from glob import glob
import shutil
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from collections import deque
import quaternion

def load_npz_data(npz_path):
    """Load data from a single npz file with optimized loading"""
    with np.load(npz_path, allow_pickle=True) as data:
        return dict(data)

def quat_to_euler(q_w, q_x, q_y, q_z):
    """Convert quaternion to Euler angles (roll, pitch, yaw)"""
    q = quaternion.quaternion(q_w, q_x, q_y, q_z)
    return quaternion.as_euler_angles(q)  # returns (roll, pitch, yaw)

def process_episode_batch(npz_path):
    """Optimized batch processing with quaternion to Euler conversion"""
    try:
        data = load_npz_data(npz_path)
        
        # Process images in bulk
        wrist_imgs = np.stack([cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in data['wrist_image']])
        imgs = np.stack([cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in data['image']])
        
        # Original states: [pos(3), quat(4)] -> convert to [pos(3), euler(3), gripper(1)]
        states = np.zeros((len(data['tcp_pose']), 7), dtype=np.float32)
        for i, state in enumerate(data['tcp_pose']):
            pos = state[:3]
            euler = quat_to_euler(*state[3:7])
            gripper = data['gripper_width'][i]
            states[i] = np.concatenate([pos, euler, [gripper]])
        
        # Original actions: [pos(3), quat(4), gripper(1)] -> convert to [pos(3), euler(3), gripper(1)]
        actions = np.zeros((len(data['action']), 7), dtype=np.float32)
        for i, action in enumerate(data['action']):
            pos = action[:3]
            euler = quat_to_euler(*action[3:7])
            gripper = action[7]
            actions[i] = np.concatenate([pos, euler, [gripper]])
        
        if len(states) == 0:
            return None
            
        return {
            'wrist_imgs': wrist_imgs,
            'imgs': imgs,
            'states': states,
            'actions': actions,
            'n_frames': len(states)
        }
    except Exception as e:
        print(f"\nError processing {npz_path}: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Optimized NPZ to Zarr converter with Euler angles")
    parser.add_argument("input_dir", type=str, help="Directory containing npz files")
    parser.add_argument("output_path", type=str, help="Output zarr path")
    parser.add_argument("--max_episodes", type=int, default=None, help="Max episodes to process")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    args = parser.parse_args()

    # Find all npz files
    npz_files = sorted(glob(os.path.join(args.input_dir, "*/trajectory.npz")))
    if args.max_episodes is not None:
        npz_files = npz_files[:args.max_episodes]
    
    if not npz_files:
        print(f"No npz files found in {args.input_dir}")
        return

    # Prepare output directory
    if os.path.exists(args.output_path):
        shutil.rmtree(args.output_path)

    # Create zarr groups
    zarr_root = zarr.group(args.output_path)
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")

    # Process files in parallel
    results = deque()
    episode_ends = []
    total_frames = 0
    
    print(f"Processing {len(npz_files)} episodes with {args.workers} workers...")
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_episode_batch, f): f for f in npz_files}
        
        for future in tqdm(as_completed(futures), total=len(npz_files)):
            result = future.result()
            if result:
                results.append(result)
                total_frames += result['n_frames']
                episode_ends.append(total_frames)

    if not results:
        print("No valid episodes found")
        return
        
    # Get sample data for shape info
    sample = results[0]
    img_shape = sample['wrist_imgs'].shape[1:]
    state_dim = sample['states'].shape[1]  # Now 7 (pos3 + euler3 + gripper1)
    action_dim = sample['actions'].shape[1]  # Now 7 (pos3 + euler3 + gripper1)

    print(f"Allocating arrays for {total_frames} total frames...")
    
    # Create zarr datasets with pre-allocation
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    
    wrist_ds = zarr_data.zeros(
        "wrist_camera",
        shape=(total_frames, *img_shape),
        chunks=(100, *img_shape),
        dtype='uint8',
        compressor=compressor
    )
    
    front_ds = zarr_data.zeros(
        "front_camera",
        shape=(total_frames, *img_shape),
        chunks=(100, *img_shape),
        dtype='uint8',
        compressor=compressor
    )
    
    state_ds = zarr_data.zeros(
        "state",
        shape=(total_frames, state_dim),
        chunks=(100, state_dim),
        dtype='float32',
        compressor=compressor
    )
    
    action_ds = zarr_data.zeros(
        "action",
        shape=(total_frames, action_dim),
        chunks=(100, action_dim),
        dtype='float32',
        compressor=compressor
    )

    # Fill the datasets
    print("Writing data to zarr...")
    current_idx = 0
    for result in tqdm(results):
        n = result['n_frames']
        end_idx = current_idx + n
        
        wrist_ds[current_idx:end_idx] = result['wrist_imgs']
        front_ds[current_idx:end_idx] = result['imgs']
        state_ds[current_idx:end_idx] = result['states']
        action_ds[current_idx:end_idx] = result['actions']
        
        current_idx = end_idx

    # Save episode ends
    zarr_meta.create_dataset(
        "episode_ends",
        data=np.array(episode_ends),
        dtype="int64",
        compressor=compressor
    )

    print("\nFinal dataset structure:")
    print(f"- wrist_camera: {wrist_ds.shape} (dtype: {wrist_ds.dtype})")
    print(f"- front_camera: {front_ds.shape} (dtype: {front_ds.dtype})")
    print(f"- state: {state_ds.shape} (dtype: {state_ds.dtype})")
    print(f"  • Contains: position(3), euler angles(3), gripper_width(1)")
    print(f"- action: {action_ds.shape} (dtype: {action_ds.dtype})")
    print(f"  • Contains: position(3), euler angles(3), gripper_action(1)")
    print(f"- episode_ends: {len(episode_ends)} markers")
    
    print(f"\nSuccessfully saved optimized zarr dataset to {args.output_path}")

if __name__ == "__main__":
    main()

# python npz_to_zarr_euler.py /path/to/npz/files /output/path.zarr \
#     --max_episodes 1000 \
#     --workers 12