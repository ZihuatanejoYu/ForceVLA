import shutil
import numpy as np
import zarr
from pathlib import Path
import tqdm
import cv2
import pdb
import quaternion


def main(raw_dir: Path, output_zarr_path: Path):
    if output_zarr_path.exists():
        shutil.rmtree(output_zarr_path)

    trajectory_file_paths = sorted(raw_dir.glob("*/trajectory.npz"))

    zarr_root = zarr.group(output_zarr_path)
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")

    # Initialize arrays to store data
    front_camera_arrays = []
    wrist_images_arrays = []
    state_arrays = []
    action_arrays = []
    episode_ends_arrays = []
    total_frames = 0

    for ep_idx in tqdm.tqdm(range(len(trajectory_file_paths)), desc="Processing episodes"):
        trajectory_path = trajectory_file_paths[ep_idx]
        data = np.load(trajectory_path)

        images = data["image"]
        wrist_images = data["wrist_image"]
        gripper_width = data['gripper_width']
        actions = data['action']
        states = data['tcp_pose']

        quats = states[:, 3:7]
        eulers = np.array([quaternion.as_euler_angles(np.quaternion(*q)) for q in quats])
        states_new = np.hstack([
            states[:, :3],
            eulers,
        ])

        quats = actions[:, 3:7]
        eulers = np.array([quaternion.as_euler_angles(np.quaternion(*q)) for q in quats])
        actions_new = np.hstack([
            actions[:, :3],
            eulers,
            actions[:, 7:8]
        ])

        if len(images) == len(wrist_images) == len(gripper_width) == len(states) == len(actions):
            num_frames = len(images)
        else:
            raise ValueError("Mismatch in lengths of images, wrist_images, gripper_width, states, and actions.")
        
        resized_images = np.array([cv2.resize(img, (640, 480)) for img in images])
        resized_wrist_images = np.array([cv2.resize(img, (640, 480)) for img in wrist_images])
        front_camera_arrays.append(resized_images)
        wrist_images_arrays.append(resized_wrist_images)
        
        gripper_width_expanded = gripper_width[:num_frames].reshape(-1, 1)
        states_combined = np.concatenate([states_new[:num_frames], gripper_width_expanded], axis=1)
        state_arrays.append(states_combined)
        
        action_arrays.append(actions_new[:num_frames])
        
        total_frames += num_frames
        episode_ends_arrays.append(total_frames)

    front_camera_arrays = np.concatenate(front_camera_arrays, axis=0)
    wrist_images_arrays = np.concatenate(wrist_images_arrays, axis=0)
    state_arrays = np.concatenate(state_arrays, axis=0)
    action_arrays = np.concatenate(action_arrays, axis=0)
    
    front_camera_arrays = np.moveaxis(front_camera_arrays, -1, 1)  # NHWC to NCHW
    wrist_images_arrays = np.moveaxis(wrist_images_arrays, -1, 1)  # NHWC to NCHW

    # Create zarr datasets with compression
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    
    # Define chunk sizes
    front_camera_chunk_size = (100, *front_camera_arrays.shape[1:])
    wrist_images_chunk_size = (100, *wrist_images_arrays.shape[1:])
    state_chunk_size = (100, state_arrays.shape[1])
    action_chunk_size = (100, action_arrays.shape[1])
    
    zarr_data.create_dataset(
        "front_camera",
        data=front_camera_arrays,
        chunks=front_camera_chunk_size,
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "wrist_camera",
        data=wrist_images_arrays,
        chunks=wrist_images_chunk_size,
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "state",
        data=state_arrays,
        chunks=state_chunk_size,
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "action",
        data=action_arrays,
        chunks=action_chunk_size,
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_meta.create_dataset(
        "episode_ends",
        data=episode_ends_arrays,
        dtype="int64",
        overwrite=True,
        compressor=compressor,
    )

if __name__ == "__main__":
    raw_dir = Path("/data/hairuo/forcevla_train_raw/test")
    output_zarr_path = Path("/home/hairuo/.cache/zarr")
    
    main(raw_dir, output_zarr_path)
    print(f"Successfully converted NPZ files to Zarr format at {output_zarr_path}")