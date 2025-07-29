import h5py
import numpy as np
import tqdm
import quaternion
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tyro
import cv2

TOY_FEATURES = {
    "action": {
        "dtype": "float64",
        "shape": (7,),
        "names": [
            "target_eef_pose",  # (6,)
            "target_gripper_width",  # (1,)
        ],
    },
    "observation.state": {
        "dtype": "float64",
        "shape": (13,),
        "names": [
            "current_eef_pose",  # (6,)
            "gripper_width",  # (1,)
            "f_ext_base_frame"  # (6,)
        ],
    },
    # "observation.state": {
    #     "dtype": "float64",
    #     "shape": (7,),
    #     "names": [
    #         "current_eef_pose",  # (6,)
    #         "gripper_width",  # (1,)
    #     ],
    # },
    "observation.image": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": [
            "height",
            "width",
            "channels",
        ],
    },
    "observation.wrist_image": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": [
            "height",
            "width",
            "channels",
        ],
    },
}


def main(
    raw_dir: str,
    repo_id: str,
    push_to_hub: bool = False,
):
    """
    Convert Flexiv dataset to LeRobot format and save to a specified output path.

    Args:
      - raw_dir: Path to the directory containing trajectory.npz files
      - repo_id: Hugging Face repository ID to save the dataset
      - push_to_hub: Whether to push the converted dataset to Hugging Face Hub
    """
    raw_dir = Path(raw_dir)
    trajectory_file_paths = sorted(raw_dir.glob("*/trajectory.npz"))

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=30,
        robot_type="flexiv",
        features=TOY_FEATURES,
        use_videos=True,
    )
    dataset.start_image_writer(num_processes=4, num_threads=4)

    episodes = range(len(trajectory_file_paths))
    print(f"Found {len(episodes)} episodes")

    for ep_idx in tqdm.tqdm(episodes):
        trajectory_data = np.load(trajectory_file_paths[ep_idx])
        instruction = str(trajectory_data["instruction"])
        num_frames = trajectory_data["action"].shape[0]
        
        # preprocess the state
        state_tcp_pose = trajectory_data["tcp_pose"]  # (n, 7)
        state_gripper = trajectory_data["gripper_width"]  # (n,)
        gripper_column = state_gripper.reshape(-1, 1)  # (n, 1)
        state = np.concatenate([state_tcp_pose, gripper_column], axis=1)  # (n, 8)
        
        # get action
        action = trajectory_data["action"]  # (n, 7) xyz, quat
        # get f_ext_base_frame
        f_ext_base_frame = trajectory_data["f_ext_base_frame"]  # (n, 6)
        # get the images
        images = trajectory_data["image"]
        wrist_images = trajectory_data["wrist_image"]

        for frame_idx in range(num_frames - 1):
            frame = {}
            action_idx = action[frame_idx]
            action_quat = quaternion.quaternion(action_idx[3], action_idx[4], action_idx[5], action_idx[6])
            action_euler = quaternion.as_euler_angles(action_quat)  # array of shape (3,)
            action_gripper = action_idx[7]
            action_new = [
                action_idx[0], action_idx[1], action_idx[2],
                action_euler[0], action_euler[1], action_euler[2],
                action_gripper
            ]
            frame["action"] = np.array(action_new)
            
            state_idx = state[frame_idx]
            state_quat = quaternion.quaternion(state_idx[3], state_idx[4], state_idx[5], state_idx[6])
            state_euler = quaternion.as_euler_angles(state_quat)  # array of shape (3,)
            state_gripper = state_idx[7]
            state_new = [
                state_idx[0], state_idx[1], state_idx[2],
                state_euler[0], state_euler[1], state_euler[2],
                state_gripper
            ]
            f_ext_base_frame_idx = f_ext_base_frame[frame_idx]
            state_new = state_new + f_ext_base_frame_idx.tolist()
            frame["observation.state"] = np.array(state_new)

            # resize the image & wrist image
            resized_image = cv2.resize(images[frame_idx], (640, 480))
            resized_wrist_image = cv2.resize(wrist_images[frame_idx], (640, 480))
            frame["observation.image"] = images[frame_idx]
            frame["observation.wrist_image"] = resized_wrist_image

            dataset.add_frame(frame)

        dataset.save_episode(encode_videos=True, task=instruction)

    dataset.consolidate(run_compute_stats=True)

    if push_to_hub:
        dataset.push_to_hub(
            tags=["flexiv", "robot-learning"],
            private=True,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)

# python flexiv_pi0/examples/port_datasets/convert_to_video_mode.py --raw_dir /data/hairuo/forcevla_test_raw/flexiv_test_wipe_board_raw/ --repo_id flexiv_test/flexiv_wipe_board_inputForce