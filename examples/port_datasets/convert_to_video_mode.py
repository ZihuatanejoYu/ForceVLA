# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import shutil
from pathlib import Path

import cv2
import numpy as np
import tqdm
# from huggingface_hub import HfApi
import quaternion

from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset

TOY_FEATURES = {
    "action": {
        "dtype": "float64",
        "shape": (7,),
        "names": [
            "target_eef_pose", # (6,)
            "target_gripper_width", # (1,)
        ],
    },
    "observation.state": {
        "dtype": "float64",
        "shape": (7,),
        "names": [
            "current_eef_pose", # (6,)
            "gripper_width", # (1,)
            # "f_ext_base_frame" # (6, )
        ],
    },
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


def main(raw_dir: Path, root_path: Path, repo_id: str, push_to_hub: bool = True):

    if (HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    if (root_path).exists():
        shutil.rmtree(root_path)

    trajectory_file_paths = sorted(raw_dir.glob("*/trajectory.npz"))

    features = TOY_FEATURES
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=30,
        root=root_path,
        robot_type="flexiv",
        features=features,
        use_videos=True
    )
    episodes = range(len(trajectory_file_paths))
    print("episode number:", episodes)
    for ep_idx in tqdm.tqdm(episodes):
    # for ep_idx in tqdm.tqdm(range(10)):
        trajectory_data = np.load(trajectory_file_paths[ep_idx])
        
        instruction = str(trajectory_data["instruction"])
        num_frames = trajectory_data["action"].shape[0]

        # preprocess the state
        # state_q = trajectory_data["q"] # (n, 7)
        state_tcp_pose = trajectory_data["tcp_pose"] # (n, 7)
        state_gripper = trajectory_data["gripper_width"] # (n, )
        
        gripper_column = state_gripper.reshape(-1, 1) # (n, 1)
        # state = np.concatenate([state_q, gripper_column], axis=1)  # (n, 8)
        state = np.concatenate([state_tcp_pose, gripper_column], axis=1)  # (n, 8)

        # get action
        action = trajectory_data["action"] # (n, 7) xyz, quat

        # get f_ext_base_frame
        f_ext_base_frame = trajectory_data["f_ext_base_frame"] # (n, 6)

        # # use f_ext_base_frame of next frame as expect f_ext_base_frame
        # next_f_ext_base_frame = trajectory_data["f_ext_base_frame"][1:] # (n-1, 6) fx, fy, fz, mx, my, mz

        # # Duplicate the last row
        # last_f_ext_base_frame = trajectory_data["f_ext_base_frame"][-1:] # (1, 6)

        # # Concatenate to get the full expect_f_ext_base_frame
        # expect_f_ext_base_frame = np.concatenate([next_f_ext_base_frame, last_f_ext_base_frame], axis=0)  # (n, 6)

        # get the images
        images = trajectory_data["image"]
        wrist_images = trajectory_data["wrist_image"]

        for frame_idx in range(num_frames - 1):

            action_idx = action[frame_idx]
            
            action_quat = quaternion.quaternion(action_idx[3], action_idx[4], action_idx[5], action_idx[6])
            action_euler = quaternion.as_euler_angles(action_quat) # array of shape (3, )

            action_gripper = action_idx[7]

            # expect_f_ext_base_frame_idx = expect_f_ext_base_frame[frame_idx]

            action_new = [action_idx[0], action_idx[1], action_idx[2], action_euler[0], action_euler[1], action_euler[2], action_gripper]
            # action_new = action_new + expect_f_ext_base_frame_idx.tolist() # (7,) + (6,)

            frame = {
                # "action": action[frame_idx],
                "action": np.array(action_new),
                "task": instruction,
            }

            state_idx = state[frame_idx]
            
            state_quat = quaternion.quaternion(state_idx[3], state_idx[4], state_idx[5], state_idx[6])
            state_euler = quaternion.as_euler_angles(state_quat) # array of shape (3, )

            state_gripper = state_idx[7]

            state_new = [state_idx[0], state_idx[1], state_idx[2], state_euler[0], state_euler[1], state_euler[2], state_gripper]

            # f_ext_base_frame_idx = f_ext_base_frame[frame_idx]
            # state_new = state_new + f_ext_base_frame_idx.tolist()

            frame["observation.state"] = np.array(state_new)
            # frame["observation.state"] = state[frame_idx]

            # resize the image & wrist image
            resized_image = cv2.resize(images[frame_idx], (640, 480))
            resized_wrist_image = cv2.resize(wrist_images[frame_idx], (640, 480))

            frame["observation.image"] = resized_image
            frame["observation.wrist_image"] = resized_wrist_image

            dataset.add_frame(frame)

        dataset.save_episode(encode_videos=True)

    if push_to_hub:
        dataset.push_to_hub()
        hub_api = HfApi()
        hub_api.create_tag(repo_id, tag=CODEBASE_VERSION, repo_type="dataset")

    dataset.consolidate()

if __name__ == "__main__":
    # To try this script, modify the repo id with your own HuggingFace user (e.g cadene/pusht)
    repo_id = "Predentsi/flexiv_wipe_board_noForce"

    raw_dir = Path("/data_1/flexiv_teleop_dataset/flexiv_wipe_board_raw")
    root_path = Path("/data_1/flexiv_teleop_dataset/Predentsi/flexiv_wipe_board_noForce")

    # load raw dataset, create LeRobotDataset, populate it, push to hub
    main(raw_dir, root_path, repo_id=repo_id, push_to_hub=False)

    print(f"Saved {repo_id} lerobot dataset.")

    # Uncomment if you want to load the local dataset and explore it
    # dataset = LeRobotDataset(repo_id=repo_id)
    # breakpoint()
