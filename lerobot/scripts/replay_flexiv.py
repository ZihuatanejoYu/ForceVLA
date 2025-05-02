

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
"""
Utilities to control a robot.

Useful to record a dataset, replay a recorded episode, run the policy on your robot
and record an evaluation dataset, and to recalibrate your robot if needed.

Examples of usage:

- Replay this test episode:
```bash
python lerobot/scripts/replay_flexiv.py \
    --robot.type=aloha \
    --control.type=replay \
    --control.fps=30 \
    --control.repo_id=Predentsi/flexiv_box_pp_targetpose \
    --control.root=/hdd1/flexiv_teleop_dataset/Predentsi/flexiv_box_pp_targetpose \
    --control.episode=10
```
"""
import pdb
import sys
from copy import copy
import torch
from contextlib import nullcontext

# 添加绝对路径
sys.path.append('/home/qiaojun/flexiv_manipulation/example_py/lerobot/lerobot')

# Import Flexiv RDK Python library
sys.path.insert(0, "/home/qiaojun/flexiv_manipulation/lib_py")

import time
import argparse
import numpy as np
import quaternion  # 用于四元数操作

# Flexiv RDK Python library
import flexivrdk

import logging
import time
from dataclasses import asdict
from pprint import pformat

# from safetensors.torch import load_file, save_file
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.factory import make_policy
from lerobot.common.robot_devices.control_configs import (
    CalibrateControlConfig,
    ControlPipelineConfig,
    RecordControlConfig,
    RemoteRobotConfig,
    ReplayControlConfig,
    TeleoperateControlConfig,
)
from lerobot.common.robot_devices.control_utils import (
    control_loop,
    init_keyboard_listener,
    log_control_info,
    record_episode,
    reset_environment,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
    stop_recording,
    warmup_record,
)
from lerobot.common.robot_devices.robots.utils import Robot, make_robot_from_config
from lerobot.common.robot_devices.utils import busy_wait, safe_disconnect
from lerobot.common.utils.utils import has_method, init_logging, log_say, get_safe_torch_device
from lerobot.configs import parser

from utils.d415_double_rs_record import DualRealSenseModule, get_rgbd
from utils.utility import parse_pt_states


# 设置日志
log = flexivrdk.Log()


@parser.wrap()
def control_robot(cfg: ControlPipelineConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    try:
        # 初始化机器人
        log.info("Initializing robot...")
        robot = flexivrdk.Robot("192.168.2.100", "192.168.2.106")

        # 清除机器人故障（如果有）
        if robot.isFault():
            log.warn("Fault occurred on robot server, trying to clear...")
            robot.clearFault()
            time.sleep(2)
            if robot.isFault():
                log.error("Fault cannot be cleared, exiting...")
                return
            log.info("Fault on robot server is cleared")

        # 启用机器人
        log.info("Enabling robot...")
        robot.enable()

        # 等待机器人进入操作状态
        log.info("Waiting for robot to become operational...")
        seconds_waited = 0
        while not robot.isOperational():
            time.sleep(1)
            seconds_waited += 1
            if seconds_waited == 10:
                log.warn("Still waiting for robot to become operational, please check the robot mode and fault status.")
        log.info("Robot is now operational")

        # 设置机器人模式为 Cartesian Motion Force 模式
        robot.setMode(flexivrdk.Mode.NRT_CARTESIAN_MOTION_FORCE)

        # 初始化夹爪
        log.info("Initializing gripper...")
        gripper = flexivrdk.Gripper(robot)
        gripper.move(1, 0.1, 20)  # 打开夹爪

        mode = flexivrdk.Mode
        robot.setMode(mode.NRT_PRIMITIVE_EXECUTION)
        log.info("Executing primitive: Move to record HOME Joints")

        # Send command to robot
        robot.executePrimitive("MoveJ(target=-4.77 11.06 2.92 113.00 9.84 18.09 -8.46)")

        # Wait for reached target
        while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
            time.sleep(1)
        # time.sleep(2)

        robot.setMode(mode.NRT_CARTESIAN_MOTION_FORCE)

        cfg = cfg.control

        dataset = LeRobotDataset(cfg.repo_id, root=cfg.root, episodes=[cfg.episode])
        actions = dataset.hf_dataset.select_columns("action")

        log_say("Replaying episode", cfg.play_sounds, blocking=True)
        for idx in range(dataset.num_frames):
            start_episode_t = time.perf_counter()

            action = actions[idx]["action"]
            
            robot_states = flexivrdk.RobotStates()
            robot.getRobotStates(robot_states)
            gripper_states = flexivrdk.GripperStates()
            gripper.getGripperStates(gripper_states)

            # target tcp pose
            target_pos = np.array(action[:3])  # 目标位置 (x, y, z)
            target_quat = quaternion.quaternion(action[3], action[4], action[5], action[6])  # 目标四元数
            target_gripper = action[7]  # 夹爪开合度

            # 发送目标位姿到机器人
            target_wrench = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 目标力矩（设置为零）
            new_target = [target_pos[0], target_pos[1], target_pos[2], target_quat.w, target_quat.x, target_quat.y, target_quat.z]
            # robot.sendCartesianMotionForce(new_target, target_wrench, 0.02)
            step = 0
            while True:
                robot.sendCartesianMotionForce(new_target, target_wrench, 0.05)
                robot.getRobotStates(robot_states)
                robot_pose = robot_states.tcpPose.copy()
                if np.linalg.norm(np.array(new_target[:3]) - np.array(robot_pose[:3])) < 0.005:
                    break
                step += 1
                if step > 10:
                    break

            gripper.move(target_gripper, 0.1, 20)

            # 控制循环频率
            time.sleep(1.0 / cfg.fps)

        

    except Exception as e:
        log.error(f"An error occurred: {str(e)}")
    finally:
        log.info("Inference script finished.")


if __name__ == "__main__":
    control_robot()
