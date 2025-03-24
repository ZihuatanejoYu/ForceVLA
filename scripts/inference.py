#!/usr/bin/env python

"""flexiv_inference.py

This script performs inference using a policy server to control the Flexiv robot arm.
"""

import sys

# 添加绝对路径
sys.path.append('/home/qiaojun/flexiv_manipulation/example_py/flexiv_pi0-dev')

# Import Flexiv RDK Python library
sys.path.insert(0, "/home/qiaojun/flexiv_manipulation/lib_py")

import time
import argparse
import numpy as np
import quaternion  # 用于四元数操作

# Flexiv RDK Python library
import flexivrdk
import cv2

# Policy client
from openpi_client import websocket_client_policy

# Camera module (假设你已经有一个相机模块)
from utils.d415_double_rs_record import DualRealSenseModule, get_rgbd

# 设置日志
log = flexivrdk.Log()

def make_flexiv_example(robot_states, wrist_image, image, prompt) -> dict:
    """
    构造一个与 Flexiv 配置兼容的观测示例。
    Args:
        robot_states: 机器人状态对象
        wrist_image: 腕部相机图像
        image: 全局相机图像
    Returns:
        dict: 包含状态、图像和提示的观测字典
    """
    return {
        "state": np.array(robot_states.q),  # 使用当前关节状态
        "image": image,  # 使用全局相机图像
        "wrist_image": wrist_image,  # 使用腕部相机图像
        "prompt": prompt,  # 任务提示
    }

def main(robot_ip, robot_local_ip, policy_host="localhost", policy_port=8000, control_frequency=30):
    """
    主函数，用于运行 Flexiv 推理脚本。
    Args:
        robot_ip (str): 机器人服务器 IP 地址
        robot_local_ip (str): 本地机器 IP 地址
        policy_host (str): 策略服务器主机地址
        policy_port (int): 策略服务器端口
        control_frequency (int): 控制频率 (Hz)
    """
    # 初始化策略客户端
    log.info("Initializing policy client...")
    client = websocket_client_policy.WebsocketClientPolicy(host=policy_host, port=policy_port)

    # 初始化相机
    log.info("Initializing cameras...")
    cameras = DualRealSenseModule()

    try:
        # 初始化机器人
        log.info("Initializing robot...")
        robot = flexivrdk.Robot(robot_ip, robot_local_ip)

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
        gripper.move(0.1, 0.1, 20)  # 打开夹爪
        time.sleep(2)

        prompt = input("Please input task prompt:")

        # 主控制循环
        log.info("Starting inference loop...")
        while True:
            # 获取机器人状态
            robot_states = flexivrdk.RobotStates()
            robot.getRobotStates(robot_states)

            # 获取相机图像
            image, whole_depth,_, wrist_image, wrist_depth, _ = get_rgbd(cameras)

            # 构造观测
            observation = make_flexiv_example(robot_states, wrist_image, image, prompt)

            # 调用策略服务器获取动作
            log.info("Querying policy server for actions...")
            action_chunk = client.infer(observation)["actions"]
            action = action_chunk[0]  # 使用第一个动作

            # 解析动作
            target_pos = action[:3]  # 目标位置 (x, y, z)
            target_quat = quaternion.quaternion(action[3], action[4], action[5], action[6])  # 目标四元数
            target_gripper = action[7]  # 夹爪开合度

            # 发送目标位姿到机器人
            target_wrench = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 目标力矩（设置为零）
            new_target = [target_pos[0], target_pos[1], target_pos[2], target_quat.w, target_quat.x, target_quat.y, target_quat.z]
            robot.sendCartesianMotionForce(new_target, target_wrench, 0.02)

            # 控制夹爪
            gripper.move((1.0 - target_gripper), 0.1, 20)

            # 控制循环频率
            time.sleep(1.0 / control_frequency)

    except Exception as e:
        log.error(f"An error occurred: {str(e)}")
    finally:
        log.info("Inference script finished.")


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Flexiv inference script for policy-based control.")
    parser.add_argument("--robot_ip", type=str, default="192.168.2.100", help="Robot server IP address")
    parser.add_argument("--robot_local_ip", type=str, default="192.168.2.106", help="Local machine IP address")
    parser.add_argument("--policy_host", type=str, default="localhost", help="Policy server host address")
    parser.add_argument("--policy_port", type=int, default=8000, help="Policy server port")
    parser.add_argument("--control_frequency", type=int, default=10, help="Control frequency (Hz)")
    args = parser.parse_args()

    # 运行主函数
    main(args.robot_ip, args.robot_local_ip, args.policy_host, args.policy_port, args.control_frequency)