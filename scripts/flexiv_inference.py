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
from openpi_client.runtime.agents import policy_agent as _policy_agent
from openpi_client import action_chunk_broker

# Camera module (假设你已经有一个相机模块)
from utils.d415_double_rs_record import DualRealSenseModule, get_rgbd
from utils.utility import parse_pt_states

# 设置日志
log = flexivrdk.Log()

def make_flexiv_example(robot_states, gripper_states, wrist_image, image, prompt, force=False) -> dict:
    """
    构造一个与 Flexiv 配置兼容的观测示例。
    Args:
        robot_states: 机器人状态对象
        wrist_image: 腕部相机图像
        image: 全局相机图像
    Returns:
        dict: 包含状态、图像和提示的观测字典
    """

    state_idx = np.array(robot_states.tcpPose)
            
    state_quat = quaternion.quaternion(state_idx[3], state_idx[4], state_idx[5], state_idx[6])
    state_euler = quaternion.as_euler_angles(state_quat) # array of shape (3, )

    state_gripper = gripper_states.width

    state_new = [state_idx[0], state_idx[1], state_idx[2], state_euler[0], state_euler[1], state_euler[2], state_gripper]

    if force:
        f_ext_base_frame = robot_states.extWrenchInBase
        state_new += f_ext_base_frame

    return {
        "state": np.array(state_new),  
        "image": image,  # 使用全局相机图像
        "wrist_image": wrist_image,  # 使用腕部相机图像
        "prompt": prompt,  # 任务提示
    }

def main(args):
    """
    主函数，用于运行 Flexiv 推理脚本。
    Args:
        robot_ip (str): 机器人服务器 IP 地址
        robot_local_ip (str): 本地机器 IP 地址
        policy_host (str): 策略服务器主机地址
        policy_port (int): 策略服务器端口
        control_frequency (int): 控制频率 (Hz)
    """
    robot_ip = args.robot_ip
    robot_local_ip = args.robot_local_ip
    task_prompt = args.prompt
    policy_host = args.policy_host
    policy_port = args.policy_port
    control_frequency = args.control_frequency
    force_flag = args.force

    # 初始化策略客户端
    log.info("Initializing policy client...")
    client = websocket_client_policy.WebsocketClientPolicy(host=policy_host, port=policy_port)
    agent=_policy_agent.PolicyAgent(
        policy=action_chunk_broker.ActionChunkBroker(
            policy=client,
            action_horizon=50,
        )
    )

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

        # prompt = input("Please input task prompt:")
        prompt = task_prompt
        print("Please input task prompt:", prompt)

        # 主控制循环
        log.info("Starting inference loop...")
        while True:
            # 获取机器人状态
            robot_states = flexivrdk.RobotStates()
            robot.getRobotStates(robot_states)
            gripper_states = flexivrdk.GripperStates()
            gripper.getGripperStates(gripper_states)

            # 获取相机图像
            image, whole_depth,_, wrist_image, wrist_depth, _ = get_rgbd(cameras)

            # 构造观测
            observation = make_flexiv_example(robot_states, gripper_states, wrist_image, image, prompt, force=force_flag)

            # 调用策略服务器获取动作
            log.info("Querying policy server for actions...")
            action_chunk = agent.get_action(observation)["actions"]
            action = action_chunk.tolist()  # 使用第一个动作
            # action_chunk = client.infer(observation)["actions"]
            # action = action_chunk[0]

            # 解析动作
            target_pos = np.array(action[:3])  # 目标位置 (x, y, z)
            action_euler = np.array(action[3:6])
            target_quat = quaternion.from_euler_angles(action_euler)
            # target_quat = quaternion.quaternion(action[3], action[4], action[5], action[6])  # 目标四元数
            target_gripper = action[6]  # 夹爪开合度


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

            # 控制夹爪
            gripper.move(target_gripper, 0.1, 20)

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
    parser.add_argument("--robot_local_ip", type=str, default="192.168.2.206", help="Local machine IP address")
    parser.add_argument("--policy_host", type=str, default="localhost", help="Policy server host address")
    parser.add_argument("--policy_port", type=int, default=8000, help="Policy server port")
    parser.add_argument("--control_frequency", type=int, default=30, help="Control frequency (Hz)")
    parser.add_argument("--prompt", type=str, default="Insert the power plug into the socket.", help="Task instruction.")
    parser.add_argument("--force", type=bool, default=True, help="Control flag of whether input force or not.")
    args = parser.parse_args()

    # 运行主函数
    main(args)