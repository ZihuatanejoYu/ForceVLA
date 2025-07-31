#!/usr/bin/env python

"""teleop_with_recording.py

This script combines Quest VR controller teleoperation with simultaneous robot trajectory recording.
"""

import json
import pdb
import time
import argparse
import threading
import os
import numpy as np

# Utility methods
from utility import quat2eulerZYX
from utility import parse_pt_states
from utility import list2str

# Import Flexiv RDK Python library
import sys
sys.path.insert(0, "../lib_py")
import flexivrdk
import quaternion
from quest_receive import quest_teleop

from d415_double_rs_record import DualRealSenseModule, get_rgbd


class TrajectoryRecorder:
    def __init__(self, output_file="robot_trajectory.npz"):
        self.output_file = output_file
        # 初始化各个状态的列表
        self.timestamps = []
        self.q_list = []
        self.theta_list = []
        self.dq_list = []
        self.dtheta_list = []
        self.tau_list = []
        self.tau_des_list = []
        self.tau_dot_list = []
        self.tau_ext_list = []
        self.tcp_pose_list = []
        self.tcp_pose_d_list = []
        self.tcp_velocity_list = []
        self.camera_pose_list = []
        self.flange_pose_list = []
        self.ft_sensor_raw_list = []
        self.f_ext_tcp_frame_list = []
        self.f_ext_base_frame_list = []

        self.gripper_width_list = []

        # 是否正在记录
        self.is_recording = True

        self.cameras = DualRealSenseModule()

        self.wrist_image_list = []
        self.image_list = []
        self.wrist_depth_list = []
        self.whole_depth_list = []

        self.action_list = []

    def add_state(self, robot_states, gripper_states, quest_input=None):
        """添加一个时刻的状态数据"""
        if not self.is_recording:
            return
            
        try:
            self.timestamps.append(time.time())
            self.q_list.append([float(i) for i in robot_states.q])
            self.theta_list.append([float(i) for i in robot_states.theta])
            self.dq_list.append([float(i) for i in robot_states.dq])
            self.dtheta_list.append([float(i) for i in robot_states.dtheta])
            self.tau_list.append([float(i) for i in robot_states.tau])
            self.tau_des_list.append([float(i) for i in robot_states.tauDes])
            self.tau_dot_list.append([float(i) for i in robot_states.tauDot])
            self.tau_ext_list.append([float(i) for i in robot_states.tauExt])
            self.tcp_pose_list.append([float(i) for i in robot_states.tcpPose])
            self.tcp_pose_d_list.append([float(i) for i in robot_states.tcpPoseDes])
            self.tcp_velocity_list.append([float(i) for i in robot_states.tcpVel])
            self.camera_pose_list.append([float(i) for i in robot_states.camPose])
            self.flange_pose_list.append([float(i) for i in robot_states.flangePose])
            self.ft_sensor_raw_list.append([float(i) for i in robot_states.ftSensorRaw])
            self.f_ext_tcp_frame_list.append([float(i) for i in robot_states.extWrenchInTcp])
            self.f_ext_base_frame_list.append([float(i) for i in robot_states.extWrenchInBase])

            self.gripper_width_list.append(float(gripper_states.width))

            image, whole_depth, _, wrist_image, wrist_depth, _ = get_rgbd(self.cameras)
            
            self.wrist_image_list.append(np.array(wrist_image).copy())
            # self.wrist_depth_list.append(np.array(wrist_depth).copy())
            self.image_list.append(np.array(image).copy())
            # self.whole_depth_list.append(np.array(whole_depth).copy())
            
        except KeyboardInterrupt:
            # 不保存当前帧，直接重新抛出异常
            raise
        except Exception as e:
            print(f"添加状态数据时发生错误: {str(e)}")
    
    def add_action(self, offset_pos, offset_quat, gripper_close):
        """添加一个时刻的action数据"""
        if not self.is_recording:
            return
        
        action = [offset_pos[0], offset_pos[1], offset_pos[2], offset_quat.w, offset_quat.x, offset_quat.y, offset_quat.z, gripper_close]
        self.action_list.append([float(i) for i in action])

    def align_frames(self):
        if len(self.action_list) == (len(self.timestamps) - 1):
            self.action_list.append(self.action_list[-1])
        if len(self.image_list) == (len(self.timestamps) - 1):
            self.image_list.append(self.image_list[-1])
        if len(self.wrist_image_list) == (len(self.timestamps) - 1):
            self.wrist_image_list.append(self.wrist_image_list[-1])
        assert len(self.action_list) == len(self.timestamps), \
                f"Action长度({len(self.action_list)})不等于timestamps长度({len(self.timestamps)})"
        assert len(self.image_list) == len(self.timestamps), \
                f"Image长度({len(self.image_list)})不等于timestamps长度({len(self.timestamps)})"
        assert len(self.wrist_image_list) == len(self.timestamps), \
                f"wrist_image长度({len(self.wrist_image_list)})不等于timestamps长度({len(self.timestamps)})"

    def save_trajectory(self, task):
        """保存整条轨迹数据到同一个文件"""
        self.align_frames()

        trajectory_data = {
            'timestamps': np.array(self.timestamps),
            'q': np.array(self.q_list),
            'theta': np.array(self.theta_list),
            'dq': np.array(self.dq_list),
            'dtheta': np.array(self.dtheta_list),
            'tau': np.array(self.tau_list),
            'tau_des': np.array(self.tau_des_list),
            'tau_dot': np.array(self.tau_dot_list),
            'tau_ext': np.array(self.tau_ext_list),
            'tcp_pose': np.array(self.tcp_pose_list),
            'tcp_pose_d': np.array(self.tcp_pose_d_list),
            'tcp_velocity': np.array(self.tcp_velocity_list),
            'camera_pose': np.array(self.camera_pose_list),
            'flange_pose': np.array(self.flange_pose_list),
            'ft_sensor_raw': np.array(self.ft_sensor_raw_list),
            'f_ext_tcp_frame': np.array(self.f_ext_tcp_frame_list),
            'f_ext_base_frame': np.array(self.f_ext_base_frame_list),
            'gripper_width': np.array(self.gripper_width_list),
            'wrist_image': np.array(self.wrist_image_list),
            'wrist_depth': np.array(self.wrist_depth_list),
            'image': np.array(self.image_list),
            'whole_depth': np.array(self.whole_depth_list),
            'action': np.array(self.action_list),
            'instruction': task
        }
        print("Saving file...")
        np.savez(self.output_file, **trajectory_data)
        print("task name:", task)
        print("epoch sequence length:", len(self.timestamps))
        
def print_description():
    """
    Print tutorial description.
    """
    print("This script combines Quest VR controller teleoperation with simultaneous robot trajectory recording.")
    print()

def get_cur_pose(robot, gripper):

    # 获取机器人当前状态
    robot_states = flexivrdk.RobotStates()
    robot.getRobotStates(robot_states)
    # 获取当前机器人TCP位姿
    current_tcp_pose = robot_states.tcpPose
    current_tcp_pos = np.array([current_tcp_pose[0], current_tcp_pose[1], current_tcp_pose[2]])
    current_tcp_quat = quaternion.quaternion(current_tcp_pose[3], current_tcp_pose[4], current_tcp_pose[5], current_tcp_pose[6])

    gripper_states = flexivrdk.GripperStates()
    gripper.getGripperStates(gripper_states)

    return robot_states, current_tcp_pos, current_tcp_quat, gripper_states

def main(task, scene_num=0, file_save_path="./teleop_recordings/", record_frequency = 30, home_target="-4.77 11.06 2.92 113.00 9.84 18.09 -8.46"):
    # Define alias
    log = flexivrdk.Log()
    mode = flexivrdk.Mode
    
    # Print description
    log.info("Script description:")
    print_description()
    
    # 创建保存目录
    recording_dir = os.path.join(file_save_path, str(scene_num))
    os.makedirs(recording_dir, exist_ok=True)
    
    # 创建轨迹记录器
    recorder = TrajectoryRecorder(os.path.join(recording_dir, "trajectory.npz"))
    
    # 初始化Quest遥操作控制器
    quest_controller = quest_teleop()
    
    try:
        # RDK Initialization
        # ==========================================================================================
        # Instantiate robot interface
        robot = flexivrdk.Robot("192.168.2.100", "192.168.2.106")

        # Clear fault on robot server if any
        if robot.isFault():
            log.warn("Fault occurred on robot server, trying to clear ...")
            # Try to clear the fault
            robot.clearFault()
            time.sleep(2)
            # Check again
            if robot.isFault():
                log.error("Fault cannot be cleared, exiting ...")
                return
            log.info("Fault on robot server is cleared")

        # Enable the robot, make sure the E-stop is released before enabling
        log.info("Enabling robot ...")
        robot.enable()
        
        # Wait for the robot to become operational
        seconds_waited = 0
        while not robot.isOperational():
            time.sleep(1)
            seconds_waited += 1
            if seconds_waited == 10:
                log.warn(
                    "Still waiting for robot to become operational, please check that the robot 1) "
                    "has no fault, 2) is in [Auto (remote)] mode")

        log.info("Robot is now operational")
        robot.setMode(mode.NRT_PLAN_EXECUTION)

        # Instantiate gripper control interface
        gripper = flexivrdk.Gripper(robot)

        log.info("Opening gripper")
        gripper.move(0.1, 0.1, 20)
        time.sleep(1)

        robot.setMode(mode.NRT_PRIMITIVE_EXECUTION)
        log.info("Executing primitive: Move to record HOME Joints")

        # Send command to robot
        robot.executePrimitive(f"MoveJ(target={home_target})")

        # Wait for reached target
        while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
            time.sleep(1)

        # Switch to cartesian motion force mode for teleoperation
        robot.setMode(mode.NRT_CARTESIAN_MOTION_FORCE)
        
        log.info("Starting teleoperation with trajectory recording...")

        last_input = None
        frame_cnt = 0
        last_robot_states, last_tcp_pos, last_tcp_quat, last_gripper_states = get_cur_pose(robot, gripper)
        cur_tcp_pos, cur_tcp_quat = None, None

        try:
            while True:
                # 获取机器人当前状态
                robot_states = flexivrdk.RobotStates()
                robot.getRobotStates(robot_states)
                gripper_states = flexivrdk.GripperStates()
                gripper.getGripperStates(gripper_states)
                
                # 更新Quest控制器的关节状态
                joint_states = robot_states.q
                quest_controller.joint_states = np.array(joint_states)
                
                # 获取Quest控制器输入
                current_input, _, _ = quest_controller.get_input_frame()

                if current_input is None:
                    time.sleep(0.02)
                    continue

                if current_input['Y'] and current_input['B']:
                    log.info("Y + B detected, recording stopped by user. Saving trajectory...")
                    break
                    
                if last_input is None:
                    last_input = current_input
                
                # 记录机器人状态
                recorder.add_state(robot_states, gripper_states)
                
                # 获取当前机器人TCP位姿
                current_tcp_pose = robot_states.tcpPose
                current_tcp_pos = np.array([current_tcp_pose[0], current_tcp_pose[1], current_tcp_pose[2]])
                current_tcp_quat = quaternion.quaternion(current_tcp_pose[3], current_tcp_pose[4], current_tcp_pose[5], current_tcp_pose[6])
                
                # 获取Quest控制器输入位姿
                current_input_pos = np.array([current_input['rightPos']["z"], -current_input['rightPos']["x"], current_input['rightPos']["y"]])
                current_input_quat = quaternion.quaternion(current_input['rightRot']["w"], current_input['rightRot']["z"], current_input['rightRot']["x"], current_input['rightRot']["y"])
                
                # 当按下Quest右手柄按钮时进行遥操作
                if current_input['rightHand'] > 0.5:
                    if last_input['rightHand'] <= 0.5:
                        # 记录起始位姿
                        start_tcp_pos = current_tcp_pos
                        start_tcp_quat = current_tcp_quat
                        start_input_pos = current_input_pos
                        start_input_quat = current_input_quat

                    # 计算位置和旋转偏移
                    offset_pos = current_input_pos - start_input_pos
                    offset_quat = quaternion.quaternion.inverse(start_input_quat) * current_input_quat

                    # 计算目标位姿
                    pos = start_tcp_pos + offset_pos
                    quat = start_tcp_quat * offset_quat
                
                    # 发送目标位姿到机器人
                    target_wrench = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    new_target = [pos[0], pos[1], pos[2], quat.w, quat.x, quat.y, quat.z]
                    robot.sendCartesianMotionForce(new_target, target_wrench)

                    # 控制夹爪
                    # 根据右手食指扳机参数调整夹爪开合度
                    gripper_close = 0.09 * (1 - current_input['rightIndex']) + 0.01
                    # gripper_flag = (current_input['rightIndex'] > 0)

                    gripper.move(gripper_close, 0.1, 20)

                    # cur_robot_states, cur_tcp_pos, cur_tcp_quat, cur_gripper_states = get_cur_pose(robot, gripper)
                    # action_pos = cur_tcp_pos - last_tcp_pos
                    # action_quat = 1.0 * cur_tcp_quat / last_tcp_quat

                    # recorder.add_action(action_pos, action_quat, gripper_flag)
                    recorder.add_action(pos, quat, gripper_close)
                else:
                    # gripper_flag = (current_input['rightIndex'] > 0)
                    # offset_pos = current_input_pos - current_input_pos
                    # offset_quat = quaternion.quaternion.inverse(current_input_quat) * current_input_quat
                    cur_robot_states, cur_tcp_pos, cur_tcp_quat, cur_gripper_states = get_cur_pose(robot, gripper)
                    recorder.add_action(cur_tcp_pos, cur_tcp_quat, cur_gripper_states.width)

                
                # 更新上一次输入
                last_input = current_input
                frame_cnt += 1
                if frame_cnt % 30 == 0:
                    print(f"Recorded {frame_cnt} frames...")
                
                # 控制循环频率
                # time.sleep(1. / record_frequency)
                
        except KeyboardInterrupt:
            log.info("Teleoperation interrupted by user. Saving trajectory...")
            
    except Exception as e:
        # Print exception error message
        log.error(str(e))
    finally:
        recorder.align_frames()
                
        robot.setMode(mode.NRT_PRIMITIVE_EXECUTION)
        log.info("Executing primitive: Move to record HOME Joints")
        # Send command to robot
        robot.executePrimitive(f"MoveJ(target={home_target})")

        # Wait for reached target
        while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
            time.sleep(0.01)
        
        # 记录机器人状态
        cur_robot_states, cur_tcp_pos, cur_tcp_quat, cur_gripper_states = get_cur_pose(robot, gripper)
        recorder.add_state(cur_robot_states, cur_gripper_states)

        # add action
        recorder.add_action(cur_tcp_pos, cur_tcp_quat, cur_gripper_states.width)
        # 确保轨迹被保存
        recorder.save_trajectory(task)
        log.info(f"Trajectory saved to {recorder.output_file}")


if __name__ == "__main__":
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(description="Teleoperation with trajectory recording")
    parser.add_argument("--scene", type=int, default=0, help="Scene number")
    parser.add_argument("--path", type=str, default="/hdd1/flexiv_teleop_dataset/flexiv_insert_USB_raw", help="Path to save recordings")
    parser.add_argument("--frequency", type=int, default=30, help="Record frequency")
    
    args = parser.parse_args()
    
    task_name = "Pick up the USB drive and plug it into the port."
    HOME_TARGET = "-4.77 11.06 2.92 113.00 9.84 18.09 -8.46"

    main(task_name, args.scene, args.path, args.frequency, home_target=HOME_TARGET)