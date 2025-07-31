#!/usr/bin/env python

"""basics3_primitive_execution.py

This tutorial executes several basic robot primitives (unit skills). For detailed documentation
on all available primitives, please see [Flexiv Primitives](https://www.flexiv.com/primitives/).
"""

__copyright__ = "Copyright (C) 2016-2021 Flexiv Ltd. All Rights Reserved."
__author__ = "Flexiv"

import json
import pdb
import time
import argparse

# Utility methods
from utility import quat2eulerZYX
from utility import parse_pt_states
from utility import list2str

# Import Flexiv RDK Python library
# fmt: off
import sys
sys.path.insert(0, "../lib_py")
import flexivrdk
import quaternion
# from realsence_function import real_scene_435_module
# fmt: on
import numpy as np
import os
from quest_receive import quest_teleop

def print_description():
    """
    Print tutorial description.
    """
    print("This tutorial executes several basic robot primitives (unit skills). For "
          "detailed documentation on all available primitives, please see [Flexiv "
          "Primitives](https://www.flexiv.com/primitives/).")
    print()


quest_teleop = quest_teleop()

def main(scene_num=0, target="0.50 -0.0 0.35 180 0 180", file_save_path="./cali_image_and_pose/"):

    # Program Setup
    # ==============================================================================================
    # Parse arguments
    # Define alias
    log = flexivrdk.Log()
    mode = flexivrdk.Mode
    # Print description
    log.info("Tutorial description:")
    print_description()
    # camera = real_scene_435_module()
    file_save_path = file_save_path + str(scene_num) + "/"
    os.makedirs(file_save_path, exist_ok=True)
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

        # log.info("Opening gripper")
        gripper.move(0.1, 0.1, 20)
        time.sleep(2)

        # robot.executePlan("PLAN-Home")
        time.sleep(2)
        # # Execute Primitives
        # # ==========================================================================================
        # Switch to primitive execution mode
        robot.setMode(mode.NRT_PRIMITIVE_EXECUTION)
        # robot.executePrimitive("MoveL(target={} WORLD WORLD_ORIGIN, maxVel=0.1)".format(target))
        # while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
        #     time.sleep(1)

        # Move robot joints to target positions
        # ------------------------------------------------------------------------------------------
        # The required parameter <target> takes in 7 target joint positions. Unit: degrees
        log.info("Executing primitive: MoveJ")

        # Send command to robot
        robot.executePrimitive("MoveJ(target=-4.77 11.06 2.92 113.00 9.84 18.09 -8.46)")

        # Wait for reached target
        while parse_pt_states(robot.getPrimitiveStates(), "reachedTarget") != "1":
            time.sleep(1)

        robot.setMode(mode.NRT_CARTESIAN_MOTION_FORCE)

        # quest_inputs = []
        last_input = None
        while True:
            robot_states = flexivrdk.RobotStates()
            robot.getRobotStates(robot_states)
            current_tcp_pose = robot_states.tcpPose
            
            joint_states = robot_states.q
            quest_teleop.joint_states = np.array(joint_states)

            current_input, _, _ = quest_teleop.get_input_frame()
            if current_input is None:
                continue
            if last_input is None:
                last_input = current_input

            current_tcp_pos = np.array([current_tcp_pose[0], current_tcp_pose[1], current_tcp_pose[2]])
            current_tcp_quat = quaternion.quaternion(current_tcp_pose[3], current_tcp_pose[4], current_tcp_pose[5], current_tcp_pose[6])
            current_input_pos = np.array([current_input['rightPos']["z"], -current_input['rightPos']["x"], current_input['rightPos']["y"]])
            current_input_quat = quaternion.quaternion(current_input['rightRot']["w"], current_input['rightRot']["z"], current_input['rightRot']["x"], current_input['rightRot']["y"])
            
            if current_input['rightHand'] > 0.5:

                if last_input['rightHand'] <= 0.5:
                    start_tcp_pos = current_tcp_pos
                    start_tcp_quat = current_tcp_quat
                    start_input_pos = current_input_pos
                    start_input_quat = current_input_quat

                # quest_inputs.append(current_input)

                offset_pos = current_input_pos - start_input_pos
                offset_quat = quaternion.quaternion.inverse(start_input_quat) * current_input_quat

                pos = start_tcp_pos + offset_pos
                quat = start_tcp_quat * offset_quat
           
                target_wrench = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                new_target = [pos[0], pos[1], pos[2], quat.w, quat.x, quat.y, quat.z]
                robot.sendCartesianMotionForce(new_target, target_wrench)

                # right trigger: current_input['rightIndex'] in [0, 1]
                # gripper distance in [0.01, 0.1]
                gripper_close = 0.09 * (1 - current_input['rightIndex']) + 0.01
                gripper.move(gripper_close, 0.1, 20)
                # gripper_states = flexivrdk.GripperStates()
                # gripper.getGripperStates(gripper_states)
                # print("gripper force", gripper_states.force)

            last_input = current_input
            time.sleep(0.02)

            # # 保存到文件
            # with open("quest_input.json", "w", encoding="utf-8") as f:
            #     json.dump(quest_inputs, f, ensure_ascii=False, indent=4)

    except Exception as e:
        # Print exception error message
        log.error(str(e))

if __name__ == "__main__":
    
    scene_num = 1

    Joint_Back_State = [-0.08326569199562073, 0.19320690631866455, 0.05103691667318344, 1.9722830057144165, 0.17173852026462555, 0.31569066643714905, -0.14763186872005463]

    for i in range(7):
            Joint_Back_State[i] = Joint_Back_State[i] / np.pi * 180


    target = "-4.77 11.06 2.92 113.00 9.84 18.09 -8.46"
    file_save_path = "./test/"
    main(scene_num, target, file_save_path=file_save_path)
