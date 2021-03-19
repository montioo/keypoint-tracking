#!/usr/bin/env python3

# Learning and Intelligent Systems Project
# Keeping track of the stick during manipulation task
#
# Marius Montebaur, WS20/21

import sys
sys.path.append("../third_party/rai/rai/ry")
import libry as ry
import numpy as np
from random import random
from math import radians
from common.utility import euler_to_quaternion
import time
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
# from KOMOProblem import *
from math import atan2, degrees
from keypoint_inference import MankeyKeypointDetection, RegionOfInterest


def build_goal_frame_from_prediction(C, prediction):
    keypoint_frame_names = []
    for i in range(prediction.num_keypoints):
        kp = prediction.keypoints_camera_frame[i]
        pos = [-kp.x, kp.y, -kp.z]

        kp_frame_name = f"kp_{i}"
        if kp_frame_name in C.getFrameNames():
            kp_frame = C.frame(kp_frame_name)
        else:
            kp_frame = C.addFrame(kp_frame_name, parent="camera")
        keypoint_frame_names.append(kp_frame_name)
        kp_frame.setShape(ry.ST.sphere, [.05])
        kp_frame.setColor([.5,1,1])
        kp_frame.setRelativePosition(pos)

    create_goal_frame(C, keypoint_frame_names)

def setup_stick(C):
    table = C.frame('table')
    pos_table = table.getPosition()
    size_table = table.getSize()[:2] - 0.2
    size_table = [1., 1.]

    stick = C.frame('stick')
    stick_p1 = C.frame('stick_p1')
    stick_p2 = C.frame('stick_p2')
    xy_stick = (np.random.rand(2) - 0.5) * size_table * np.array([0.7, 0.35]) + pos_table[:2]
    xy_stick[0] += .5
    stick.setPosition(np.hstack([xy_stick, pos_table[-1] + 0.036]))
    phi_stick = 0.95*(np.random.rand(1) - 0.5) * np.pi
    stick.setRotationRad(phi_stick, 0.0, 0.0, 1.0)
    len_stick = np.random.rand(1) * 0.3 + 0.3
    stick.setShape(ry.ST.ssBox, [len_stick, 0.045, 0.045, 0.015])
    stick_p1.setRelativePosition([len_stick / 2 - 0.0125, 0, 0])
    stick_p2.setRelativePosition([-len_stick / 2 + 0.0125, 0, 0])
    print(stick.getPosition())


def attach_stick_to_EE(C, EE_frame_name="endeffR"):
    stick = C.addFrame('EE_stick', parent=EE_frame_name)
    stick_p1 = C.addFrame('EE_stick_p1', parent="EE_stick")
    stick_p2 = C.addFrame('EE_stick_p2', parent="EE_stick")

    # stick.setRotationRad(*euler_to_quaternion(0, 0, 0))
    stick.setRelativeQuaternion(euler_to_quaternion(0, 0, radians(90)))
    len_stick = np.random.rand(1) * 0.3 + 0.3
    stick.setShape(ry.ST.ssBox, [len_stick, 0.045, 0.045, 0.015])
    stick_p1.setRelativePosition([len_stick / 2 - 0.0125, 0, 0])
    stick_p2.setRelativePosition([-len_stick / 2 + 0.0125, 0, 0])


def create_goal_frame(C, keypoint_frame_names, goal_frame_name="goal"):
    center_pos = np.zeros(3)
    rot = np.zeros(2)
    for keypoint_frame_name in keypoint_frame_names:
        keypoint_frame = C.frame(keypoint_frame_name)
        center_pos += np.array(keypoint_frame.getPosition())
    center_pos /= len(keypoint_frame_names)
    if goal_frame_name in C.getFrameNames():
        goal_frame = C.frame(goal_frame_name)
    else:
        goal_frame = C.addFrame(goal_frame_name, parent="world")

    goal_frame.setShape(ry.ST.sphere, [.05])
    # goal_frame.setShape(ry.ST.cylinder, [0.2, 0.02])
    goal_frame.setColor([1,0.5,1])
    goal_frame.setPosition(center_pos)

    def get_x_y_pos(C, frame_name):
        f = C.frame(frame_name)
        return f.getPosition()[:2]
    kp1_x, kp1_y = get_x_y_pos(C, keypoint_frame_names[0])
    kp2_x, kp2_y = get_x_y_pos(C, keypoint_frame_names[1])
    rotation_offset = degrees( atan2(kp1_y - kp2_y, kp1_x - kp2_x) )
    print("rotation_offset", rotation_offset)

    goal_frame.setRelativeQuaternion(euler_to_quaternion(0, 0, radians(rotation_offset)))
