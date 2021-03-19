#!/usr/bin/env python3

# Learning and Intelligent Systems Project
# ---
# Domain randomization in the sim scene. Mainly used to generate training data.
#
# Marius Montebaur, WS20/21

import sys
sys.path.append("../third_party/rai/rai/ry")
import libry as ry
import numpy as np
from random import random, uniform
from math import radians
from common.utility import euler_to_quaternion


def random_stick_pos_in_gripper_local(C, gripper_frame_name="gripper_R"):
    """
    Will place the stick in the gripper's hand
    NOTE: *Assumes the stick to have the gripper as its parent frame*
    """
    stick = C.frame('stick')

    len_stick = np.random.rand(1) * 0.5 + 0.2
    x_offset = uniform(0, len_stick/3)
    stick.setRelativePosition([0, x_offset, 0])
    stick.setRelativeQuaternion(euler_to_quaternion(0, 0, radians(90)))

    keypoint_frames = ["stick_p1", "stick_p2"]
    keypoint_offsets = [len_stick / 2 - 0.0125, -len_stick / 2 + 0.0125]
    for i in range(len(keypoint_frames)):
        kpn = keypoint_frames[i]
        kp_offset = keypoint_offsets[i]
        C.frame(kpn).setRelativePosition([kp_offset, 0, 0])

    stick.setShape(ry.ST.ssBox, [len_stick, 0.045, 0.045, 0.015])

def random_stick_pos_in_gripper_global(C, gripper_frame_name="gripper_R"):
    """
    Will place the stick in the gripper's hand
    NOTE: *Assumes the stick to have the world as its parent frame*
    """
    stick = C.frame('stick')

    tmp_stick_name = "tmp_stick"
    while tmp_stick_name in C.getFrameNames():
        C.delFrame(tmp_stick_name)

    tmp_stick = C.addFrame(tmp_stick_name, parent=gripper_frame_name)

    def ri(a, b): return (b-a)*random() + a

    len_stick = np.random.rand(1) * 0.5 + 0.2
    x_offset = ri(0, len_stick/3)
    tmp_stick.setRelativePosition([0, x_offset, 0])
    tmp_stick.setRelativeQuaternion(euler_to_quaternion(0, 0, radians(90)))

    keypoint_frames = ["stick_p1", "stick_p2"]
    keypoint_offsets = [len_stick / 2 - 0.0125, -len_stick / 2 + 0.0125]
    for i in range(len(keypoint_frames)):
        kpn = keypoint_frames[i]
        kp_offset = keypoint_offsets[i]
        kpf = C.frame(kpn)
        kpf.setRelativePosition([kp_offset, 0, 0])

        # tmp_kpn = "tmp_" + kpn
        # while tmp_kpn in C.getFrameNames():
        #     C.delFrame(tmp_kpn)

        # tmp_kpf = C.addFrame(tmp_kpn, parent=tmp_stick_name)
        # tmp_kpf.setRelativePosition([kp_offset, 0, 0])
        # kpf.setPosition(tmp_kpf.getPosition())
        # kpf.setQuaternion(tmp_kpf.getQuaternion())
        # C.delFrame(tmp_kpn)

    stick.setPosition(tmp_stick.getPosition())
    stick.setQuaternion(tmp_stick.getQuaternion())
    stick.setShape(ry.ST.ssBox, [len_stick, 0.045, 0.045, 0.015])
    C.delFrame(tmp_stick_name)


def set_random_gripper_goal(C, goal_frame_name="gripper_goal", vis=False, z_offset=0, z_interval=(0.05, 0.3)):
    """
    Will set a random position in front of the right robot arm. Komo can
    then be used to plan a path or an IK joint state.
    """

    orbit_home = [0.3, .12, .01]  # in table frame

    while goal_frame_name in C.getFrameNames():
        C.delFrame(goal_frame_name)

    goal_frame = C.addFrame(goal_frame_name, parent="table")

    def ri(a, b): return (b-a)*random() + a

    # goal_pos = [
    #     orbit_home[0] + ri(-0.4, 0.3),
    #     orbit_home[1] + ri(-0.4, 0.3),
    #     orbit_home[2] + ri(*z_interval) + z_offset
    # ]
    # for evaluation use predefined location:
    goal_pos = [
        orbit_home[0] - 0.25,
        orbit_home[1] - 0.1,
        orbit_home[2] + 0.15 + z_offset
    ]

    goal_frame.setRelativePosition(goal_pos)

    if vis:
        goal_frame.setShape(ry.ST.sphere, [0.05])
    else:
        # very small shape. but KOMO needs the frames to have a shape
        goal_frame.setShape(ry.ST.sphere, [0.0005])

    return goal_frame_name

def randomize_stick_length(C, greater_stick_range=False, randomize_diameter=False, given_length=None):
    """
    Randomizes the length of the stick. Also moves the enpoints accordingly.
    :param greater_stick_range: if True, the random interval used for the stick's length will be bigger.
    """
    stick = C.frame('stick')
    stick_p1 = C.frame('stick_p1')
    stick_p2 = C.frame('stick_p2')

    stick_diameter = random() / 5 + 0.02 if randomize_diameter else 0.045

    if given_length:
        len_stick = given_length
    else:
        if greater_stick_range:
            len_stick = np.random.rand(1) * 0.8 + 0.15
        else:
            len_stick = np.random.rand(1) * 0.3 + 0.3
            len_stick = 0.5
    stick.setShape(ry.ST.ssBox, [len_stick, stick_diameter, stick_diameter, 0.015])
    stick_p1.setRelativePosition([len_stick / 2 - 0.0125, 0, 0])
    stick_p2.setRelativePosition([-len_stick / 2 + 0.0125, 0, 0])

def randomize_stick_rotation(C, given_rot=None):
    stick = C.frame("stick")
    # stick.setRelativeQuaternion(euler_to_quaternion(0, 0, 90))
    if given_rot:
        rot = given_rot
    else:
        rot = uniform(-35, 35)
    stick.setRelativeQuaternion(euler_to_quaternion(radians(rot), 0, radians(90)))
    stick.setRelativeQuaternion(euler_to_quaternion(0, radians(rot), radians(90)))


def randomize_stick_pos(C):
    """ Randomizes the stick's position on the table. """
    table = C.frame('table')
    pos_table = table.getPosition()
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

def random_rgb():
    """ Returns a random RGB value as a list with each element in [0, 1]. """
    return [random() for _ in range(3)]

def set_stick_color(C, color, stick_frames=["stick", "stick_p1", "stick_p2"]):
    """ Assigns a new color to the stick. RGB values in [0, 1] each. """
    s, p1, p2 = [C.frame(n) for n in stick_frames]
    s.setColor(color)
    p1.setColor(color)
    p2.setColor(color)

def set_table_color(C, color):
    """ Assigns a new color to the table. RGB values in [0, 1] each. """
    table = C.frame('table')
    table.setColor(color)