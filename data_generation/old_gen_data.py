#!/usr/bin/env python3

# Learning and Intelligent Systems Project
# ---
# Data Generation
#
# Marius Montebaur, WS20/21

# TODO: Organize this file into multiple files or modules. Current state is terrible.
# TODO: Why is the table not always visible in the camera images?


import sys
sys.path.append('../../rai/rai/ry')
import libry as ry
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import cv2
import time
from random import random
from math import pi, sin, cos, tan, radians, degrees
# don't use pil AND opencv
from PIL import Image
from datetime import datetime
import os
import yaml

from utility import euler_to_quaternion
from data_extraction import *
from domain_randomization import *


# folder setup for training data
folder = "/home/monti/Desktop/pdc/logs_proto/{}/processed".format(datetime.now().strftime("%Y-%m-%d-%H-%M"))
image_folder = os.path.join(folder, "images")
image_mask_folder = os.path.join(folder, "image_masks")

os.makedirs(folder)
os.makedirs(image_folder)
os.makedirs(image_mask_folder)


def set_random_goal_pose(C):
    """Sets up frames that are used as goals to record training data."""
    for frame_name in C.getFrameNames():
        if frame_name.startswith("dg_"):
            C.delFrame(frame_name)

    # see notes page 8 for description
    A_radius = 0.15
    l_interv = [0.6, 0.8]
    θ_interv = [0.0, radians(60)]  # deg
    ϕ_interv = [pi, 2*pi]

    orbit_home = [0.3, .12, .01]  # in table frame
    orbit_area = C.addFrame(name="dg_orbit_area", parent="table")
    orbit_area.setRelativePosition(orbit_home)

    # visualization of the current camera orientation
    # orbit_area.setShape(ry.ST.cylinder, [0.002, A_radius*2])
    # orbit_area.setColor([0, 0, 1])
    # A_frame = C.addFrame(name="dg_A", parent="dg_orbit_area")
    # A_frame.setShape(ry.ST.cylinder, [0.002, A_radius*2])
    # A_frame.setColor([0, 0, 1])
    # A_frame.setRelativePosition([0, 0, 0])

    r_c, α_c = A_radius*random(), 1*pi*random()
    x_c, y_c = r_c*cos(α_c), -r_c*sin(α_c)

    orbit_center = C.addFrame(name="dg_orbit_center", parent="dg_orbit_area")
    orbit_center.setRelativePosition([x_c, y_c, 0.0])
    # orbit_center.setShape(ry.ST.sphere, [0.02])
    # orbit_center.setColor([0, 1, 1])

    # random_interval
    def ri(a, b): return (b-a)*random() + a

    r, θ, ϕ = ri(*l_interv), ri(*θ_interv), ri(*ϕ_interv)
    x, y, z = r*sin(θ)*cos(ϕ), r*sin(θ)*sin(ϕ), r*cos(θ)

    # print(x, y, z)
    cam_goal = C.addFrame("dg_cam_goal", parent="dg_orbit_center")
    cam_goal.setRelativePosition([x, y, z])
    cam_goal.setRelativeQuaternion(euler_to_quaternion(0, θ, ϕ))
    # cam_goal.setShape(ry.ST.cylinder, [0.9, 0.02])
    # cam_goal.setColor([1, 1, 1])


def update_goal_frame(C):
    """Builds the frame that is used as a reference for KOMO."""
    C.delFrame("goal")
    goal = C.addFrame("goal", parent="dg_cam_goal")
    # TODO: Remove goal shape
    goal.setShape(ry.ST.sphere, [.0005])
    # goal.setShape(ry.ST.cylinder, [0.15, .04])
    goal.setColor([.5,1,1])
    goal.setRelativeQuaternion([.0, 1., .0, .0])


def build_new_path(C, steps_per_phase):
    # feature table: https://github.com/MarcToussaint/robotics-course/blob/master/tutorials/2-features.ipynb
    # explanation: https://github.com/MarcToussaint/robotics-course/blob/master/tutorials/3-IK-optimization.ipynb

    X0 = C.getFrameState()

    goal_frame = "goal"
    arm = "endeffR"  # "R_panda_link0"
    gripper = "R_gripper"

    # phases: float, stepsPerPhase: int = 20, timePerPhase: float = 5.0, useSwift: bool
    komo = C.komo_path(1., steps_per_phase, 10, False)
    komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq)
    komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq)
    komo.addObjective([1.0], ry.FS.distance, [arm, goal_frame], ry.OT.eq, [1e2])
    komo.addObjective([0.9, 1.0], ry.FS.quaternionDiff, [gripper, goal_frame], ry.OT.sos, scale=[4]*4)  # , [1e0])
    komo.addObjective([0.9, 1.0], ry.FS.positionDiff, [gripper, goal_frame], ry.OT.sos)  # , [1e0])
    komo.addObjective(time=[1.], feature=ry.FS.qItself, type=ry.OT.eq, order=1)

    komo.optimize()
    # komo.getReport()
    return komo

def build_goal_config(C):
    goal_frame = "goal"
    arm = "endeffR"  # "R_panda_link0"
    gripper = "R_gripper"

    IK = C.komo_IK(False)
    # IK.addObjective(type=ry.OT.eq, feature=ry.FS.positionDiff, frames=[gripper, goal_frame])
    # IK.addObjective(type=ry.OT.eq, feature=ry.FS.poseDiff, frames=[gripper, goal_frame])
    IK.addObjective(type=ry.OT.sos, feature=ry.FS.poseDiff, frames=[gripper, goal_frame])
    # IK.addObjective(type=ry.OT.eq, feature=ry.FS.positionDiff, frames=[gripper, goal_frame])
    # IK.addObjective(ry.FS.quaternionDiff, [gripper, goal_frame], ry.OT.sos, scale=[4]*4)  # , [1e0])
    # IK.addObjective(ry.FS.positionDiff, [gripper, goal_frame], ry.OT.sos)  # , [1e0])
    IK.optimize()
    return IK


def store_img(filename, img_np):
    img = Image.fromarray(img_np)
    img.save(filename)


def save_dataframe(C, camera, sample_id):
    global folder, image_folder, image_mask_folder
    filename = os.path.join(folder, "stick_2_keypoint_image.yaml")

    kp_pix_xyd = keypoint_pixel_xy_depth(C, camera)
    if type(kp_pix_xyd) is bool or len(kp_pix_xyd) != 2:
        print("Stick not fully visible in sample:", sample_id)
        return False

    # input("pre bbox")
    bbr_xy, btl_xy = calc_bounding_box(camera)
    # input("bbox")
    rgb, depth = gen_images(camera)
    # input("rgbd")
    mask = masked_image(camera)
    # input("mask")

    # depth image needs scaling. "pixel" values are in meters
    # TODO: somehow access the camera's parameterization for max z range
    depth[depth == -1] = 2.0
    depth *= 255/2
    depth = depth.astype("uint8")

    rgb_image_filename = f"{sample_id:06}_rgb.png"
    store_img(os.path.join(image_folder, rgb_image_filename), rgb)

    depth_image_filename = f"{sample_id:06}_depth.png"
    store_img(os.path.join(image_folder, depth_image_filename), depth)

    masked_image_filename = f"{sample_id:06}_mask.png"
    store_img(os.path.join(image_mask_folder, masked_image_filename), mask)

    threed_keypoints_in_cam_frame = threed_keypoint_camera_frame(C)
    # input("3d keypoints in cam frame")

    cam_to_world_T = camera_to_world(C)
    # input("cam to world T")

    d = {sample_id: {
        "3d_keypoint_camera_frame": threed_keypoints_in_cam_frame,
        "bbox_bottom_right_xy": bbr_xy,
        "bbox_top_left_xy": btl_xy,
        "camera_to_world": cam_to_world_T,
        "depth_image_filename": depth_image_filename,
        "keypoint_pixel_xy_depth": kp_pix_xyd,
        "rgb_image_filename": rgb_image_filename,
        "timestamp": str(time.time()).replace('.', '')
    }}

    print("sample description:", d)

    try:
        with open(filename,'r') as yamlfile:
            cur_yaml = yaml.load(yamlfile)
            if cur_yaml is None:
                cur_yaml = d
            else:
                cur_yaml.update(d)
    except FileNotFoundError:
        cur_yaml = d

    with open(filename,'w') as yamlfile:
        # yaml.safe_dump(cur_yaml, yamlfile)
        yaml.dump(cur_yaml, yamlfile)

    print("Wrote to file for sample:", sample_id)
    return True


def main():
    ###
    # Setup
    max_datapoints = 100

    C = ry.Config()
    C.addFile("world.g")
    Viewer = C.view()

    camera = C.cameraView()
    camera.addSensorFromFrame('camera')
    camera.selectSensor('camera')

    ###
    # Movement
    if "goal" not in C.getFrameNames():
        goal = C.addFrame("goal")
    goal.setShape(ry.ST.sphere, [.05])
    goal.setColor([.5,1,1])
    goal.setPosition([0.7, 0.3, 1])

    # for i in range(max_datapoints):
    i = 0
    while i < max_datapoints:

        set_stick_color(C, random_rgb())
        set_table_color(C, random_rgb())

        randomize_stick_pos(C)
        set_random_goal_pose(C)

        update_goal_frame(C)

        for frame_name in C.getFrameNames():
            if frame_name.startswith("dg_"):
                C.delFrame(frame_name)

        # steps, duration = 30, 1
        # step_duration = duration / steps
        steps_per_phase = 40
        # path planning
        # komo = build_new_path(C, steps_per_phase)
        # IK:
        IK = build_goal_config(C)

        # C.setFrameState( X0 )
        print("--- Done Planing")

        # for i in range(steps_per_phase):
        #     step = komo.getConfiguration(i)
        #     C.setFrameState(step)
        #     time.sleep(0.1)
        C.setFrameState( IK.getConfiguration(0) )

        print("--- Done Moving")

        # input("pre update cam config")
        camera.updateConfig(C)
        # input("post update cam config")
        print("--- Done Updating Camera")

        if save_dataframe(C, camera, i):
            # Data is randomly generated and sometimes invalid
            i += 1

        print("--- Done Showing Images")
        # input()

    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
