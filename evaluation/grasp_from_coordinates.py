#!/usr/bin/env python3

# Learning and Intelligent Systems Project
# ---
# Will place a stick on the table, infer its position using the NN and reach out for the stick.
#
# Marius Montebaur, WS20/21

import sys
sys.path.append("../third_party/rai/rai/ry")
import libry as ry
import numpy as np
import time
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
from math import atan2, degrees
from common.data_extraction import calc_bounding_box
from common.utility import euler_to_quaternion
from common.robot_movement import plan_komo_path, execute_komo_path

# not available anymore
from contact_ros_service import request_prediction


C = ry.Config()
C.addFile("single_world_stick_global.g")
Viewer = C.view()

def setup_stick():
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

    goal_frame.setRelativeQuaternion(euler_to_quaternion(0, 0, rotation_offset))



setup_stick()

cf = C.frame("camera")
cf.setPosition([0.5, 0.5, 1.4])
cf.setRotationRad(*[ 0, 0, 0.7071068, 0.7071068 ])

camera = C.cameraView()
camera.addSensorFromFrame('camera')
camera.selectSensor('camera')

camera.updateConfig(C)
rgb, depth = camera.computeImageAndDepth(False)
depth_min = np.min(depth)
depth_max = np.max(depth)
while depth_min + 0.001 > depth_max:
    print("faulty depth image")
    depth = camera.extractDepth()
    depth_min = np.min(depth)
    depth_max = np.max(depth)

depth_mm = depth.copy()
print("max in depth_mm", np.max(depth_mm))
depth_mm[depth_mm < 0] = 2.0  # camera's max distance
depth_mm = depth_mm * 1000
depth_mm = depth_mm.astype(np.uint16)

bboxes = calc_bounding_box(camera)

visualize_bboxes = False
if visualize_bboxes:
    x, y = zip(*bboxes)
    x1, x2, y1, y2 = *x, *y

    plt.imshow(rgb)
    plt.scatter(x, y)
    plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1])
    plt.show()

resp = request_prediction(rgb, depth_mm, bboxes)

print("--- Received response!")

keypoints_zs = [kp.z for kp in resp.keypoints_camera_frame]
keypoint_frame_names = []
for i in range(resp.num_keypoints):
    kp = resp.keypoints_camera_frame[i]
    # pos = [-kp.x, -kp.y, -kp.z]
    # pos = [3*kp.x, -3*kp.y, -0.6]
    # pos = [3*kp.x, -3*kp.y, -min(keypoints_zs) * 0.95]
    # pos = [3*kp.x, -3*kp.y, -kp.z]
    pos = [-kp.x, -kp.y, kp.z]
    print("keypoint", kp.x, kp.y, kp.z)

    kp_frame_name = f"kp_{i}"
    kp_frame = C.addFrame(kp_frame_name, parent="camera")
    keypoint_frame_names.append(kp_frame_name)
    kp_frame.setShape(ry.ST.sphere, [.05])
    kp_frame.setColor([.5,1,1])
    kp_frame.setRelativePosition(pos)

print("camera rot", C.frame("camera").getQuaternion())
C.frame("camera").setQuaternion([0, 0, 0, 1])

create_goal_frame(C, keypoint_frame_names)


visualize_pred = True
if visualize_pred:
    print("--- Plotting xy")
    fig, (ax1, ax2) = plt.subplots(1, 2)
    # fig, ax1 = plt.subplots(1, 1)
    ax2.imshow(depth)
    ax1.imshow(rgb)

    # bboxes
    x, y = zip(*bboxes)
    x1, x2, y1, y2 = *x, *y
    ax1.scatter(x, y)
    ax1.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1])

    # predictions (x, y)
    x_pred, y_pred = [], []
    for i in range(resp.num_keypoints):
        kp_xy_d = resp.keypoints_xy_depth[i]

        x_pred.append(kp_xy_d.x)
        y_pred.append(kp_xy_d.y)
        print("depth:", kp_xy_d.z)

    ax1.scatter(x_pred, y_pred)
    plt.show()


use_marius_komo = True
if use_marius_komo:
    print("--- Start KOMO")
    X0 = C.getFrameState()
    komo = plan_komo_path(C)
    execute_komo_path(C, komo)

camera = 0
camera = C.cameraView()
camera.addSensorFromFrame('camera')
camera.selectSensor('camera')

camera.updateConfig(C)

while True:
    time.sleep(7)
