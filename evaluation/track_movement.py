#!/usr/bin/env python3

# Learning and Intelligent Systems Project
# ---
# Keeping track of the stick during manipulation task
#
# Marius Montebaur, WS20/21

import sys
sys.path.append("../third_party/rai/rai/ry")
import libry as ry
import numpy as np
from random import random
from math import radians, atan2, degrees
import time
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
from common.utility import euler_to_quaternion, transform_position, dirty_fix_mirror_problem, export_pred_and_truth_json, get_datetime_str, dirty_fix_numpy_json_export
from common.robot_movement import plan_komo_path, execute_komo_path
from common.data_extraction import gen_images_adjust_depth, calc_bounding_box
from common.domain_randomization import *
from common.visualization import visualize_keypoints_in_rgb, visualize_eval_stages
from query_bbox_prediction import socket_client
from common.stats import Stats
from PIL import Image, ImageDraw

from track_movement_utility import *
from keypoint_inference import MankeyKeypointDetection, MankeyKeypointDetectionLSTM, RegionOfInterest


def get_keypoint_ground_truth(C, keypoint_frame_names=["stick_p1", "stick_p2"]):
    res = []
    for frame_name in keypoint_frame_names:
        frame = C.getFrame(frame_name)
        res.append(frame.getPosition().tolist())
    return res


def bbox_web(rgb):
    """ Query bbox inference server. """

    # rgb to bgr
    img_np = rgb.copy()
    img_np[:, :, 0], img_np[:, :, 2] = img_np[:, :, 2], img_np[:, :, 0]
    coord, mode = socket_client(img_np)

    if not len(coord):
        return None

    coord = coord.flatten()
    p = 10  # padding
    max_height, max_width = rgb.shape[0] - 1, rgb.shape[1] - 1
    return [
        [ max(0, coord[0]-p), max(0, coord[1]-p) ],
        [ min(max_width, coord[2]+p), min(max_height, coord[3]+p) ]
    ]

old_bboxes = None

def compute_on_frame(C, camera, keypoint_inference, visualize_pred):
    global old_bboxes

    start_time = time.time()
    rgb, depth_mm = gen_images_adjust_depth(C, camera, "camera", show_only_vision_geom=False)
    print("rgb and depth generation took:", time.time() - start_time)

    start_time = time.time()
    gt_bboxes = calc_bounding_box(camera, object_frame="stick")
    print("bbox ground truth generation took:", time.time() - start_time)
    print("bbox truth", gt_bboxes)

    bboxes = None  # bbox_web(rgb)
    print("bboxes pred", bboxes)

    if not bboxes:
        bboxes = gt_bboxes

    # Use old bounding box coordinates in case something was wrong. This was not included to
    # account for failed bounding box predictions. The predictions are very reliable. This is
    # included because it happened sometimes that the stick would vanish from the sim scene.
    if not bboxes:
        bboxes = old_bboxes
    old_bboxes = bboxes

    roi = RegionOfInterest.build_from_bbox_list(bboxes)
    resp = keypoint_inference.handle_keypoint_request(rgb.copy(), depth_mm, roi)

    print("--- Received response!")

    # build_goal_frame_from_prediction(C, resp)

    if visualize_pred:
        visualize_keypoints_in_rgb(rgb, bboxes, resp, depth_mm, True)
        # visualize_eval_stages(rgb, bboxes, resp, depth_mm, "processing_stages_overview.pdf")

    return resp


def vis_point_list(C, points, frame_prefix, parent_frame=None, color=None):
    """ Add points to the simulation scene that represent these positions. """
    for i, point in enumerate(points):
        frame_name = f"{frame_prefix}_{i:04}"
        pos = [point[0], point[1], point[2]]
        f = C.addFrame(frame_name)
        f.setPosition(pos)
        f.setShape(ry.ST.sphere, [0.02])
        color = color if color else [0., 0., 0.]
        f.setColor(color)

def save_trajectory_image(C, video_cam, img_id, keypoints_xy=None):
    """
    Save images of the robot's movement as .png files to create a video later.
    Command to create a video: ffmpeg -r 6 -i 0%03d.png -c:v libx264 -vf fps=25 -pix_fmt yuv420p out.mp4
    """
    video_output_dir = "/home/monti/Desktop/pdc/lstm_2021-03-02_cbrc/video/std_p1"
    # video_output_dir = "/home/monti/Desktop/pdc/lstm_2021-03-02_cbrc/video/lstm_p1"
    filename = f"{video_output_dir}/{img_id:04}.png"
    video_cam.updateConfig(C)
    np_img, _ = video_cam.computeImageAndDepth(True)
    pil_img = Image.fromarray(np_img)
    if keypoints_xy:
        w, h = pil_img.size
        pil_img_draw = ImageDraw.Draw(pil_img)
        for (x, y) in keypoints_xy:
            d = 4  # circle dia
            bounds = (max(0, x-d), max(0, y-d), min(w-1, x+d), min(h-1, y+d))
            pil_img_draw.ellipse(bounds, fill="red", outline="black")
    pil_img.save(filename)


def plan_path(C, camera, keypoint_inference, sl=None, video_cam=None, place_occlusion_geom=False):
    """
    :param sl: Stats logger
    """

    gripper_goal_name = "gripper_goal"
    steps_count = 50
    duration = 2.0

    set_random_gripper_goal(C, gripper_goal_name, vis=True)

    time.sleep(1)

    print("--- Start KOMO")
    X0 = C.getFrameState()
    print("-------------> X0", len(X0))

    # will move one end of the stick to the goal frame
    komo = plan_komo_path(C, obj_goal=gripper_goal_name, steps=steps_count, duration=duration, arm="stick_p1")

    # [ [keypoint1_cam_frame, keypoint2_cam_frame], ... ]
    keypoints_predicted_world_frame = []
    keypoints_truth_world_frame = []

    vis_interval = 10

    for step_idx in range(0, steps_count):

        step_conf = komo.getConfiguration(step_idx)
        print("------------->", step_idx, len(step_conf))
        print(len(C.getFrameState()))
        C.setFrameState(step_conf)

        # puts an occlusion object into the scene
        occ_cube = C.frame("occ_frame")
        if place_occlusion_geom:
            occ_cube_pos = [0.11534094401646417, 0.6521136637844796, 1.4246536710888913]
            occ_cube.setPosition(occ_cube_pos)
            # occ_cube.setShape(ry.ST.box, [uniform(0.15, 0.3), uniform(0.15, 0.3), uniform(0.15, 0.3)])
            occ_cube.setShape(ry.ST.box, [0.25, 0.01, 0.3])
        else:
            occ_cube.setPosition((0, 0, -2))

        camera.updateConfig(C)

        visualize_pred = False and step_idx % vis_interval == 0
        response = compute_on_frame(C, camera, keypoint_inference, visualize_pred)

        # response holds the keypoint 3D estimation
        # convert keypoint_inference.py:Points to list
        keypoint_positions_cam_frame = [p.tolist() for p in response.keypoints_camera_frame]
        keypoint_positions_cam_frame = dirty_fix_mirror_problem(keypoint_positions_cam_frame)
        keypoint_positions_world_frame = [transform_position(C, p, "camera") for p in keypoint_positions_cam_frame]
        keypoint_positions_world_frame = dirty_fix_numpy_json_export(keypoint_positions_world_frame)
        keypoints_predicted_world_frame.append(keypoint_positions_world_frame)
        keypoints_truth_world_frame.append( get_keypoint_ground_truth(C) )

        # Could use new logger insted
        # if sl:
        #     sl.log_for_key("keypoints_truth_world", keypoints_truth_world_frame)
        #     sl.log_for_key("keypoints_pred_world", keypoints_predicted_world_frame)
        #     sl.new_log_entry()

        if video_cam:
            # keypoints_xy = [(p.x, p.y) for p in response.keypoints_xy_depth]
            # save_trajectory_image(C, video_cam, step_idx, keypoints_xy=keypoints_xy)
            save_trajectory_image(C, video_cam, step_idx)

        time.sleep(0.05)

    print("done")

    def flatten_list(l):
        return [item for sublist in l for item in sublist]

    vis_point_list(C, flatten_list(keypoints_truth_world_frame), "truth", color=[0., 1., 0.])
    vis_point_list(C, flatten_list(keypoints_predicted_world_frame), "pred", color=[1., 0., 0.])

    export_pred_and_truth_json(
        flatten_list(keypoints_truth_world_frame),
        flatten_list(keypoints_predicted_world_frame),
        f"trajectories/point_trajectory_{get_datetime_str(for_filename=True)}.json"
    )

    if video_cam:
        save_trajectory_image(C, video_cam, steps_count)

    while True:
        time.sleep(1)


def main():
    sl = Stats()

    use_lstm = False
    if use_lstm:
        # lstm net
        # keypoint_network_path = "/home/monti/Desktop/pdc/lstm_2021-02-28/checkpoint_0386.pth"
        # keypoint_network_path = "/home/monti/Desktop/pdc/lstm_2021-02-29/checkpoint_0346.pth"
        # keypoint_network_path = "/home/monti/Desktop/pdc/lstm_2021-02-26/checkpoint_0546.pth"
        # keypoint_network_path = "/home/monti/Desktop/pdc/lstm_2021-03-02/checkpoint_0246.pth"
        # keypoint_network_path = "/home/monti/Desktop/pdc/lstm_2021-03-02_cbrc/checkpoint_0296.pth"
        keypoint_network_path = "/home/monti/Desktop/pdc/lstm_2021-03-02_cbrc/checkpoint_0316.pth"
        keypoint_inference = MankeyKeypointDetectionLSTM(keypoint_network_path)
    else:
        # standard net
        # keypoint_network_path = "/home/monti/Desktop/pdc/resnet18_2021-01-30/checkpoint_0118.pth"
        keypoint_network_path = "/home/monti/Desktop/pdc/resnet18_2021-02-01/checkpoint_0236.pth"
        keypoint_inference = MankeyKeypointDetection(keypoint_network_path)


    C = ry.Config()
    C.addFile("common/single_world.g")
    C.addFrame("occ_frame")

    use_side_cam = True
    if use_side_cam:
        # this camera is located at the side of the robot
        video_cam_frame = C.frame("video_camera")
        video_cam_45 = [1.3, 1.3, 1.35, 0.245372, 0.193942, 0.548013, 0.775797]
        video_cam_more_side = [1.3, 1.1, 1.35, 0.401579, 0.298422, 0.537793, 0.67857]
        video_cam_X = video_cam_more_side
        video_cam_frame.setPosition(video_cam_X[:3])
        video_cam_frame.setQuaternion(video_cam_X[3:])
        video_cam = C.cameraView()
        video_cam.addSensorFromFrame("video_camera")
        video_cam.selectSensor("video_camera")
        video_cam.updateConfig(C)
    else:
        video_cam = C.cameraView()
        video_cam.addSensorFromFrame("camera")
        video_cam.selectSensor("camera")
        video_cam.updateConfig(C)

    viewer = C.view()

    jn = "R_panda_joint6"
    C.setJointState([C.getJointState([jn])[0] + 0.3], [jn])

    # Set stick so that it's in the robot's gripper.
    C.frame("stick").setRelativePosition([0, 0, 0])
    C.frame("stick").setRelativeQuaternion(euler_to_quaternion(0, 0, radians(90)))

    # attach_stick_to_EE(C)
    # random_stick_pos_in_gripper(C, "endeffR")
    set_stick_color(C, random_rgb())
    randomize_stick_length(C)
    # setup_stick(C)

    set_stick_color(C, (0.3, 0.9, 0.1))
    randomize_stick_length(C)

    cf = C.frame("camera")
    camera_front_facing = True
    if camera_front_facing:
        # cf.setPosition([0.5, 1.5, 1.7])  # too far. Depth estimation not reliable
        cf.setPosition([0.5, 1.1, 1.7])
        cf.setQuaternion(euler_to_quaternion(radians(-110), radians(180), -radians(0)))
    else:
        # top down view
        cf.setPosition([0.5, 0.5, 1.4])
        cf.setRotationRad(*[ 0, 0, 0.7071068, 0.7071068 ])

    camera = C.cameraView()
    camera.addSensorFromFrame('camera')
    camera.selectSensor('camera')
    camera.updateConfig(C)

    plan_path(C, camera, keypoint_inference, sl, video_cam, use_lstm)


if __name__ == "__main__":
    main()