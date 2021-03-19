#!/usr/bin/env python3

# Learning and Intelligent Systems Project
# ---
# Will keep pointing one end of the stick to a goal frame while the length of
#   the stick keeps changing.
#
# Marius Montebaur, WS20/21

import sys
sys.path.append("../third_party/rai/rai/ry")
import libry as ry
from math import radians
import time
from common.utility import euler_to_quaternion, delFrameIfExists
from common.robot_movement import plan_komo_path
from common.data_extraction import calc_bounding_box, gen_images_adjust_depth
from common.domain_randomization import random_rgb, randomize_stick_length, set_random_gripper_goal, set_stick_color, randomize_stick_rotation
from common.visualization import visualize_keypoints_in_rgb

# TODO: No please there has to be a cleaner way.
sys.path.append("../")
from keypoint_inference import MankeyKeypointDetection, RegionOfInterest
from PIL import Image


image_id = 0


def save_trajectory_image(C, video_cam):
    global image_id
    video_output_dir = "/home/monti/Desktop/pdc/stick_pointer_demo"
    filename = f"{video_output_dir}/{image_id:04}.png"
    video_cam.updateConfig(C)
    np_img, _ = video_cam.computeImageAndDepth(True)
    pil_img = Image.fromarray(np_img)
    pil_img.save(filename)
    image_id += 1

def hide_goal_frame(hide, C, goal_frame_name):
    """
    Hide the goal frame by decreasing the diameter of its shape. We want to
    have the goal frame visible in the simulation viewer but not in the
    camera's images.
    """
    goal_frame = C.frame(goal_frame_name)
    if hide:
        goal_frame.setShape(ry.ST.sphere, [0.0001])
    else:
        goal_frame.setShape(ry.ST.sphere, [0.05])


def compute_on_frame(C, camera, keypoint_inference, visualize_pred):

    start_time = time.time()
    rgb, depth_mm = gen_images_adjust_depth(C, camera, "camera")
    print("rgb and depth generation took:", time.time() - start_time)

    start_time = time.time()
    gt_bboxes = calc_bounding_box(camera, object_frame="stick")
    print("bbox ground truth generation took:", time.time() - start_time)
    print("bbox truth", gt_bboxes)

    bboxes = None  # bbox_web(rgb)
    print("bboxes pred", bboxes)

    # use ground truth if detection server is not available
    if not bboxes:
        bboxes = gt_bboxes

    roi = RegionOfInterest.build_from_bbox_list(bboxes)
    resp = keypoint_inference.handle_keypoint_request(rgb.copy(), depth_mm, roi)

    print("--- Received response!")

    if visualize_pred:
        visualize_keypoints_in_rgb(rgb, bboxes, resp, depth_mm)

    return resp


def _0_setup(C, gripper_goal_name):
    set_random_gripper_goal(C, gripper_goal_name, vis=True, z_offset=0.2)


def _1_estimate_keypoint_position(C, camera, keypoint_inference, gripper_goal_name, visualize_pred=False):
    """
    Will estimate the keypoint position in camera frame and return the 3D coordinates.
    """
    hide_goal_frame(True, C, gripper_goal_name)
    camera.updateConfig(C)
    response = compute_on_frame(C, camera, keypoint_inference, visualize_pred)
    hide_goal_frame(False, C, gripper_goal_name)
    return response.keypoints_camera_frame


def _2_set_estimated_keypoint_frame(C, positions_in_cam_frame, color=None):
    """
    Translates keypoint positions from position in camera frame to position
    in robot's gripper frame. Will also attach a frame with a shape to the
    gripper frame to have the predicted keypoint as a 3D object relative to
    the gripper.
    """
    tmp_frame_name = "tmp"  # Temporary frame to translate coordinates
    keypoints_in_gripper_frame_name = []

    for i, pos in enumerate(positions_in_cam_frame):
        # TODO: Fix sign bug
        pos_cam_frame = (-pos[0], pos[1], -pos[2])

        delFrameIfExists(C, tmp_frame_name)

        tmp_frame = C.addFrame(tmp_frame_name, parent="camera")
        tmp_frame.setRelativePosition(pos_cam_frame)
        world_pos = tmp_frame.getPosition()
        C.delFrame(tmp_frame_name)

        kp_gripper_frame_name = f"kp_pred_{i}"
        delFrameIfExists(C, kp_gripper_frame_name)
        kp_gripper_frame = C.addFrame(kp_gripper_frame_name, parent="endeffR")
        kp_gripper_frame.setPosition(world_pos)

        kp_gripper_frame.setShape(ry.ST.sphere, [0.02])
        color = color if color else [1., 1., 1.]
        kp_gripper_frame.setColor(color)

        keypoints_in_gripper_frame_name.append(kp_gripper_frame_name)

    return keypoints_in_gripper_frame_name


def _3_move_keypoint_to_goal_pos(C, estimated_keypoint_frame, gripper_goal_name, video_cam=None):
    """
    The predicted keypoint position in the robot gripper's frame is known.
    Now a path is planned to reach a goal location with this keypoint.
    """
    steps_count = 50
    duration = 2.0

    print("--- Start KOMO")

    # will move one end of the stick to the goal frame
    komo = plan_komo_path(C, obj_goal=gripper_goal_name, steps=steps_count, duration=duration, arm=estimated_keypoint_frame)

    for step_idx in range(0, steps_count):

        step_conf = komo.getConfiguration(step_idx)
        C.setFrameState(step_conf)

        if video_cam:
            save_trajectory_image(C, video_cam)

        time.sleep(0.05)

    print("--- Did KOMO")


def main():
    # keypoint_network_path = "/home/monti/Desktop/pdc/resnet18_2021-01-30/checkpoint_0118.pth"
    keypoint_network_path = "/home/monti/Desktop/pdc/resnet18_2021-02-01/checkpoint_0236.pth"

    keypoint_inference = MankeyKeypointDetection(keypoint_network_path)

    C = ry.Config()
    C.addFile("common/single_world.g")

    # new cam for video capture
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

    viewer = C.view()

    jn = "R_panda_joint6"
    C.setJointState([C.getJointState([jn])[0] + 0.3], [jn])

    # Set stick so that it's in the robot's gripper.
    C.frame("stick").setRelativePosition([0, 0, 0])
    # C.frame("stick").setRelativeQuaternion(euler_to_quaternion(0, 0, radians(90)))

    set_stick_color(C, random_rgb())
    randomize_stick_length(C)
    randomize_stick_rotation(C)

    cf = C.frame("camera")
    camera_front_facing = True
    if camera_front_facing:
        cf.setPosition([0.5, 1.1, 1.7])
        cf.setQuaternion(euler_to_quaternion(radians(-110), radians(180), -radians(0)))
    else:
        # top down view
        cf.setPosition([0.5, 0.5, 1.4])
        cf.setRotationRad(*[ 0, 0, 0.7071068, 0.7071068 ])

    camera = C.cameraView()
    camera.addSensorFromFrame('camera')
    camera.selectSensor('camera')

    gripper_goal_name = "gripper_goal"
    _0_setup(C, gripper_goal_name)

    rots = [35, -10, 20, -35, 0]
    lengths = [0.2, 0.7, 0.25, 0.65, 0.4]

    # Loop: Move stick end to goal location, change stick's length, repeat
    # while True:
    for rot, length in zip(rots, lengths):
        # 1. Estimate keypoint position
        # hide guidance frames before updating cam
        pos_in_cam_frame = _1_estimate_keypoint_position(C, camera, keypoint_inference, gripper_goal_name, visualize_pred=False)

        # 2. See keypoint on arm as attached
        frame_names_in_gripper = _2_set_estimated_keypoint_frame(C, pos_in_cam_frame)

        # 3. Move keypoint to goal location
        _3_move_keypoint_to_goal_pos(C, frame_names_in_gripper[1], gripper_goal_name, video_cam)

        # 4. change stick length
        randomize_stick_length(C, greater_stick_range=True, given_length=length)
        randomize_stick_rotation(C, given_rot=rot)
        # -> repeat

        # remove frames that represent a keypoint's position in the gripper's frame.
        for f_name in frame_names_in_gripper:
            delFrameIfExists(C, f_name)

if __name__ == "__main__":
    main()