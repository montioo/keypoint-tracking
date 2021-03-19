#!/usr/bin/env python3

# Learning and Intelligent Systems Project
# ---
# Functions that extract the necessary data from the rai simulation scene. The
# data can represent the ground truth when producing data for a training
# algorithm or can be used to evalute predictions on simulated scenes.
#
# Marius Montebaur, WS20/21

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np


def threed_keypoint_camera_frame(C, camera_frame="camera", point_frames=["stick_p1", "stick_p2"]):
    """ Returns the keypoint's 3D positions in the camera's coordinate frame. """

    keyp_pos_in_cam_frame = []
    for point_frame in point_frames:
        kpf_global = C.frame(point_frame)
        global_pos = kpf_global.getPosition()

        local_frame_name = "cam_tmp"
        if local_frame_name not in C.getFrameNames():
            print(f'Frame "{local_frame_name}" with parent frame "{camera_frame}" is missing.')

        local_frame = C.frame(local_frame_name)
        local_frame.setPosition(global_pos)

        p_in_cam_frame = local_frame.getRelativePosition().tolist()
        p_in_cam_frame[2] *= -1
        keyp_pos_in_cam_frame.append(p_in_cam_frame)

    return keyp_pos_in_cam_frame


def calc_bounding_box(camera, object_frame="stick", visualize=False, return_mask=False):
    """Returns the bouding box edge pixels for an object."""
    mask = camera.extractMask(object_frame)

    def bounding_box(img):
        """Returns pixels for a bounding box: ((bottom_right_xy), (top_left_xy))"""
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        try:
            y_upper, y_lower = np.where(rows)[0][[0, -1]]
            x_left, x_right = np.where(cols)[0][[0, -1]]
        except IndexError:
            return None

        # which is a really tight bounding box.
        # return [int(x_right), int(y_lower)], [int(x_left), int(y_upper)]

        # Make it larger
        s = 5  # bounding box spacing
        w, h = img.shape[1]-1, img.shape[0]-1
        return ([
             int(min(w, x_right+s)),
             int(min(h, y_lower+s))],
            [int(max(0, x_left-s)),
             int(max(0, y_upper-s))])

    bboxes = bounding_box(mask)

    if visualize:
        print(f"lower right: {bboxes[0]}, upper left: {bboxes[1]}")
        x, y = zip(*bboxes)
        x1, x2, y1, y2 = *x, *y

        rgb, _ = camera.computeImageAndDepth(False)
        plt.imshow(rgb)
        plt.scatter(x, y)
        plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1])
        plt.show()

    if return_mask:
        return bboxes, mask
    return bboxes


def camera_to_world(conf, camera_frame="camera"):
    """ Returns the transformation of the camera in the world frame. """
    cf = conf.getFrame(camera_frame)
    return {
        "quaternion": dict(zip("wxyz", cf.getQuaternion().tolist())),
        "translation": dict(zip("xyz", cf.getPosition().tolist()))
    }


def calc_point_pixel_coords(C, keypoints_camera_frame, camera_frame_name, point_frames=["stick_p1", "stick_p2"]):
    """ Calculates the pixel positions of the sticks endpoints using the camera's parameters. """
    cam_frame_info = C.frame("camera").info()
    focal_length, width, height = [cam_frame_info[key] for key in ["focalLength", "width", "height"]]
    point_pixel_coords = []

    for kp in keypoints_camera_frame:
        X, Y, Z = kp

        x = focal_length * X / Z
        y = focal_length * Y / Z

        u, v = int(width/2 + x*height), int(height/2 - y*height)

        if u < 0 or u >= width:
            continue
        if v < 0 or v >= height:
            continue

        point_pixel_coords.append((u, v))
    return point_pixel_coords


def calc_point_depth(conf, point_frames=["stick_p1", "stick_p2"], camera_frame="camera"):
    """
    Calculates the depth of a keypoint. This is *NOT* the euclidean distance
    but the distance on the z axis given by the camera's viewing direction.
    """
    distances = []
    for point_frame in point_frames:
        point_pos = conf.frame(point_frame).getPosition()
        tmp_frame = conf.frame("cam_tmp")  # temporary frame in camera's coord sys
        tmp_frame.setPosition(point_pos)
        distances.append(float( -tmp_frame.getRelativePosition()[2] ))
    return distances


def keypoint_pixel_xy_depth(conf, camera, keypoints_camera_frame):
    keypoints_xy = calc_point_pixel_coords(conf, keypoints_camera_frame, "camera")
    if not keypoints_xy:
        return False
    keypoints_depth = calc_point_depth(conf)

    # Depth is adjusted to mm
    xyd_list = [list(keypoints_xy[i]) + [int(1000*keypoints_depth[i])] for i in range(len(keypoints_xy))]

    if len(xyd_list) != len(keypoints_camera_frame):
        # Image is invalid if the keypoints are outside of the image frame.
        # Occluded keypoints would be fine but this one checks whether a
        # keypoint outside the camera's view frustum.
        return False

    return xyd_list


def visible_keypoint_count(C, seg, keypoints_xyd, valid_frame_names=["stick", "stick_p1", "stick_p2"]):
    """
    Checks whether all keypoints are visible in the output image.

    :param camera: libry.CameraView from which the image was taken
    :param keypoints_xyd: List with a tuple of x_pos, y_pos and depth (mm) for each keypoint in the image. The positions refer to pixel positions in the captured image.
    :param valid_frame_names: Names of the frames which a keypoint can belong to.
    """
    def color2id(rgb):
        """
        Converts segmentation output rgb to object IDs.
        From: rai/Gui/opengl.cpp
        """
        _id = 0
        _id |= (rgb[0]&0x80)>>7 | (rgb[1]&0x80)>>6 | (rgb[2]&0x80)>>5
        _id |= (rgb[0]&0x40)>>3 | (rgb[1]&0x40)>>2 | (rgb[2]&0x40)>>1
        _id |= (rgb[0]&0x3f)<<6 | (rgb[1]&0x3f)<<12 | (rgb[2]&0x3f)<<18
        return _id

    valid_frame_ids = [C.frame(fname).info()["ID"] for fname in valid_frame_names]

    vis_kp_count = 0

    for keypoint_xyd in keypoints_xyd:
        x, y, _ = keypoint_xyd

        pixel_id = color2id(seg[y, x])
        if pixel_id in valid_frame_ids:
            vis_kp_count += 1

    return vis_kp_count


def all_keypoints_visible(C, seg, keypoints_xyd, valid_frame_names=["stick", "stick_p1", "stick_p2"]):
    return 2 == visible_keypoint_count(C, seg, keypoints_xyd, valid_frame_names)


def gen_images_adjust_depth(C, camera, camera_frame_name, vis=False, show_only_vision_geom=True):
    """
    Generates RGB and Depth images. The latter adjusted to mm range.
    Export as png to keep pixel values > 255
    """
    rgb, depth = camera.computeImageAndDepth(show_only_vision_geom)

    camera_depth_limit = C.frame(camera_frame_name).info()["zRange"][1]

    depth_mm = depth
    depth_mm[depth_mm < 0] = camera_depth_limit
    depth_mm = depth_mm * 1000
    depth_mm = depth_mm.astype(np.uint16)

    if vis:
        print("will visualize rgb and depth")
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.imshow(rgb)
        ax2.imshow(depth)
        plt.show()

    return rgb, depth_mm


def read_img_from_file(rgb_filename, depth_filename, camera_depth_limit=3.0):
    """ Loads images from the training data. """
    from PIL import Image
    rgb = np.array(Image.open(rgb_filename))
    depth = np.array(Image.open(depth_filename))

    depth_mm = depth
    # saved images are already scaled
    # depth_mm[depth_mm < 0] = camera_depth_limit
    # depth_mm = depth_mm * 1000
    depth_mm = depth_mm.astype(np.uint16)

    return rgb, depth_mm


def masked_image(camera, masking_frame="stick"):
    mask = camera.extractMask(masking_frame)
    return mask