#!/usr/bin/env python3

# Learning and Intelligent Systems Project
# ---
# Using the trained keypoint detection model for inference
#
# Marius Montebaur, WS20/21

import mankey.network.inference as inference
from mankey.utils.imgproc import PixelCoord
import os
import numpy as np
import time
import torch


class RegionOfInterest:
    """
    Analogous to the corresponding ROS msg:
        http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/RegionOfInterest.html
    Could uses dataclasses but currently running Python 3.6
    """
    def __init__(self, x_offset, y_offset, width, height):
        self.x_offset = x_offset  # Leftmost pixel of the ROI
        self.y_offset = y_offset  # Topmost pixel of the ROI
        self.height = height  # Height of ROI
        self.width = width  # Width of ROI

    def __repr__(self):
        return f"roi: x: {self.x_offset}, y: {self.y_offset}, width: {self.width}, height: {self.height}"

    @staticmethod
    def build_from_bbox_list(bboxes):
        """
        bboxes is a list of bounding box coordinates given by
        ((right_x, bottom_y), (left_x, top_y))
        """
        x_right, y_lower, x_left, y_upper = bboxes[0] + bboxes[1]

        # The bounding box
        roi = RegionOfInterest(
            x_left,
            y_upper,
            x_right - x_left,
            y_lower - y_upper
        )

        return roi


class DetectedKeypoints:
    """
    Result of the inference.
    """

    def __init__(self, keypoints_camera_frame=None, keypoints_xy_depth=None):
        """
        :param keypoints_camera_frame: 3D backprojection of the keypoints in the camera's coordinate system.
        :param keypoints_xy_depth: The network's prediction as a list of lists with three
        values each, giving pixel positions x and y as well as depth in mm.
        """
        self.keypoints_camera_frame = [] if keypoints_camera_frame is None else keypoints_camera_frame
        self.keypoints_xy_depth = [] if keypoints_xy_depth is None else keypoints_xy_depth
        self.num_keypoints = 0


class Point:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"x: {self.x}, y: {self.y}, z: {self.z}"

    def __getitem__(self, key):
        return [self.x, self.y, self.z][key]

    def tolist(self):
        return [self.x, self.y, self.z]



class MankeyKeypointDetection:

    def __init__(self, network_chkpt_path, K=None):
        """
        Loads the model and stores the camera parameters.

        :param network_chkpt_path: Path to network save
        :param K: np.ndarray holding the camera's intrinsics
        """
        print("Keypoint Inference")
        print("  Network path:", network_chkpt_path)

        if K is None:
            # Default camera parameters
            K = np.array([
                [ 166.8,   0.0, -160.0],
                [   0.0, 166.8, -120.0],
                [   0.0,   0.0,   -1.0]
            ])
        self.K_inv = np.linalg.inv(K)

        assert os.path.exists(network_chkpt_path), "Network path does not exist! Aborting"
        self._load_network(network_chkpt_path)

    def handle_keypoint_request(self, cv_color, cv_depth, bbox):
        start_time = time.time()

        # rgb to bgr
        cv_color[:, :, 0], cv_color[:, :, 2] = cv_color[:, :, 2], cv_color[:, :, 0]

        keypoints_xy_depth, camera_keypoint = self._process_request(cv_color, cv_depth, bbox)

        response = DetectedKeypoints()
        response.num_keypoints = camera_keypoint.shape[1]

        for i in range(camera_keypoint.shape[1]):
            point = Point()
            point.x = camera_keypoint[0, i]
            point.y = camera_keypoint[1, i]
            point.z = camera_keypoint[2, i]
            response.keypoints_camera_frame.append(point)

        # This says x y z but is actually x y depth
        for i in range(camera_keypoint.shape[1]):
            point = Point()
            point.x = keypoints_xy_depth[0, i]
            point.y = keypoints_xy_depth[1, i]
            point.z = keypoints_xy_depth[2, i]
            response.keypoints_xy_depth.append(point)

        end_time = time.time()
        print(f"Complete interference took: {(end_time-start_time):.4f} secs")

        return response

    def _process_request(self, cv_color, cv_depth, bbox):
        """
        Takes RGBD image and bounding box coordinates to infer the 3D keypoints
        """

        # Parse the bounding box
        top_left, bottom_right = PixelCoord(), PixelCoord()
        top_left.x = bbox.x_offset
        top_left.y = bbox.y_offset
        bottom_right.x = bbox.x_offset + bbox.width
        bottom_right.y = bbox.y_offset + bbox.height

        # Perform the inference
        imgproc_out = inference.proc_input_img_raw(
            cv_color, cv_depth,
            top_left, bottom_right)

        # keypointxy_depth_scaled = inference.inference_resnet_nostage(self._network, imgproc_out)
        keypointxy_depth_scaled = self._query_network(imgproc_out)

        keypointxy_depth_realunit = inference.get_keypoint_xy_depth_real_unit(keypointxy_depth_scaled)
        # print("keypoint_x_y_depth:", keypointxy_depth_realunit)

        keypoint_xy_depth_img, camera_keypoint = inference.get_3d_prediction_K(
            keypointxy_depth_realunit,
            imgproc_out.bbox2patch,
            self.K_inv)

        # print(keypoint_xy_depth_img)
        # print("cam keypoint", camera_keypoint)
        return keypoint_xy_depth_img, camera_keypoint

    def _query_network(self, imgproc_out):
        keypointxy_depth_scaled = inference.inference_resnet_nostage(self._network, imgproc_out)
        return keypointxy_depth_scaled

    def _load_network(self, network_chkpt_path):
        self._network, self._net_config = inference.construct_resnet_nostage(network_chkpt_path)


class MankeyKeypointDetectionLSTM(MankeyKeypointDetection):

    def __init__(self, network_chkpt_path, K=None):
        super().__init__(network_chkpt_path, K)
        self.reset_hidden_state()

    def reset_hidden_state(self):
        layer_dim = 1
        hidden_dim = 2*2*32*32
        batch_size = 1

        # h0, c0
        self._hidden = (torch.zeros(layer_dim, batch_size, hidden_dim),
                        torch.zeros(layer_dim, batch_size, hidden_dim))

    def _query_network(self, imgproc_out):
        keypointxy_depth_scaled, self._hidden = inference.inference_resnet_nostage_lstm(self._network, imgproc_out, self._hidden)
        return keypointxy_depth_scaled

    def _load_network(self, network_chkpt_path):
        self._network, self._net_config = inference.construct_resnet_nostage_lstm(network_chkpt_path)
