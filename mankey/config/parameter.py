#!/usr/bin/env python3

# Learning and Intelligent Systems Project
# ---
# Global parameters used for training and inference with the keypoint detection
# algorithm.
#
# This script includes work by Wei Gao and Lucas Manuelli from https://github.com/weigao95/mankey-ros
# Marius Montebaur, WS20/21

# The string key for dataset
rgbd_image_key = 'rgbd_image'
keypoint_xyd_key = 'normalized_keypoint_xyd'
keypoint_validity_key = 'validity'
target_heatmap_key = 'target_heatmap'

# The bounding box given by Database is tight, make it losser
bbox_scale = 1.25

# The normalization parameter
depth_image_clip = 2000  # Clip the depth image further than 1500 mm
depth_image_mean = 580
depth_image_scale = 256  # scaled_depth = (raw_depth - depth_image_mean) / depth_image_scale

# The averaged RGB image
rgb_mean = [0.485, 0.456, 0.406]

# The default size of path
default_patch_size_input = 128
default_patch_size_output = 64

# Parameters for the simulated camera
# Only used in inference.py:get_3d_prediction which is not part of the
# inference pipeline used for evaluation.
focal_x = 695
focal_y = 695
principal_x = 320/2
principal_y = 240/2
