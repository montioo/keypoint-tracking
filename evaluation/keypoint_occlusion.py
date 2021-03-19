#!/usr/bin/env python3

# Learning and Intelligent Systems Project
# Simple test to see how the network responds to images with occluded keypoints.
#
# Marius Montebaur, WS20/21


from keypoint_inference import *
from visualization import visualize_keypoints_in_rgb
from data_extraction import read_img_from_file


# rgb_img_path = "/home/monti/Desktop/pdc/logs_proto/2021-01-30-18-57-05/processed/images/000007_rgb.png"
# depth_img_path = "/home/monti/Desktop/pdc/logs_proto/2021-01-30-18-57-05/processed/images/000007_depth.png"
rgb_img_path = "/home/monti/Desktop/depth_tests/000007_rgb.png"
depth_img_path = "/home/monti/Desktop/depth_tests/000007_depth.png"

# bbox_bottom_right_xy: [270, 138]
# bbox_top_left_xy: [149, 114]

p = 7  # padding
roi = RegionOfInterest(149-p, 114-p, 270-149+2*p, 138-114+2*p)

rgb, depth_mm = read_img_from_file(rgb_img_path, depth_img_path)

# artificial occlusion
rgb[100:145, 135:180, :] = 0
depth_mm[100:145, 135:180] = 0

keypoint_network_path = "/home/monti/Desktop/pdc/resnet18_2021-02-01/checkpoint_0236.pth"
keypoint_inference = MankeyKeypointDetection(keypoint_network_path)



# roi = RegionOfInterest.build_from_bbox_list(bboxes)
resp = keypoint_inference.handle_keypoint_request(rgb.copy(), depth_mm, roi)

print(resp.keypoints_camera_frame)

bboxes = (
    [roi.x_offset + roi.width, roi.y_offset + roi.height],
    [roi.x_offset, roi.y_offset]
)

visualize_keypoints_in_rgb(rgb, bboxes, resp, depth_mm)

