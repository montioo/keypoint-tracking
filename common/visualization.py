#!/usr/bin/env python3

# Learning and Intelligent Systems Project
# ---
# Visualization tools
#
# Marius Montebaur, WS20/21

import sys
sys.path.append("../third_party/rai/rai/ry")
import libry as ry
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt

def visualize_keypoints_in_rgb(rgb, bboxes, prediction_response, depth=None, plot_bbox=False):
    """
    Plots predicted keypoints in an image. Will also plot depth image if given.
    :param rgb: numpy.ndarray containing the RGB image
    :param bboxes: Bounding boxes as [(x_1, y_1), (x_2, y_2)]
    :param prediction_response: keypoint_inference.DetectedKeypoints instance
    :param depth: Depth image as numpy.ndarray with depth in mm
    :param plot_bbox: Whether or not to plot bounding boxes
    """
    if depth is not None:
        _, (ax1, ax2) = plt.subplots(1, 2)
        ax2.imshow(depth)
    else:
        _, ax1 = plt.subplots(1, 1)

    ax1.imshow(rgb)

    # bboxes
    if plot_bbox:
        x, y = zip(*bboxes)
        x1, x2, y1, y2 = *x, *y
        ax1.scatter(x, y)
        ax1.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1])

    # predictions (x, y)
    x_pred, y_pred = [], []
    for i in range(prediction_response.num_keypoints):
        kp_xy_d = prediction_response.keypoints_xy_depth[i]
        # print("keypoints from prediction:", kp_xy_d)

        x = kp_xy_d.x if hasattr(kp_xy_d, "x") else kp_xy_d[0]
        y = kp_xy_d.y if hasattr(kp_xy_d, "y") else kp_xy_d[1]
        x_pred.append(x)
        y_pred.append(y)
        # print("depth:", kp_xy_d.z)

    ax1.scatter(x_pred, y_pred)
    plt.show()


def visualize_eval_stages(rgb, bboxes, prediction_response, depth, filename=None):
    _, ax = plt.subplots(2, 2, figsize=(8,6))

    ax[0, 0].imshow(rgb)
    ax[0, 0].set_title("Input RGB Image")
    ax[0, 0].axis("off")

    ax[0, 1].imshow(depth)
    ax[0, 1].set_title("Input Depth Image [mm]")
    ax[0, 1].axis("off")

    ax[1, 0].imshow(rgb)
    ax[1, 0].set_title("Bounding Box from RGB")
    ax[1, 0].axis("off")

    ax[1, 1].imshow(rgb)
    ax[1, 1].set_title("Keypoints from RGBD + BBox")
    ax[1, 1].axis("off")

    # bboxes
    x, y = zip(*bboxes)
    x1, x2, y1, y2 = *x, *y
    ax[1, 0].scatter(x, y)
    ax[1, 0].plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1])
    ax[1, 1].scatter(x, y)
    ax[1, 1].plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1])

    # predictions (x, y)
    x_pred, y_pred = [], []
    for i in range(prediction_response.num_keypoints):
        kp_xy_d = prediction_response.keypoints_xy_depth[i]
        # print("keypoints from prediction:", kp_xy_d)

        x = kp_xy_d.x if hasattr(kp_xy_d, "x") else kp_xy_d[0]
        y = kp_xy_d.y if hasattr(kp_xy_d, "y") else kp_xy_d[1]
        x_pred.append(x)
        y_pred.append(y)
        # print("depth:", kp_xy_d.z)

    ax[1, 1].scatter(x_pred, y_pred)
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.show()