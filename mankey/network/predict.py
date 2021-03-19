#!/usr/bin/env python3

# Learning and Intelligent Systems Project
# ---
# Does the heat- and depthmap evaluations to get the network's keypoint
# position predictions.
#
# This script is based on work by Wei Gao and Lucas Manuelli from https://github.com/weigao95/mankey-ros
# Marius Montebaur, WS20/21

import torch
from torch.nn import functional as F
import torch.cuda.comm


def heatmap_from_predict(probability_preds, num_keypoints):  # type: (torch.Tensor, int) -> torch.Tensor:
    """
    Given the probability prediction, compute the actual probability map using softmax.
    :param probability_preds: (batch_size, n_keypoint, height, width) for 2d heatmap (batch_size, n_keypoint, height, width)
    :param num_keypoints:
    :return:
    """
    assert probability_preds.shape[1] == num_keypoints
    heatmap = probability_preds.reshape((probability_preds.shape[0], num_keypoints, -1))
    heatmap = F.softmax(heatmap, dim=2)
    heatmap = heatmap.reshape(probability_preds.shape)
    return heatmap


def heatmap2d_to_imgcoord_cpu(
        heatmap,
        num_keypoints):  # type: (torch.Tensor, int) -> (torch.Tensor, torch.Tensor)
    """
    Given the heatmap, regress the image coordinate in x (width) and y (height) direction.
    This implementation only works for 2D heatmap.
    Note that the coordinate is in [0, map_width] for x and [0, map_height] for y
    :param heatmap: (batch_size, n_keypoints, map_height, map_width) heatmap
    :param num_keypoints: For sanity check only
    :return: A tuple contains two (batch_size, n_keypoints, 1) tensor for x and y coordinate
    """
    assert heatmap.shape[1] == num_keypoints
    batch_size, _, y_dim, x_dim = heatmap.shape

    # Compute the probability on each dim
    accu_x = heatmap.sum(dim=2)
    accu_y = heatmap.sum(dim=3)

    # Element-wise product the image coord
    accu_x = accu_x * torch.arange(x_dim).type(torch.FloatTensor)
    accu_y = accu_y * torch.arange(y_dim).type(torch.FloatTensor)

    # Further reduce to three (batch_size, num_keypoints) tensor
    accu_x = accu_x.sum(dim=2, keepdim=True)
    accu_y = accu_y.sum(dim=2, keepdim=True)
    return accu_x, accu_y


def heatmap2d_to_imgcoord_gpu(
        heatmap,
        num_keypoints):  # type: (torch.Tensor, int) -> (torch.Tensor, torch.Tensor)
    assert heatmap.shape[1] == num_keypoints
    batch_size, _, y_dim, x_dim = heatmap.shape

    # Compute the probability on each dim
    accu_x = heatmap.sum(dim=2)
    accu_y = heatmap.sum(dim=3)

    # The pointwise product
    accu_x = accu_x * torch.cuda.comm.broadcast(torch.arange(x_dim), devices=[accu_x.device.index])[0].type(
        torch.cuda.FloatTensor)
    accu_y = accu_y * torch.cuda.comm.broadcast(torch.arange(y_dim), devices=[accu_y.device.index])[0].type(
        torch.cuda.FloatTensor)

    # Further reduce to three (batch_size, num_keypoints) tensor
    accu_x = accu_x.sum(dim=2, keepdim=True)
    accu_y = accu_y.sum(dim=2, keepdim=True)
    return accu_x, accu_y


def heatmap2d_to_normalized_imgcoord_gpu(
        heatmap,
        num_keypoints):  # type: (torch.Tensor, int) -> (torch.Tensor, torch.Tensor)
    """
    Regress the normalized coordinate for x and y from the heatmap.
    The range of normalized coordinate is [-0.5, -0.5] in current implementation.
    :param heatmap:
    :param num_keypoints:
    :return:
    """
    # The un-normalized image coord
    _, _, y_dim, x_dim = heatmap.shape
    coord_x, coord_y = heatmap2d_to_imgcoord_gpu(heatmap, num_keypoints)

    # Normalize it
    coord_x *= float(1.0 / float(x_dim))
    coord_y *= float(1.0 / float(y_dim))
    coord_x -= 0.5
    coord_y -= 0.5
    return coord_x, coord_y

def heatmap2d_to_normalized_imgcoord_cpu(
        heatmap,
        num_keypoints):  # type: (torch.Tensor, int) -> (torch.Tensor, torch.Tensor)
    """
    Regress the normalized coordinate for x and y from the heatmap.
    The range of normalized coordinate is [-0.5, -0.5] in current implementation.
    :param heatmap:
    :param num_keypoints:
    :return:
    """
    # The un-normalized image coord
    _, _, y_dim, x_dim = heatmap.shape
    coord_x, coord_y = heatmap2d_to_imgcoord_cpu(heatmap, num_keypoints)

    # Normalize it
    coord_x *= float(1.0 / float(x_dim))
    coord_y *= float(1.0 / float(y_dim))
    coord_x -= 0.5
    coord_y -= 0.5
    return coord_x, coord_y

def depth_integration(heatmap, depth_pred):  # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
    """
    Given the heatmap (normalized) and depthmap prediction, compute the
    depth value at the keypoint. There is no normalization on the depth.
    The implementation only works for 2D heatmap and depthmap.
    :param heatmap: (batch_size, n_keypoint, map_height, map_width) normalized heatmap
                    heatmap[batch_idx, keypoint_idx, :, :].sum() == 1
    :param depth_pred: (batch_size, n_keypoint, map_height, map_width) depth map
    :return: (batch_size, n_keypoint, 1) depth prediction.
    """
    assert heatmap.shape == depth_pred.shape
    n_batch, n_keypoint, _, _ = heatmap.shape
    # The pointwise product
    predict = heatmap * depth_pred

    # Aggreate on the result
    predict = predict.reshape(shape=(n_batch, n_keypoint, -1))
    return predict.sum(dim=2, keepdim=True)
