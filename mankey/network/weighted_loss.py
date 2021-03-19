#!/usr/bin/env python3

# Learning and Intelligent Systems Project
# ---
# Different loss functions used with keypoint training. Normally, using the L1
# or L2 loss would be sufficient, but since this repository focusses on the
# example use-case of keeping track of keypoints of a stick, some modifications
# to the loss functions were necessary. A stick is symmetric and thus the model
# can't learn a mapping for which keypoint has index 0 and which has index 1 in
# the network's output. A setting where the input data would always show the
# keypoint with index zero on the left half of the image was tried, but gave
# worse results than the loss functions used here.
#
# The loss functions defined here will always use the minimum loss for either
# keypoint association. The loss function will calculate the loss for mapping
# each keypoint in the ground truth data to either output heatmap of the
# keypoint prediction algorithm that is used. This way, the network is free to
# predict the keypoints where they are and doesn't have to learn conditions
# like "keypoint with index 0 is always left".
#
# This, however, imposes the limitation that the loss functions defined here only
# work for objects that have two keypoints.
#
# The sequence length is an optional parameter that is used for training on
# input sequence, i.e. multiple images with temporal relationship. In a
# sequence, the loss function will pick a mapping from keypoint output heatmap
# to ground truth for the first image of a sequence and use the same
# association for the remaining images in that sequence. This is supposed to
# help the model learn to track a keypoint.
#
# Marius Montebaur, WS20/21

import torch


def _sequence_symmetric_loss(pred, target, weight, size_avg=True, sequence_length=1, exp=1):
    """ Can calculate several different loss functions for scenes with two symmetric keypoints.
    See explanation at the top of this file for more info. Shape of pred and target tensors:
    (batch_and_sequence_size, number_of_keypoints_per_image=2, coordinates_per_keypoint)

    :param pred: Tensor with predicted values, shape given above
    :param target: Tensor with target values, shape given above
    :param weight: Tensor assigning a weight to each keypoint's 3D coordinate (xy and depth). Value zero will not include the keypoint in the calculated loss.
    :param size_avg: If True (default) the loss will be averaged over the number of batches
    :param sequence_length: Length of sequence data. The first dimension of the pred shape divided by sequence_length gives the batch_size. batch_size == batch_and_sequence_size for seqeunce_length == 1
    :param exp: Power of the exponent that is applied after the L1 loss is calculated. Defaults to 1, thus giving the L1 loss.
    :returns: Tensor with one element which is the loss
    """
    assert pred.shape[1] == 2, "This loss function is only suited for objects with two keypoints."

    loss = torch.abs(pred - target)**exp * weight
    flip_loss = torch.abs(torch.flip(pred, dims=(1,)) - target)**exp * weight

    s_sum = torch.sum(loss, dim=(1, 2))[::sequence_length]
    f_sum = torch.sum(flip_loss, dim=(1, 2))[::sequence_length]

    idx_to_swap = s_sum > f_sum
    idx_to_swap = idx_to_swap.repeat_interleave(sequence_length)
    loss[idx_to_swap] = flip_loss[idx_to_swap]

    return loss.sum() / len(pred) if size_avg else loss.sum()


def symmetric_loss_l1(pred, target, weight, size_avg=True):
    return _sequence_symmetric_loss(pred, target, weight, size_avg, 1)

def sequence_symmetric_loss_l1(pred, target, weight, size_avg=True, sequence_length=5):
    return _sequence_symmetric_loss(pred, target, weight, size_avg, sequence_length)

def symmetric_loss_mse(pred, target, weight, size_avg=True):
    return _sequence_symmetric_loss(pred, target, weight, size_avg, 2)

def sequence_symmetric_loss_mse(pred, target, weight, size_avg=True, sequence_length=5):
    return _sequence_symmetric_loss(pred, target, weight, size_avg, sequence_length, 2)


def symmetric_loss_l1_separate(pred, target, weight, size_avg=True):
    """ Inputs as above but will return xy position and depth loss seperately. """
    std_loss = torch.abs(pred - target) * weight
    flip_loss = torch.abs(torch.flip(pred, dims=(1,)) - target) * weight

    s_sum = torch.sum(std_loss, dim=(1, 2))
    f_sum = torch.sum(flip_loss, dim=(1, 2))

    idx_to_swap = s_sum > f_sum
    std_loss[idx_to_swap] = flip_loss[idx_to_swap]
    # separate returns for pixel position and depth
    xy = std_loss[:, :, 0:2]
    d = std_loss[:, :, 2]

    if size_avg:
        return xy.sum() / len(pred), d.sum() / len(pred)
    return xy.sum(), d.sum()
