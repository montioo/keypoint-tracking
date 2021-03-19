#!/usr/bin/env python3

# Learning and Intelligent Systems Project
# ---
# All sorts of utility functions.
#
# Marius Montebaur, WS20/21

import numpy as np
import json
import datetime
import os


def euler_to_quaternion(roll, pitch, yaw):
    if abs(roll) > 2*np.pi or abs(pitch) > 2*np.pi or abs(yaw) > 2*np.pi:
        print("WARNING: roll pitch yaw required in radians, not degrees")
    # adapted from https://computergraphics.stackexchange.com/questions/8195/how-to-convert-euler-angles-to-quaternions-and-get-the-same-euler-angles-back-fr/8229#8229
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    # WARNING: Order for rai, not ros!
    return [qw, qx, qy, qz]


def delFrameIfExists(C, frame_name):
    """
    Deletes a frame from libry python simulator.
    """
    while frame_name in C.getFrameNames():
        C.delFrame(frame_name)

def get_datetime_str(for_filename=False):
    """
    Will return a date string for the current time. Optionally formatted with
    - and _ to use with filenames instead of . and :
    """

    n = datetime.datetime.now()
    if for_filename:
        return n.strftime("%d-%m-%y_%H-%M-%S")
    return n.strftime("%d.%m.%y %H:%M:%S")

def export_pred_and_truth_json(ground_truth, predictions, file_name):
    dirname = os.path.dirname(os.path.abspath(file_name))
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    out_dict = {
        "ground_truth": ground_truth,
        "predictions": predictions
    }

    try:
        with open(file_name, "w") as f:
            json.dump(out_dict, f)
    except TypeError:
        # Should the json export fail because there are numpy types in the dict, use pickle instead
        import pickle
        with open(file_name + ".p", "wb") as f:
            pickle.dump(out_dict, f)


def transform_position(C, pos, from_frame, to_frame=None):
    """
    Returns the coordinates of the given position (pos which is list of x, y,
    z) in the frame from_frame as the corresponding position in frame
    to_frame. If to_frame is None, the world frame will be used.
    """

    if to_frame and to_frame != "world":
        raise NotImplementedError("Currently no support for translation in target frame that is not the world frame")
        # TODO: Add support for arbitrary target frames

    tmp_frame_name_source = "tmp_from"
    # tmp_frame_name_target = "tmp_to"

    delFrameIfExists(C, tmp_frame_name_source)
    # delFrameIfExists(C, tmp_frame_name_target)

    tmp_frame_source = C.addFrame(tmp_frame_name_source, parent=from_frame)
    tmp_frame_source.setRelativePosition(pos)

    world_pos = tmp_frame_source.getPosition()
    delFrameIfExists(C, tmp_frame_name_source)

    return world_pos

def dirty_fix_numpy_json_export(pos_list):
    # numpy.ndarray.tolist() returns a list with np.float64 values and those are useless when it comes to json export.
    fixed = []
    for pos in pos_list:
        new_pos = [float(pos[0]), float(pos[1]), float(pos[2])]
        fixed.append(new_pos)
    return fixed

def dirty_fix_mirror_problem(pos_list):
    # TODO: Remove this and fix the projcetion
    # TODO: While doing so, remember that numpy.ndarray.tolist() returns a list with np.float64 values and those are useless when it comes to json export.
    fixed = []
    for pos in pos_list:
        new_pos = [float(-pos[0]), float(pos[1]), float(-pos[2])]
        fixed.append(new_pos)
    return fixed