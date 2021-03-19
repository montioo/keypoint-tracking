#!/usr/bin/env python3

# Learning and Intelligent Systems Project
# ---
# Data generation for a simple case where the stick is laying flat on the
# table and the camera is orbiting around it.
#
# Marius Montebaur, WS20/21

import sys
sys.path.append("../third_party/rai/rai/ry")

import libry as ry
from math import pi
from datetime import datetime
import os
from config import dataset_desc_filename, datapoint_count
from synthetic_data_frame import DataFrame, OutputFolders, dump_data_as_yaml, create_folder_structure
from randomize_camera_position import set_random_camera_pose
from common.domain_randomization import random_rgb, set_stick_color, set_table_color, randomize_stick_pos


def move_arm_away(C):
    """ Moves the right arm upright to make sure it's not visible in the camera's image. """
    joint_name_id_map = dict(zip(C.getJointNames(), range(len(C.getJointState()))))

    q = C.getJointState()
    for arm in "LR":
        q[joint_name_id_map[arm + "_panda_joint2"]] = 0
        q[joint_name_id_map[arm + "_panda_joint4"]] = 0
        q[joint_name_id_map[arm + "_panda_joint6"]] = pi/2
    C.setJointState(q)


def main():
    # === Setup ===

    output_folders = create_folder_structure()

    # optionally add a description to the dataset
    if len(sys.argv) >= 2:
        with open(os.path.join(output_folders.root_folder, "ds_desc.txt"), "w") as f:
            f.write(sys.argv[1])

    dataframes = {}

    # === Data Generation ===

    C = ry.Config()
    C.addFile("common/world.g")
    # viewer = C.view()
    C.addFrame("cam_tmp", parent="camera")

    move_arm_away(C)

    i = 0
    while i < datapoint_count:

        # Randomize Scene Contents
        set_stick_color(C, random_rgb())
        set_table_color(C, random_rgb())
        randomize_stick_pos(C)
        set_random_camera_pose(C)

        df = DataFrame(C, output_folders)
        df.extract_data(C)

        if df.shows_two_keypoints():
            # Data is randomly generated and sometimes doesn't show two keypoints
            dataframes[i] = df.write_images_and_return_dataframe_dict(i)
            if i % 50 == 0 and i != 0:
                print("Done with sample", i)
            i += 1

    filename = os.path.join(output_folders.root_folder, dataset_desc_filename)
    dump_data_as_yaml(filename, dataframes)


if __name__ == "__main__":
    main()
