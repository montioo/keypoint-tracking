#!/usr/bin/env python3

# Learning and Intelligent Systems Project
# ---
# Data Generation for a scene where the gripper is visible within the data set
#
# Marius Montebaur, WS20/21

import sys
sys.path.append("../third_party/rai/rai/ry")
import libry as ry
import os
from config import dataset_desc_filename, datapoint_count
from common.robot_movement import set_komo_inv_kin
from synthetic_data_frame import DataFrame, OutputFolders, dump_data_as_yaml, create_folder_structure
from randomize_camera_position import set_random_camera_pose
from common.domain_randomization import random_rgb, set_stick_color, set_table_color, randomize_stick_pos, set_random_gripper_goal, random_stick_pos_in_gripper_local


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
    C.addFile("common/single_world.g")
    # viewer = C.view()
    C.addFrame("cam_tmp", parent="camera")

    i = 0
    while i < datapoint_count:

        # Randomize Scene Contents
        set_stick_color(C, random_rgb())
        set_table_color(C, random_rgb())
        random_stick_pos_in_gripper_local(C, gripper_frame_name="endeffR")
        set_random_camera_pose(C)

        # move arm and stick to a random location
        EE_goal_frame_name = set_random_gripper_goal(C)
        set_komo_inv_kin(C, EE_goal_frame_name, EE_frame="endeffR")

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
