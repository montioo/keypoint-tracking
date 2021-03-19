#!/usr/bin/env python3

# Learning and Intelligent Systems Project
# ---
# Data Generation for a scene where the gripper is visible within the data set and
# data is collected in sequences with temporal correspondence.
#
# Marius Montebaur, WS20/21

import sys
sys.path.append("../third_party/rai/rai/ry")
import libry as ry
from random import random, uniform, choice
import os
from config import dataset_desc_filename, datapoint_count
from synthetic_data_frame import DataFrame, OutputFolders, dump_data_as_yaml, create_folder_structure
from randomize_camera_position import set_random_camera_pose
from common.domain_randomization import random_rgb, set_stick_color, set_table_color, randomize_stick_pos, set_random_gripper_goal, random_stick_pos_in_gripper_local
from common.robot_movement import plan_komo_path


def hide_occlusion_geometry(C):
    occ_frame = C.frame("occlusion_geometry")
    occ_frame.setPosition([0, 0, 4])

def place_occlusion_geometry(C):
    """ Will place an object between camera and one keypoint to simulate occluded keypoints. """

    occ_frame = C.frame("occlusion_geometry")
    cam_pos = C.frame("camera").getPosition()
    keypoint_pos = C.frame(choice(["stick_p1", "stick_p2"])).getPosition()

    vector_cam_keypoint = cam_pos - keypoint_pos
    occ_pos = keypoint_pos + vector_cam_keypoint * uniform(0.0, 0.6)
    occ_frame.setPosition(occ_pos)

    shape_type = choice([1, 2, 3])
    if shape_type == 1:
        occ_frame.setShape(ry.ST.cylinder, [uniform(0.05, 0.10), uniform(0.03, 0.07)])
    elif shape_type == 2:
        occ_frame.setShape(ry.ST.sphere, [uniform(0.05, 0.15)])
    elif shape_type == 3:
        occ_frame.setShape(ry.ST.box, [uniform(0.05, 0.15), uniform(0.05, 0.15), uniform(0.05, 0.15)])

    occ_frame.setColor(random_rgb())


def main():
    # === Setup ===

    # Lenght of one sequence, i.e. length of a group of images that have a temporal relation
    time_series_length = 5

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
    C.addFrame("occlusion_geometry")

    sample_id = 0
    while sample_id < datapoint_count:

        # Data generation loop for one movement, consisting of multiple images

        # Randomize Scene Contents
        set_table_color(C, random_rgb())
        set_stick_color(C, random_rgb())
        random_stick_pos_in_gripper_local(C, gripper_frame_name="endeffR")
        set_random_camera_pose(C)

        # Set gripper goal
        EE_goal_frame_name = set_random_gripper_goal(C, vis=False, z_interval=(0.05, 0.5))

        # Plan gripper path
        steps, duration = 15, 5.
        komo = plan_komo_path(C, steps=steps, obj_goal=EE_goal_frame_name, duration=duration)

        # === Store data (image and label) for all configs of this sequence ===

        invalid_counter = 0  # used to skip a single robot configuration that doesn't show keypoints
        sequence = []  # holds an images sequence

        for sequence_step_id in range(steps):
            # Step through the different configurations on the movement path.
            # Capture images to generate training data with a temporal context.

            world_config = komo.getConfiguration(sequence_step_id)
            C.setFrameState(world_config)

            ds = DataFrame(C, output_folders)
            ds.extract_data(C)

            # === State machine that captures

            if len(sequence) == 0:
                # first, need an image with two visible keypoints to start a sequence
                if ds.vis_kp_count == 2:
                    sequence.append(ds)
                    invalid_counter = 0

            elif len(sequence) < time_series_length:
                # capture more images for the sequence

                if ds.vis_kp_count == 0:
                    # one image without a keypoint can be skipped
                    if invalid_counter == 1:
                        sequence = []
                        invalid_counter = 0
                    else:
                        # skip this image
                        invalid_counter += 1

                else:
                    # if the image only shows one keypoint, it is ok
                    if ds.vis_kp_count == 2 and random() > 0.5:
                        # for two visible keypoints: maybe place occlusion
                        # geometry and recapture image
                        place_occlusion_geometry(C)
                        ds.extract_data(C)
                    sequence.append(ds)

            elif len(sequence) == time_series_length:
                # done with this sequence. Save it.
                for d_point in sequence:
                    dataframes[sample_id] = d_point.write_images_and_return_dataframe_dict(sample_id)
                    sample_id += 1

                invalid_counter = 0
                sequence = []

                print(f"Added a sequence of {time_series_length} to the dataset.")

            else:
                print("len(sequence) > time_series_length")
                raise RuntimeError("this shouldn't happen")

            hide_occlusion_geometry(C)

            if sample_id >= datapoint_count:
                break

    filename = os.path.join(output_folders.root_folder, dataset_desc_filename)
    dump_data_as_yaml(filename, dataframes)


if __name__ == "__main__":
    main()
