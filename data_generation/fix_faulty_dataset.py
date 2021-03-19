#!/usr/bin/env python3

# Learning and Intelligent Systems Project
# ---
# I had the problem that some ground truth data for my time series training data
# was accitentally not saved but instead replaced with None. This script finds those
# datapoints and removes the whole sequence they belong to.
#
# Marius Montebaur, WS20/21

import yaml
import os
import shutil


def main():

    pdc_folder_path = "/home/monti/Desktop/pdc/logs_proto_null"

    subfolders = os.listdir(pdc_folder_path)

    yaml_file = "processed/stick_2_keypoint_image.yaml"

    for subfolder in subfolders:

        print("looking at", subfolder)
        yf = os.path.join(pdc_folder_path, subfolder, yaml_file)

        try:
            with open(yf) as f:
                data = yaml.load(f)
        except FileNotFoundError:
            # dataset ground truth doesn't exist. Remove the dataset
            dataset_folder = os.path.join(pdc_folder_path, subfolder)
            shutil.rmtree(dataset_folder)
            print("  no dataset ground truth. continue.")
            continue

        fixed_data = {}

        for ds_idx in range(0, 2000, 5):
            valid = True
            for i in range(5):
                try:
                    if data[ds_idx + i] is None:
                        valid = False
                        break
                except KeyError:
                    # if key doesn't exist, this dataset was cleaned from errors before.
                    valid = False
                    break

            if valid:
                # move to new dataset
                for i in range(5):
                    fixed_data[ds_idx + i] = data[ds_idx + i]

            else:
                # remove images
                for i in range(5):
                    rgb_name = f"{ds_idx + i:06}_rgb.png"
                    rgb_path = os.path.join(pdc_folder_path, subfolder, "processed/images", rgb_name)

                    depth_name = f"{ds_idx + i:06}_depth.png"
                    depth_path = os.path.join(pdc_folder_path, subfolder, "processed/images", depth_name)

                    try:
                        os.remove(rgb_path)
                        os.remove(depth_path)
                    except FileNotFoundError:
                        pass

        with open(yf, "w") as f:
            yaml.dump(fixed_data, f)


if __name__ == "__main__":
    main()
