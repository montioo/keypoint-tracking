#!/usr/bin/env python3

# Learning and Intelligent Systems Project
# ---
# Lists the amount of rgb images (i.e. training datasamples) found in the pdc folder
#
# Marius Montebaur, WS20/21

import yaml
import os
from config import dataset_root_folder


def main():

    pdc_folder_path = f"{dataset_root_folder}/online/logs_proto"

    subfolders = sorted(os.listdir(pdc_folder_path))
    sample_count = 0

    for subfolder in subfolders:

        print("looking at", subfolder, end="")
        images_folder = os.path.join(pdc_folder_path, subfolder, "processed/images")

        images_contents = os.listdir(images_folder)
        rgb_images_count = len(list(filter(lambda s: s.endswith("_rgb.png"), images_contents)))

        print(f" +++ found {rgb_images_count} samples")

        sample_count += rgb_images_count

    print("Total sample count:", sample_count)


if __name__ == "__main__":
    main()
