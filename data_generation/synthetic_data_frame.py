#!/usr/bin/env python3

# Learning and Intelligent Systems Project
# ---
# Domain randomization in the sim scene. Mainly used for training data generation.
#
# Marius Montebaur, WS20/21

import os
import yaml
from common.data_extraction import *
from PIL import Image
import time
from datetime import datetime
from config import dataset_root_folder


class OutputFolders:
    """
    Holds the paths of different folders used with the data generation. Refer
    to the training data layout to learn more about this.
    """
    # make this a dataclass with python >= 3.7
    def __init__(self):
        self.root_folder = ""
        self.image_folder = ""
        self.image_mask_folder = ""

    def as_tuple(self):
        return (self.root_folder, self.image_folder, self.image_mask_folder)


def create_folder_structure():
    """ Creates and returns the folder structure for a new dataset. """

    output_folders = OutputFolders()
    time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_folders.root_folder = f"{dataset_root_folder}/logs_proto/{time_str}/processed"
    output_folders.image_folder = os.path.join(output_folders.root_folder, "images")
    output_folders.image_mask_folder = os.path.join(output_folders.root_folder, "image_masks")

    os.makedirs(output_folders.root_folder)
    os.makedirs(output_folders.image_folder)
    os.makedirs(output_folders.image_mask_folder)
    return output_folders



def dump_data_as_yaml(filename, d):
    """
    Can append to yaml file and can store a new file if it doesn't exist already.
    Make sure to not include things like numpy arrays in the data `d`. Also
    just converting a numpy array to a python list with np.ndarray.tolist()
    won't work, since the containing type is a list but the individual
    entries are still numpy numerical values and no default python types.
    """
    try:
        with open(filename,'r') as yamlfile:
            cur_yaml = yaml.load(yamlfile)
            if cur_yaml is None:
                cur_yaml = d
            else:
                cur_yaml.update(d)
    except FileNotFoundError:
        cur_yaml = d

    with open(filename,'w') as yamlfile:
        yaml.dump(cur_yaml, yamlfile)

    print(f"Wrote to file {filename} for dataframe")


def store_image(filename, img_np):
    img = Image.fromarray(img_np)
    img.save(filename)


def store_images(sample_id, rgb, depth_mm, mask, image_folder, image_mask_folder):
    """ Saves multiple images to disk with file naming convention for the dataset. """
    rgb_image_filename = f"{sample_id:06}_rgb.png"
    store_image(os.path.join(image_folder, rgb_image_filename), rgb)

    depth_image_filename = f"{sample_id:06}_depth.png"
    store_image(os.path.join(image_folder, depth_image_filename), depth_mm)

    masked_image_filename = f"{sample_id:06}_mask.png"
    store_image(os.path.join(image_mask_folder, masked_image_filename), mask)

    return rgb_image_filename, depth_image_filename, masked_image_filename


class DataFrame:
    """
    Collects all of the data from the simulated scene that is needed. Extract
    data can be called multiple times if something in the scene changed.
    """

    def __init__(self, C, output_folders):
        """
        self.vis_kp_count refers to the number of keypoints visible in the
        scene stored in this class. If no keypoint is visible, this sample
        might not be suited to be part of the training data.
        :param C: rai configuration
        :param output_folders: An instance of class OutputFolders containing
        information on where to store the resulting images.
        """
        self.output_folders = output_folders
        self.vis_kp_count = 0

    def shows_two_keypoints(self):
        return self.vis_kp_count == 2

    def create_cam(self, C):
        """
        Creates a new rai camera for this data frame. A new camera is
        generated for every sample because this turned out to be more stable.
        :param C: rai configuration
        """
        camera = C.cameraView()
        camera.addSensorFromFrame("camera")
        camera.selectSensor("camera")
        return camera


    def extract_data(self, C):
        """
        Extract data from the rai simulation scene. Will not write it to disk.
        :param C: rai configuration
        """
        camera = self.create_cam(C)
        camera.updateConfig(C)

        rgb, depth_mm = gen_images_adjust_depth(C, camera, "camera", vis=False)
        seg_img = camera.computeSegmentation()

        self.threed_keypoints_in_cam_frame = threed_keypoint_camera_frame(C)

        self.kp_pix_xyd = keypoint_pixel_xy_depth(C, camera, self.threed_keypoints_in_cam_frame)
        if isinstance(self.kp_pix_xyd, bool):
            # no keypoint visible no need to collect more data.
            self.vis_kp_count = 0
            del camera
            return

        self.vis_kp_count = visible_keypoint_count(C, seg_img, self.kp_pix_xyd)

        bbox_ret, mask = calc_bounding_box(camera, return_mask=True)
        if bbox_ret is None:
            # couldn't find bounding box. Data is invalid.
            self.vis_kp_count = 0
            del camera
            return

        self.bbr_xy, self.btl_xy = bbox_ret

        self.cam_to_world_T = camera_to_world(C)

        self.rgb = rgb
        self.depth_mm = depth_mm
        self.mask = mask
        del camera

    def write_images_and_return_dataframe_dict(self, sample_id):
        """
        Writes the images to disk and returns a dict that holds the ground truth.
        :param sample_id: an integer value that is used to identify the images.
        """
        self._write_images_to_disk(sample_id)

        d = {
            "3d_keypoint_camera_frame": self.threed_keypoints_in_cam_frame,
            "bbox_bottom_right_xy": self.bbr_xy,
            "bbox_top_left_xy": self.btl_xy,
            "camera_to_world": self.cam_to_world_T,
            "depth_image_filename": self.depth_image_filename,
            "keypoint_pixel_xy_depth": self.kp_pix_xyd,
            "rgb_image_filename": self.rgb_image_filename,
            "timestamp": str(time.time()).replace('.', '')
        }

        return d

    def _write_images_to_disk(self, sample_id):
        root_folder, image_folder, image_mask_folder = self.output_folders.as_tuple()
        rgb_image_filename, depth_image_filename, masked_image_filename = store_images(sample_id, self.rgb, self.depth_mm, self.mask, image_folder, image_mask_folder)
        self.rgb_image_filename, self.depth_image_filename, self.masked_image_filename = rgb_image_filename, depth_image_filename, masked_image_filename
