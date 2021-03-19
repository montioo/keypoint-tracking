#!/usr/bin/env python3

# Learning and Intelligent Systems Project
# ---
# This file contains setup code for the datasets used for training. A lot of
# information in here are hardcoded because this config was used for all
# different training experiments.
#
# Marius Montebaur, WS20/21

from mankey.dataproc.spartan_supervised_db import SpartanSupvervisedKeypointDBConfig, SpartanSupervisedKeypointDatabase
from mankey.dataproc.supervised_keypoint_loader import SupervisedKeypointDatasetConfig, SupervisedKeypointDataset
from mankey.dataproc.supervised_timeseries_wrapper import TimeseriesWrapper
from mankey.network.resnet_nostage import ResnetNoStageConfig, ResnetNoStage, ResnetNoStageLSTM


segmented_img_size = 128


def construct_dataset(is_train: bool, db_config_file_path, time_sequence_length=None):
    """
    -> (torch.utils.data.Dataset, SupervisedKeypointDatasetConfig)
    """
    # Construct the db
    db_config = SpartanSupvervisedKeypointDBConfig()
    db_config.keypoint_yaml_name = 'stick_2_keypoint_image.yaml'
    db_config.pdc_data_root = "/home/monti/Desktop/pdc"
    db_config.config_file_path = db_config_file_path
    database = SpartanSupervisedKeypointDatabase(db_config)

    # Construct torch dataset
    config = SupervisedKeypointDatasetConfig()
    config.network_in_patch_width = segmented_img_size
    config.network_in_patch_height = segmented_img_size
    config.network_out_map_width = 64
    config.network_out_map_height = 64
    config.image_database_list.append(database)
    config.is_train = is_train
    dataset = SupervisedKeypointDataset(config)

    if time_sequence_length:
        # Optionally wraps the dataset to query time series data
        dataset = TimeseriesWrapper(dataset, time_sequence_length)

    return dataset, config


def construct_network(for_timeseries_data=False):
    net_config = ResnetNoStageConfig()
    net_config.num_keypoints = 2
    net_config.image_channels = 4
    net_config.depth_per_keypoint = 2
    net_config.num_layers = 18
    if for_timeseries_data:
        network = ResnetNoStageLSTM(net_config)
    else:
        network = ResnetNoStage(net_config)
    return network, net_config