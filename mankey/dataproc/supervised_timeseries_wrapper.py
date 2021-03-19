#!/usr/bin/env python3

# Learning and Intelligent Systems Project
# ---
# Wrapper for the keypoint training dataset that allows loading time series
# data without changing anything to the data loading code. The only restriction
# this approach imposes is that time series data, e.g. five images that show a
# movement, have to be stored directly after another in the dataset.
#
# Marius Montebaur, WS20/21

import numpy as np
import mankey.config.parameter as parameter


class TimeseriesWrapper:
    """
    This class is a wrapper for
    supervised_keypoint_loader.SupervisedKeypointDataset that groups time
    series data to make it usable with a LSTM enhanced version of the
    keypoint detection network.
    """

    def __init__(self, dataset, time_sequence_length):
        """
        :param dataset: A dataset that implements __getitem__ and __len__
        :param time_sequence_length: The length of a sequence that should be used as an input for the NN raining
        """
        self._dataset = dataset
        self._time_sequence_length = time_sequence_length

        # sanity check:
        if len(self._dataset) % self._time_sequence_length != 0:
            # if the dataset consists of 16 images and a time series has 5 images, something's wrong.
            raise RuntimeError("Can't be a time series dataset, as the images in the sequence don't align.")

    def __getitem__(self, index):
        """
        Returns the same data that is found in the dataset but converts it to
        a time series dataset by returning not one entry at a time but
        self._time_sequence_length entries. Therefore, each element in that
        range is queried from the original dataset, to then construct a new
        dict with stacked numpy arrays.
        The zeroth dim of each numpy array is the sequence dimension.
        """
        # Query the entries as it would be if each element was queried individually
        processed_entries = [self._dataset[i] for i in range(index, index+self._time_sequence_length)]

        keys = [parameter.rgbd_image_key, parameter.keypoint_xyd_key, parameter.keypoint_validity_key, parameter.target_heatmap_key]
        merged_time_series_data = {}

        for key in keys:
            entry_values = [processed_entries[i][key] for i in range(self._time_sequence_length)]
            merged_time_series_data[key] = np.stack(entry_values)

        return merged_time_series_data

    def __len__(self):
        return len(self._dataset) // self._time_sequence_length
