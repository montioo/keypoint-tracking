#!/usr/bin/env python3

# Learning and Intelligent Systems Project
# ---
# Utility functions that are used in various parts of the training and data
# processing scripts.
#
# Marius Montebaur, WS20/21

import sys
import datetime
import numpy as np


def get_date_string():
    return datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d_%H:%M:%S")


def quat_to_rot_mat(quaternion):
    """ Compute homogeneous rotation matrix from quaternion. """
    q0, q1, q2, q3 = quaternion
    return np.array([
        [2*(q0**2 + q1**2) - 1, 2*(q1*q2 - q0*q3),     2*(q1*q3 + q0*q2),     0],
        [2*(q1*q2 + q0*q3),     2*(q0**2 + q2**2) - 1, 2*(q2*q3 - q0*q1),     0],
        [2*(q1*q3 - q0*q2),     2*(q2*q3 + q0*q1),     2*(q0**2 + q3**2) - 1, 0],
        [0,                     0,                     0,                     1]
    ])


class Logger:
    """
    Instead of using the logging module, use this to overwrite the print
    function to write to std out and to file.
    Usage: sys.stdout = Logger()
    """

    def __init__(self, filename_base="keypoint_training"):
        self.terminal = sys.stdout
        logfile_date = get_date_string()
        logfile_name = f"{filename_base}_{logfile_date}.log"
        self.log = open(logfile_name, "a")

    def __del__(self):
        self.log.close()

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
