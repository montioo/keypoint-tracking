#!/usr/bin/env python3

# Learning and Intelligent Systems Project
# ---
# Custom class to collect and dump execution statistics
#
# Marius Montebaur, WS20/21

import time
import pickle
import json


class Stats:
    """
    Features:
    - log an object/value for a specific key
    - collect this data
    - have function for "new entry", like next line
    - dump all the data at the end
    """

    def __init__(self):
        self._all_data = []
        self._current_entry = {}

    def _clean_numpy_elements(self, input_list):
        """ Will replace numpy types with native python types to enable export. """
        pass

    def log_for_key(self, key, data, apply_cleaning=False):
        if apply_cleaning:
            raise NotImplementedError("Cleaning numpy is not implemented")
        self._current_entry[key] = data

    def new_log_entry(self):
        self._current_entry["closed"] = time.time()
        self._all_data.append(self._current_entry)
        self._current_entry = {}

    def write_to_file(self, filename):
        """ Write to json. Should that fail, write to pickle. """
        print("no json export. Using pickle")

        if len(self._current_entry):
            self.new_log_entry()

        with open(filename, "wb"):
            pickle.dump(self._all_data, f)
