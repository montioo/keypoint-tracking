#!/bin/bash

# Adds this repository's root to the python path to make mankey importable
# Run this script with $ source set_pythonpath.sh
export PYTHONPATH=`pwd`":${PYTHONPATH}"

