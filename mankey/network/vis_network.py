#!/usr/bin/env python3

# Learning and Intelligent Systems Project
# ---
# Inspect networks, their size and visualize them. Visualization of the
# network's output and testing its performance is done with scripts located in
# the evaluation directory.
#
# Marius Montebaur, WS20/21

import time
import torch
from torchviz import make_dot, make_dot_from_trace
from resnet_nostage import ResnetNoStageConfig, ResnetNoStage, init_from_modelzoo, ResnetNoStageLSTM
from torchsummaryX import summary


def construct_network(for_timeseries_data=True):
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


def print_network_summary(network, input_shape):
    inp = torch.zeros(input_shape)
    summary(network, inp)


def visualize_network_architecture(input_shape, pdf_filename=None):
    inp = torch.zeros(*input_shape, dtype=torch.float, requires_grad=False)
    out, hidden = net(inp)
    dot = make_dot(out)
    dot.format = "pdf"
    dot.render(pdf_filename)


net, conf = construct_network(True)
input_shape = (4, 5, 4, 128, 128)
