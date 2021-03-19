#!/usr/bin/env python3

# Learning and Intelligent Systems Project
# ---
# Inspect networks and their size
#
# Marius Montebaur, WS20/21


import time
import torch
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




net, conf = construct_network(True)

inp = torch.zeros((4, 5, 4, 128, 128))

summary(net, inp)

# out_path = "normal_net.pth"
# torch.save(net.state_dict(), out_path)
#
#
# net, conf = construct_network(True)
# inp = torch.zeros((4, 5, 4, 128, 128))
#
# start = time.time()
# pred, h = net(inp)
# print("dur", time.time() - start)
# print(pred.shape)
# print(h[0].shape)
# print(h[1].shape)
#
# print('---')
# pred2, h2 = net(inp, (h[0], h[1]))
# print(pred2.shape)
# print(h2[0].shape)
# print(h2[1].shape)
#
# out_path = "lstm_net.pth"
# torch.save(net.state_dict(), out_path)
