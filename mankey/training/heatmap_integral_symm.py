#!/usr/bin/env python3

# Learning and Intelligent Systems Project
# ---
# Training for the keypoint detection network.
#
# This script is based on work by Wei Gao and Lucas Manuelli from https://github.com/weigao95/mankey-ros
# Marius Montebaur, WS20/21

import torch
import os
import sys
from torch.utils.data import DataLoader
from mankey.network.resnet_nostage import init_from_modelzoo
from mankey.network.weighted_loss import symmetric_loss_l1,  symmetric_loss_l1_separate
import mankey.network.predict as predict
import mankey.config.parameter as parameter
from mankey.utils.utility import get_date_string, Logger
from training_setup import construct_dataset, construct_network, segmented_img_size


sys.stdout = Logger()

print("Training with:", __file__)


# Some global parameter
learning_rate = 2e-4
n_epoch = 120

# Training Data Paths (commented ones are from past training runs)
# stick on the table with parallel view and angled view:
# training_data_path = "/home/monti/Desktop/pdc/logs_proto/stick_samples_2021-01-30_train.txt"
# like previous one but with stick attached to gripper:
# training_data_path = "/home/monti/Desktop/pdc/logs_proto/stick_samples_2021-02-01_train.txt"
training_data_path = "../config/stick_single_image_train.txt"

# Validation Data Paths (commented ones are from past training runs)
# validation_data_path = "/home/monti/Desktop/pdc/logs_proto/stick_samples_2021-01-30_val.txt"
# validation_data_path = "/home/monti/Desktop/pdc/logs_proto/stick_samples_2021-02-01_val.txt"
validation_data_path = "../config/stick_single_image_val.txt"


def train(checkpoint_dir: str, start_from_ckpnt="", save_epoch_offset=0):
    global training_data_path, validation_data_path, learning_rate, n_epoch, segmented_img_size

    dataset_train, train_config = construct_dataset(True, training_data_path)
    dataset_val, val_config = construct_dataset(False, validation_data_path)

    loader_train = DataLoader(dataset=dataset_train, batch_size=16, shuffle=True, num_workers=16)
    loader_val = DataLoader(dataset=dataset_val, batch_size=16, shuffle=False, num_workers=4)

    network, net_config = construct_network()
    if len(start_from_ckpnt) > 0:
        network.load_state_dict(torch.load(start_from_ckpnt))
    else:
        init_from_modelzoo(network, net_config)
    network.cuda()

    # The checkpoint
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    # The optimizer and scheduler
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [60, 90], gamma=0.1)

    # The training loop
    for epoch in range(n_epoch):
        # Save the network
        if epoch % 5 == 0 and epoch > 0:
            file_name = f"checkpoint_{(epoch + save_epoch_offset):04d}.pth"
            checkpoint_path = os.path.join(checkpoint_dir, file_name)
            print('Save the network at %s' % checkpoint_path)
            torch.save(network.state_dict(), checkpoint_path)

        # Prepare info for training
        network.train()
        train_error_xy = 0
        train_error_depth = 0

        # The learning rate step
        scheduler.step()
        for param_group in optimizer.param_groups:
            print('The learning rate is ', param_group['lr'])

        # The training iteration over the dataset
        for idx, data in enumerate(loader_train):
            # Get the data
            image = data[parameter.rgbd_image_key]
            keypoint_xy_depth = data[parameter.keypoint_xyd_key]
            keypoint_weight = data[parameter.keypoint_validity_key]

            # Upload to GPU
            image = image.cuda()
            keypoint_xy_depth = keypoint_xy_depth.cuda()
            keypoint_weight = keypoint_weight.cuda()

            # To predict
            optimizer.zero_grad()
            raw_pred = network(image)
            prob_pred = raw_pred[:, 0:net_config.num_keypoints, :, :]
            depthmap_pred = raw_pred[:, net_config.num_keypoints:, :, :]
            heatmap = predict.heatmap_from_predict(prob_pred, net_config.num_keypoints)
            _, _, heatmap_height, heatmap_width = heatmap.shape

            # Compute the coordinate
            coord_x, coord_y = predict.heatmap2d_to_normalized_imgcoord_gpu(heatmap, net_config.num_keypoints)
            depth_pred = predict.depth_integration(heatmap, depthmap_pred)

            # Concantate them
            xy_depth_pred = torch.cat((coord_x, coord_y, depth_pred), dim=2)

            # Compute loss
            loss = symmetric_loss_l1(xy_depth_pred, keypoint_xy_depth, keypoint_weight)
            loss.backward()
            optimizer.step()

            # cleanup
            del loss

            # Log info
            xy_loss, depth_loss = symmetric_loss_l1_separate(xy_depth_pred, keypoint_xy_depth, keypoint_weight)

            xy_error = float(xy_loss.item())
            depth_error = float(depth_loss.item())
            if idx % 400 == 0:
                print('Iteration %d in epoch %d' % (idx, epoch))
                s = segmented_img_size
                print(f"The averaged pixel error is (pixel in {s}x{s} image):", segmented_img_size * xy_error / len(xy_depth_pred))
                print("The averaged depth error is (mm):", train_config.depth_image_scale * depth_error / len(xy_depth_pred))

            # Update info
            train_error_xy += float(xy_error)
            train_error_depth += float(depth_error)

        # The info at epoch level
        print('Epoch %d' % epoch)
        s = segmented_img_size
        print(f"The training averaged pixel error is (pixel in {s}x{s} image):", s * train_error_xy / len(dataset_train))
        print('The training averaged depth error is (mm): ', train_config.depth_image_scale * train_error_depth / len(dataset_train))


        # Evaluation for epochs
        network.eval()
        val_error_xy = 0
        val_error_depth = 0

        # The validation iteration of the data
        for idx, data in enumerate(loader_val):
            # Get the data
            image = data[parameter.rgbd_image_key]
            keypoint_xy_depth = data[parameter.keypoint_xyd_key]
            keypoint_weight = data[parameter.keypoint_validity_key]

            # Upload to GPU
            image = image.cuda()
            keypoint_xy_depth = keypoint_xy_depth.cuda()
            keypoint_weight = keypoint_weight.cuda()

            # To predict
            raw_pred = network(image)

            prob_pred = raw_pred[:, 0:net_config.num_keypoints, :, :]
            depthmap_pred = raw_pred[:, net_config.num_keypoints:, :, :]
            heatmap = predict.heatmap_from_predict(prob_pred, net_config.num_keypoints)
            _, _, heatmap_height, heatmap_width = heatmap.shape

            # Compute the coordinate
            coord_x, coord_y = predict.heatmap2d_to_normalized_imgcoord_gpu(heatmap, net_config.num_keypoints)
            depth_pred = predict.depth_integration(heatmap, depthmap_pred)

            # Concantate them
            xy_depth_pred = torch.cat((coord_x, coord_y, depth_pred), dim=2)

            # Log info
            xy_loss, depth_loss = symmetric_loss_l1_separate(xy_depth_pred, keypoint_xy_depth, keypoint_weight)

            # Update info
            val_error_xy += float(xy_error)
            val_error_depth += float(depth_error)

        print('Validation for epoch', epoch)
        s = segmented_img_size
        print(f"The averaged pixel error is (pixel in {s}x{s} image):", s * val_error_xy / len(dataset_val))
        print("The averaged depth error is (mm):", val_config.depth_image_scale * val_error_depth / len(dataset_val))

        print("-"*20)


if __name__ == '__main__':

    # === Training ===

    # Not starting from a previously trained keypoint detection network.
    # Example to continue training from checkpoint found in
    # heatmap_integral_symm_ts.py
    start_from_ckpnt = ""
    save_epoch_offset = 0

    checkpoint_dir_name = f"ckpnt/stick_single_frame"
    checkpoint_dir = os.path.join(os.path.dirname(__file__), checkpoint_dir_name)
    print("Checkpoint directory:", checkpoint_dir)

    print("Started at", get_date_string())

    train(checkpoint_dir=checkpoint_dir, start_from_ckpnt=start_from_ckpnt, save_epoch_offset=save_epoch_offset)

    print("Done by", get_date_string())

