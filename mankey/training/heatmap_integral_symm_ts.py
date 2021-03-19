#!/usr/bin/env python3

# Learning and Intelligent Systems Project
# ---
# Training for the network on time series data. A keypoint detection network with a LSTM is used.
#
# This script is based on work by Wei Gao and Lucas Manuelli from https://github.com/weigao95/mankey-ros
# Marius Montebaur, WS20/21

import torch
import os
import sys
from torch.utils.data import DataLoader
from mankey.network.resnet_nostage import init_from_modelzoo
from mankey.network.weighted_loss import symmetric_loss_l1, symmetric_loss_l1_separate, sequence_symmetric_loss_l1
import mankey.network.predict as predict
import mankey.config.parameter as parameter
from mankey.utils.utility import get_date_string, Logger
from training_setup import construct_dataset, construct_network, segmented_img_size


def merge_first_two_dims(t):
    """ Will transform tensor of shape (a, b, c, ...) into (a*b, c, ...) """
    shape = tuple(t.shape)
    if len(shape) < 2:
        print("Warning: Not enough dims to merge the first two dims;", shape)
        return t
    new_shape = (shape[0] * shape[1],) + shape[2:]
    return t.view(*new_shape)

sys.stdout = Logger()

print("Training with:", __file__)


# Some global parameter
learning_rate = 2e-4
n_epoch = 100

# Training Data Paths (commented ones are from past training runs)
# stick on the table with parallel view and angled view:
# training_data_path = "/home/monti/Desktop/pdc/logs_proto/stick_samples_2021-01-30_train.txt"
# like previous one but with stick attached to gripper:
# training_data_path = "/home/monti/Desktop/pdc/logs_proto/timeseries_test_2021-02-01_train.txt"
training_data_path = "/home/monti/Desktop/pdc/logs_proto/stick_samples_2021-02-26_train.txt"
training_data_path = "../config/stick_time_series_train.txt"

# Validation Data Paths (commented ones are from past training runs)
# validation_data_path = "/home/monti/Desktop/pdc/logs_proto/stick_samples_2021-01-30_val.txt"
# validation_data_path = "/home/monti/Desktop/pdc/logs_proto/timeseries_test_2021-02-01_val.txt"
# validation_data_path = "/home/monti/Desktop/pdc/logs_proto/stick_samples_2021-02-26_val.txt"
validation_data_path = "../config/stick_time_series_val.txt"


def train(checkpoint_dir: str, start_from_ckpnt="", save_epoch_offset=0):
    global training_data_path, validation_data_path, learning_rate, n_epoch, segmented_img_size

    time_sequence_length = 5
    dataset_train, train_config = construct_dataset(True, training_data_path, time_sequence_length=time_sequence_length)
    dataset_val, val_config = construct_dataset(False, validation_data_path, time_sequence_length=time_sequence_length)

    loader_train = DataLoader(dataset=dataset_train, batch_size=4, shuffle=True, num_workers=4)
    loader_val = DataLoader(dataset=dataset_val, batch_size=4, shuffle=False, num_workers=1)

    network, net_config = construct_network(True)
    if start_from_ckpnt:
        # strict=False will allow the model to load the parameters from a
        # previously trained model that didn't use the LSTM extension. The
        # loaded model was still trained on keypoint detection for the same
        # object class but only learned on images without occluded keypoints.
        network.load_state_dict(torch.load(start_from_ckpnt), strict=False)
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
        if epoch % 10 == 0 and epoch > 0:
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

            # Those have the shape [batch_size, seq_length, ...]
            keypoint_xy_depth = merge_first_two_dims(data[parameter.keypoint_xyd_key])
            keypoint_weight = merge_first_two_dims(data[parameter.keypoint_validity_key])

            # Upload to GPU
            image = image.cuda()
            keypoint_xy_depth = keypoint_xy_depth.cuda()
            keypoint_weight = keypoint_weight.cuda()

            # To predict
            optimizer.zero_grad()
            # don't care about hidden states
            raw_pred, _ = network(image)
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
            loss = sequence_symmetric_loss_l1(xy_depth_pred, keypoint_xy_depth, keypoint_weight, sequence_length=5)
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
        sample_count = len(dataset_train) * 5
        print(f"The training averaged pixel error is (pixel in {s}x{s} image):", s * train_error_xy / sample_count)
        print('The training averaged depth error is (mm): ', train_config.depth_image_scale * train_error_depth / sample_count)


        # Evaluation for epochs
        network.eval()
        val_error_xy = 0
        val_error_depth = 0

        # The validation iteration of the data
        for idx, data in enumerate(loader_val):
            # Get the data
            image = data[parameter.rgbd_image_key]
            keypoint_xy_depth = merge_first_two_dims(data[parameter.keypoint_xyd_key])
            keypoint_weight = merge_first_two_dims(data[parameter.keypoint_validity_key])

            # Upload to GPU
            image = image.cuda()
            keypoint_xy_depth = keypoint_xy_depth.cuda()
            keypoint_weight = keypoint_weight.cuda()

            # Predict. Don't care about hidden states because the input data is already a sequence.
            raw_pred, _ = network(image)

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
        sample_count = len(dataset_val) * 5
        print(f"The averaged pixel error is (pixel in {s}x{s} image):", s * val_error_xy / sample_count)
        print("The averaged depth error is (mm):", val_config.depth_image_scale * val_error_depth / sample_count)

        print("-"*20)


if __name__ == '__main__':

    # === Training ===
    perform_training = True
    if perform_training:

        # Continue training from model without LSTM:
        # Started with weights from network that was trained without LSTM and
        # had no occluded keypoints. Can be downloaded as single frame model
        # (link in README.md)
        start_from_ckpnt = "ckpnt/lstm_2021-03-02_p2/checkpoint_0236.pth"
        save_epoch_offset = 236

        # Or: Train a new model
        # start_from_ckpnt = ""
        # save_epoch_offset = 0

        # Training checkpoint
        checkpoint_dir_name = f"ckpnt/stick_time_series"
        checkpoint_dir = os.path.join(os.path.dirname(__file__), checkpoint_dir_name)
        print("Checkpoint directory:", checkpoint_dir)

        print("Started at", get_date_string())

        train(checkpoint_dir=checkpoint_dir, start_from_ckpnt=start_from_ckpnt, save_epoch_offset=save_epoch_offset)

        print("Done by", get_date_string())


    # === Interactive testing ===

    dummy_stuff = False
    if dummy_stuff:

        time_sequence_length = 5
        dataset_train, train_config = construct_dataset(True, training_data_path, time_sequence_length=time_sequence_length)
        dataset_val, val_config = construct_dataset(False, validation_data_path, time_sequence_length=time_sequence_length)

        loader_train = DataLoader(dataset=dataset_train, batch_size=4, shuffle=True, num_workers=4)
        loader_val = DataLoader(dataset=dataset_val, batch_size=4, shuffle=False, num_workers=1)

        # Construct the regressor
        network, net_config = construct_network(True)
        start_from_ckpnt = False
        if start_from_ckpnt:
            network.load_state_dict(torch.load(start_from_ckpnt))
        else:
            init_from_modelzoo(network, net_config)
        # network.cuda()

        for data in loader_val:
            d = data
            break

        image = d["rgbd_image"]
