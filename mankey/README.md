
# Manipulation based on Keypoints

The code in this directory is based on work by Wei Gao and Lucas Manuelli from another [GitHub repository](https://github.com/weigao95/mankey-ros). Modifications have been made to train a keypoint detection algorithm on data that shows a stick. There is also an improved version of the network that includes a LSTM to keep track of keypoints over multiple image frames, even if one of them is occluded.

## Folder Structure

- `config/`: Holds configuration data for training image processing and examples for files that list trainingsets.
- `dataproc/`: Data processing scripts. Read and preprocess the training data.
- `network/`: Code to build the network structure and the loss functions along with scripts for using a trained network for inference.
- `training/`: Training Scripts and some code to set the training up:
    - `heatmap_integral_symm.py`: Trains a keypoint detection network for a symmetric object with two keypoints, in this example a stick.
    - `heatmap_integral_symm_ts.py`: Trains a keypoint detection network on **t**ime **s**eries data.


## Training Config Files

Examples of the config files for a dataset are given in `config/*.txt`. These files contain a list of directories in the root folder of a dataset. This root folder is specified in `training/training_setup.py` as `db_config.pdc_data_root`.
For every training run, there is one config file listing training data and another one listing validation data. The data that is used is specified in the training script.
The files ending with `*_all.txt` are just there for the sake of completeness. They list the paths to all data from one dataset but don't make a distinction between what data is used for training or validation.

A folder of training data is then read like `<db_config.pdc_data_root>/<line of config/stick_time_sereis_train.txt>`, e.g. `/home/monti/Desktop/pdc/logs_proto/2021-02-26-15-02-41`.

For more information on how a training data set is structured and generated, refer to [`data_generation/README.md`](../data_generation/README.md). Download links for the training data sets listed in `config/` are also found there.


## Download Models

You can download the models that were trained on simulated images showing sticks.

- Single Frame: Model that does keypoint detection on a single input image. [Download](https://mega.nz/file/Mm5XFBgK#kqW35BxJduuuvjpPfUKMkmMnT9Eu3VBzMk6nXQ3ONGY)
- Time Series: Keypoint detection on several consecutive images by using an LSTM. [Download](https://mega.nz/file/xjJGULqJ#n_klogvMJRgS2Vtld6vGRsV8fuJApZ4tHdFRuusnry8)

Theses are the models used with the performance evaluation in the `evaluation/` subdirectory in this repository's root.
