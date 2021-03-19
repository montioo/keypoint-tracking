

import numpy as np
import matplotlib.pyplot as plt
import json
from math import sqrt

# path = "/home/monti/git/marius-keypoints/examples/stick/trajectories/point_trajectory_24-02-21_12-53-25.json"
paths = [
    "/home/monti/git/marius-keypoints/evaluation/training_logs/trajectory_logs/point_trajectory_03-03-21_09-03-58.json",
    "/home/monti/git/marius-keypoints/evaluation/training_logs/trajectory_logs/point_trajectory_03-03-21_09-29-04.json"
]

for path in paths:
    with open(path, "r") as f:
        d = json.load(f)

    # truth = d["ground_truth"]
    # pred = d["predictions"]
    truth = np.array(d["ground_truth"])
    pred = np.array(d["predictions"])

    assert len(truth) == len(pred), "predictions and ground truth don't have the same length"

    pred_1, pred_2 = [], []
    truth_1, truth_2 = [], []

    for i in range(0, len(truth), 2):
        pred_1.append(pred[i])
        pred_2.append(pred[i+1])

        truth_1.append(truth[i+1])
        truth_2.append(truth[i])

    kp1_err, kp2_err = [], []
    kp_ids = list(range(len(pred_1)))

    for i in range(0, len(pred_1)):
        xp, yp, zp = pred_1[i]
        xt, yt, zt = truth_1[i]
        err1 = sqrt((xp - xt)**2 + (yp - yt)**2 + (zp - zt)**2)
        kp1_err.append(err1 * 100)

        xp, yp, zp = pred_2[i]
        xt, yt, zt = truth_2[i]
        err2 = sqrt((xp - xt)**2 + (yp - yt)**2 + (zp - zt)**2)
        kp2_err.append(err2 * 100)


    # truth = truth.reshape((-1, 2, 3))
    # truth[:, 0, :], truth[:, 1, :] = truth[:, 1, :], truth[:, 0, :]
    # print(truth.shape)
    # pred = pred.reshape((-1, 2, 3))
    # print(pred.shape)
    #
    # err = np.linalg.norm(truth - pred, axis=-1)
    # print(err.shape)
    #
    # kp_count = err.shape[0]
    # print(kp_count)
    # kp_ids = list(range(kp_count))

    # plt.plot(kp_ids, err[:, 0])
    # plt.plot(kp_ids, err[:, 1])

    fig, ax = plt.subplots(1, 1, figsize=(6, 2.5))
    # fig.suptitle("Training and validation loss")

    # ax.plot(kp_ids, err[:, 0], label="position error keypoint 1")
    # ax.plot(kp_ids, err[:, 1], label="position error keypoint 2")
    ax.plot(kp_ids, kp1_err, label="position error keypoint 1")
    ax.plot(kp_ids, kp2_err, label="position error keypoint 2")

    ax.legend()

    ax.set(xlabel="frame", ylabel="keypoint position error [cm]")
    ax.set_title("3D keypoint prediction performance")

    plt.tight_layout()
    plt.savefig(path + ".pdf")

    plt.show()