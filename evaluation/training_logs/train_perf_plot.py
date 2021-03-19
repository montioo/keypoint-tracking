#!/usr/bin/env python3

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import sys

# example logfile name for keypoint training
# training_log_filename = "keypoint_training_2021-01-30_20:48:33.log"

if len(sys.argv) < 2:
    print(f"Usage: python3 {sys.args[0]} <logfile_name>")
    exit()

training_log_filename = sys.argv[1]

# if not training_log_filename.startswith("keypoint_training_"):
#     print("Currently only logfiles from keypoint training supported")
#     exit()


with open(training_log_filename) as f:
    lines = f.readlines()

training_error_xy = []
training_error_depth = []
validation_error_xy = []
validation_error_depth = []

for i in range(len(lines)):
    if not lines[i].startswith("----"):
        continue

    def val_from_line(line):
        return float(line.split(":")[-1].strip())

    training_error_xy.append(val_from_line(lines[i-5]))
    training_error_depth.append(val_from_line(lines[i-4]))

    validation_error_xy.append(val_from_line(lines[i-2]))
    validation_error_depth.append(val_from_line(lines[i-1]))


epochs = list(range(len(training_error_xy)))

use_pyplot = True

if use_pyplot:
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2.5))
    # fig.suptitle("Training and validation loss")

    ax1.plot(epochs, training_error_xy, label="training error xy")
    ax1.plot(epochs, validation_error_xy, label="validation error xy")
    ax1.legend()
    ax1.set(xlabel="Epochs", ylabel="Error [pixel]")
    ax1.set_title("Pixel Position Performance")

    ax2.plot(epochs, training_error_depth, label="training error depth")
    ax2.plot(epochs, validation_error_depth, label="validation error depth")
    ax2.set(xlabel="Epochs", ylabel="Error [mm]")
    ax2.legend()
    ax2.set_title("Depth Prediction Performance")

    plt.tight_layout()
    plt.savefig(training_log_filename + ".pdf")

    plt.show()

else:
    fig = make_subplots(rows=1, cols=2)

    fig.add_trace(
        go.Scatter(x=epochs, y=training_error_xy, name="training error xy"),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=validation_error_xy, name="validation error xy"),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=epochs, y=training_error_depth, name="training error depth"),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=validation_error_depth, name="validation error depth"),
        row=1, col=2
    )

    fig.update_layout(title_text="Side By Side Subplots")
    fig.show()

