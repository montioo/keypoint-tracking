#!/usr/bin/env python3

# Learning and Intelligent Systems Project
# ---
# Contact the bounding box inference server (i.e. docker container because the
# bounding box framework uses a very old pytorch version) and receive the
# coordinates of the box(es).
#
# Marius Montebaur, WS20/21

import numpy as np
import socket
import time
import pickle
from PIL import Image


def socket_client(img):

    PORT = 8134
    host = "192.168.4.25"

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(2)
        try:
            s.connect((host, PORT))
        except socket.timeout:
            print("connection timed out")
            # s.settimeout(None)
            return [], None
        except ConnectionRefusedError:
            print("Connection refused")
            return [], None
        finally:
            s.settimeout(None)

        s.settimeout(None)

        img_pickle = pickle.dumps(img)
        s.send(img_pickle)

        resp = s.recv(1024)

        # mode refers to the order in which bounding box pixels are stored
        # usually with this setup: "xyxy" => left_x, upper_y, right_x, lower_y
        coord, mode = pickle.loads(resp)

    return coord, mode


if __name__ == "__main__":
    # One image to test the connection
    image = Image.open("/home/monti/Desktop/00000.png")
    img_np = np.array(image)

    # RGB to BGR
    img_np[:, :, 0], img_np[:, :, 2] = img_np[:, :, 2], img_np[:, :, 0]

    socket_client(img_np)
