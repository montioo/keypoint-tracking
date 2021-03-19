#!/usr/bin/env python3

# Learning and Intelligent Systems Project
# ---
# Functions that randomize the camera's viewing angle
#
# Marius Montebaur, WS20/21

import sys
sys.path.append("../third_party/rai/rai/ry")
import libry as ry
from math import radians, pi, sin, cos
from random import uniform, random
from common.utility import euler_to_quaternion


def set_random_camera_pose(C, vis=False, allow_camera_tilting=True, orbit_root="table", orbit_home=[0.3, 0.12, 0.01]):
    """
    Creates a frame for the position the camera should have.
    To understand the process of building this camera target position have a
    look at page 8 of my notes.
    # TODO: "page 8 of my notes" is not helpful for other people.

    :param vis: Display shapes in the simulation to show the location of the
    camera axis and other parameters.
    :param allow_camera_tilting: If set to true will allow the camera to view
    the table at an angle of up to 60°. Otherwise the view plane will be
    perpendicular to the table.
    :param orbit_root: The camera position will be built with this frame as a starting point.
    :param orbit_home: Translational offset from orbit_root frame.
    """

    # remove data generation (dg) frames that were created in previous runs.
    for frame_name in C.getFrameNames():
        if frame_name.startswith("dg_"):
            C.delFrame(frame_name)

    # see notes page 8 for description
    A_radius = 0.15
    # l_interv = [0.6, 0.8]
    l_interv = [0.6, 1.0]
    θ_interv = [0.0, radians(60)]  # deg
    ϕ_interv = [pi, 2*pi]

    # orbit_home = [0.3, .12, .01]  # in table frame
    orbit_area = C.addFrame(name="dg_orbit_area", parent=orbit_root)
    orbit_area.setRelativePosition(orbit_home)

    r_c, α_c = A_radius*random(), 1*pi*random()
    x_c, y_c = r_c*cos(α_c), -r_c*sin(α_c)

    orbit_center = C.addFrame(name="dg_orbit_center", parent="dg_orbit_area")
    orbit_center.setRelativePosition([x_c, y_c, 0.0])

    if allow_camera_tilting:
        r, θ, ϕ = uniform(*l_interv), uniform(*θ_interv), uniform(*ϕ_interv)
    else:
        r, θ, ϕ = uniform(*l_interv), 0.0, pi

    x, y, z = r*sin(θ)*cos(ϕ), r*sin(θ)*sin(ϕ), r*cos(θ)

    cam_goal = C.addFrame("dg_cam_goal", parent="dg_orbit_center")
    cam_goal.setRelativePosition([x, y, z])
    cam_goal.setRelativeQuaternion(euler_to_quaternion(0, θ, ϕ))

    if vis:
        cam_goal.setShape(ry.ST.cylinder, [0.9, 0.02])
        cam_goal.setColor([1, 1, 1])
        orbit_center.setShape(ry.ST.sphere, [0.02])
        orbit_center.setColor([0, 1, 1])
        # visualization of the current camera orientation
        orbit_area.setShape(ry.ST.cylinder, [0.002, A_radius*2])
        orbit_area.setColor([0, 0, 1])
        A_frame = C.addFrame(name="dg_A", parent="dg_orbit_area")
        A_frame.setShape(ry.ST.cylinder, [0.002, A_radius*2])
        A_frame.setColor([0, 0, 1])
        A_frame.setRelativePosition([0, 0, 0])

    cam_frame = C.frame("camera")
    cam_frame.setPosition(cam_goal.getPosition())
    cam_frame.setQuaternion(cam_goal.getQuaternion())

    for frame_name in C.getFrameNames():
        if frame_name.startswith("dg_"):
            C.delFrame(frame_name)