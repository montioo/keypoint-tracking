#!/usr/bin/env python3

# Learning and Intelligent Systems Project
# ---
# Small function doing inverse kinematics calculations.
#
# Marius Montebaur, WS20/21

import sys
sys.path.append("../third_party/rai/rai/ry")
import libry as ry
import numpy as np
import time


def plan_komo_path(C, obj_goal="goal", arm="endeffR", steps=50, duration=1.5):
    """
    Plans a path to the specified location.
    :param obj_goal: Location to reach. Has to be a frame (and I think it has to have a shape?)
    :param arm: Frame of the arm that should reach the goal location
    :param steps: Number of steps in the resulting path
    :param duration: Duration this path's execution should take
    """

    # phases: float, stepsPerPhase: int = 20, timePerPhase: float = 5.0, useSwift: bool
    step_duration = duration / steps
    komo = C.komo_path(1., steps, step_duration, True)
    # komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq)
    komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq)
    komo.addObjective([1.], ry.FS.distance, [arm, obj_goal], ry.OT.eq, [1e2])
    komo.addObjective([.9, duration], ry.FS.poseDiff, [arm, obj_goal], ry.OT.sos)
    komo.addObjective(time=[1.0], feature=ry.FS.qItself, type=ry.OT.eq, order=1)

    komo.optimize()

    return komo


def execute_komo_path(C, komo, steps=50, duration=1.5, start_state=None):
    step_duration = duration / steps
    if start_state:
        C.setFrameState(start_state)

    for i in range(steps):
        step = komo.getConfiguration(i)
        C.setFrameState(step)
        time.sleep(step_duration * 2)


def set_komo_inv_kin(C, goal_frame="goal", EE_frame="R_gripper"):
    """
    Will try to reach the goal frame's position with the EE_frame. If this is
    not possible, the algorithm with still try to get as close as possible.
    """
    # IK engine. Goal is to make pos diff zero
    IK = C.komo_IK(False)
    IK.verbose(0)

    IK.addObjective(type=ry.OT.eq, feature=ry.FS.positionDiff, frames=[EE_frame, goal_frame])
    IK.addObjective(type=ry.OT.eq, feature=ry.FS.accumulatedCollisions)
    IK.addObjective(type=ry.OT.eq, feature=ry.FS.jointLimits)

    IK.optimize()

    C.setFrameState( IK.getConfiguration(0) )