""" A set of robotics control functions """

import numpy as np


def reactive_obst_avoid(lidar):
    """
    Simple obstacle avoidance
    lidar : placebot object with lidar data
    """
    # TP1 done

    # distances in cm a priori
    default_speed = 0.3

    # thrshold of wall detection
    threshold = 50  # TO ADAPT ACCORDING TO default_speed
    obst_avoid_angular_speed = 0.8

    # useful indexes and values
    half_nb_values = lidar.get_sensor_values().shape[0] // 2
    half_idx_range = int(0.25 * half_nb_values)

    # if wall to the right
    if lidar.get_sensor_values()[half_nb_values - half_idx_range:half_nb_values].min() < threshold:
        command = {"forward": 0, "rotation": obst_avoid_angular_speed}
    # if wall to the left
    elif lidar.get_sensor_values()[half_nb_values:half_nb_values + half_idx_range].min() < threshold:
        command = {"forward": 0, "rotation": -obst_avoid_angular_speed}
    # no wall under threshold
    else:
        command = {"forward": default_speed, "rotation": 0}

    return command


def potential_field_control(lidar, pose, goal):
    """
    Control using potential field for goal reaching and obstacle avoidance
    lidar : placebot object with lidar data
    pose : [x, y, theta] nparray, current pose in odom or world frame
    goal : [x, y, theta] nparray, target pose in odom or world frame
    """
    # TODO for TP2

    command = {"forward": 0, "rotation": 0}

    return command
