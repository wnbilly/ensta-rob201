""" A set of robotics control functions """

import numpy as np

import math


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


def closest_index_to(array, value):
    return np.argmin(np.abs(array - value))


def wall_follow(lidar):  # TODO finish it
    """
    Wall follow
    lidar : placebot object with lidar data
    """

    # parameters based on https://drive.google.com/file/d/1tzyfGYq3JjvLlYHIiSTyvoNq4kcPf53n/view
    theta = np.deg2rad(20)
    dist_goal = 50

    ninety_deg_idx = closest_index_to(lidar.get_ray_angles(), -np.pi / 2)
    ninety_minus_theta_idx = closest_index_to(lidar.get_ray_angles(), -np.pi / 2 + theta)

    a = lidar.get_sensor_values()[ninety_minus_theta_idx]
    b = lidar.get_sensor_values()[ninety_deg_idx]

    alpha = np.arctan((a * np.cos(theta) - b) / (a * np.sin(theta)))

    Dt = b * np.cos(alpha)
    dist_error = dist_goal - Dt

    Kp = 0.001
    # TODO alpha semble correct, Dt non dès qu'on sort d'un mur droit. Le robot semble détecter en face à droite
    steering_angle = min(1, Kp * dist_error) if Kp * dist_error >= 0 else max(-1, Kp * dist_error)
    print(alpha, Dt)

    if 0 < dist_error < 0.17453292519943295:  # entre 0 et 10 degrées
        speed = 1
    if 0.17453292519943295 < dist_error < 0.3490658503988659:  # entre 10 et 20 degrées
        speed = 0.7
    else:
        speed = 0.3

    command = {"forward": speed, "rotation": steering_angle}

    return command


def potential_field_control(lidar, pose, goal):
    """
    Control using potential field for goal reaching and obstacle avoidance
    lidar : placebot object with lidar data
    pose : [x, y, theta] nparray, current pose in odom or world frame
    goal : [x, y, theta] nparray, target pose in odom or world frame
    """

    # Params
    K_goal = 1  # Gain du gradient attractif
    K_ang = 0.4  # Gain en vitesse angulaire
    K_vit = 0.2  # Gain en vitesse linéaire
    d_min = 50  # Distance au goal de transition entre gradient conique et quadratique
    d_stop = 20  # Distance au goal pour arrêt
    v_max = 0.4  # Doit être <= 1

    K_obs = 10 ** 4.5  # Gain du potentiel répulsif
    d_safe = 30  # Distance à l'obstacle à partir de laquelle le potentiel répulsif est nul

    # Normalisation de l'angle du robot entre -pi et pi
    # Cela permet notamment au robot de tourner du côté le plus rapide pour atteindre angle_goal lorsqu'il a atteint son goal en x et y mais pas en theta
    pose[2] = math.atan2(math.sin(pose[2]), math.cos(pose[2]))
    # Distance de pose à goal
    d_goal = np.linalg.norm(goal[:-1] - pose[:-1])

    # Condition d'arrêt à proximité de l'objectif
    if d_goal < d_stop:
        # Si robot, assez proche, on tourne le robot jusqu'à atteindre le theta voulu
        K_ang = 8  # Gain pour la vitesse angulaire plus élevé lorsque le robot tourne sur lui-même
        v_theta = K_goal * (goal[-1] - pose[-1]) / max(d_min, d_goal)
        return {"forward": 0, "rotation": np.sign(v_theta) * min(1, np.abs(K_ang * v_theta))}

    # Obtention de la distance à l'obstacle le plus proche
    d_obs = lidar.get_sensor_values().min()
    # Indice de la distance à l'obstacle le plus proche
    d_obs_idx = np.argmin(lidar.get_sensor_values())
    # Angle entre la direction du robot et l'obstacle le plus proche
    theta_obs = lidar.get_ray_angles()[d_obs_idx]
    # Calcul des coordonnées de l'obstacle le plus proche
    obs = pose[:-1] + np.array([d_obs * np.cos(theta_obs + pose[2]), d_obs * np.sin(theta_obs + pose[2])])

    # Calcul du gradient répulsif de l'obstacle le plus proche
    rep_grad = K_obs / (d_obs ** 3) * (1 / d_obs - 1 / d_safe) * (pose[:-1] - obs) if d_obs <= d_safe else np.zeros(2)

    # Calcul de gradient attractif hybride (conique loin du goal et quadratique proche du goal)
    att_grad = K_goal * (goal[:-1] - pose[:-1]) / max(d_min, d_goal)

    # Calcul du potentiel total
    potential = att_grad + rep_grad
    # Extraction des composantes du vecteur à suivre pour se rapprocher de goal en fonction du potentiel
    vx, vy = potential[0], potential[1]

    # Angle vers lequel le robot doit aller pour suivre le gradient et se rapprocher de goal
    angular_direction = math.atan2(vy, vx)  # atan2 gère le cas vx = 0

    # Commande en vitesse angulaire proportionnelle à l'erreur angulaire
    angular_error = angular_direction - pose[2]

    # Pour garder angular_error entre -pi et pi, évite de tourner en rond
    angular_error = math.atan2(math.sin(angular_error), math.cos(angular_error))

    # Commandes cappées entre -1 et 1 sinon erreur
    angular_speed = np.sign(angular_error) * min(1, np.abs(K_ang * angular_error))
    linear_speed = min(v_max, K_vit * d_goal)  # Cappée à v_max

    return {"forward": linear_speed, "rotation": angular_speed}
