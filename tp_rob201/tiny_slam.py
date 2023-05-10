""" A simple robotics navigation code including SLAM, exploration, planning"""

import heapq
import pickle

import cv2
import numpy as np
from matplotlib import pyplot as plt


def heuristic(a, b):
    """
    Heuristic function for A* algorithm
    Parameters
    ----------
    a : point [x,y] or [x,y,theta]
    b : point [x,y] or [x,y,theta]

    Returns
    -------
    euclidian distance between a and b
    """
    # a[:2] and b[:2] are used so as to get rid of the theta if [x,y,theta] is given
    return np.linalg.norm(b[:2] - a[:2])


class TinySlam:
    """Simple occupancy grid SLAM"""

    def __init__(self, x_min, x_max, y_min, y_max, resolution):
        # Given : constructor
        self.x_min_world = x_min
        self.x_max_world = x_max
        self.y_min_world = y_min
        self.y_max_world = y_max
        self.resolution = resolution

        self.x_max_map, self.y_max_map = self._conv_world_to_map(
            self.x_max_world, self.y_max_world)

        self.occupancy_map = np.zeros(
            (int(self.x_max_map), int(self.y_max_map)))

        # Origin of the odom frame in the map frame
        self.odom_pose_ref = np.array([0, 0, 0])

    def _conv_world_to_map(self, x_world, y_world):
        """
        Convert from world coordinates to map coordinates (i.e. cell index in the grid map)
        x_world, y_world : list of x and y coordinates in m
        """
        x_map = (x_world - self.x_min_world) / self.resolution
        y_map = (y_world - self.y_min_world) / self.resolution

        if isinstance(x_map, float):
            x_map = int(x_map)
            y_map = int(y_map)
        elif isinstance(x_map, np.ndarray):
            x_map = x_map.astype(int)
            y_map = y_map.astype(int)

        return x_map, y_map

    def _conv_map_to_world(self, x_map, y_map):
        """
        Convert from map coordinates to world coordinates
        x_map, y_map : list of x and y coordinates in cell numbers (~pixels)
        """
        x_world = self.x_min_world + x_map * self.resolution
        y_world = self.y_min_world + y_map * self.resolution

        if isinstance(x_world, np.ndarray):
            x_world = x_world.astype(float)
            y_world = y_world.astype(float)

        return x_world, y_world

    def add_map_line(self, x_0, y_0, x_1, y_1, val):
        """
        Add a value to a line of points using Bresenham algorithm, input in world coordinates
        x_0, y_0 : starting point coordinates in m
        x_1, y_1 : end point coordinates in m
        val : value to add to each cell of the line
        """

        # convert to pixels
        x_start, y_start = self._conv_world_to_map(x_0, y_0)
        x_end, y_end = self._conv_world_to_map(x_1, y_1)

        if x_start < 0 or x_start >= self.x_max_map or y_start < 0 or y_start >= self.y_max_map:
            return

        if x_end < 0 or x_end >= self.x_max_map or y_end < 0 or y_end >= self.y_max_map:
            return

        # Bresenham line drawing
        dx = x_end - x_start
        dy = y_end - y_start
        is_steep = abs(dy) > abs(dx)  # determine how steep the line is
        if is_steep:  # rotate line
            x_start, y_start = y_start, x_start
            x_end, y_end = y_end, x_end
        # swap start and end points if necessary and store swap state
        if x_start > x_end:
            x_start, x_end = x_end, x_start
            y_start, y_end = y_end, y_start
        dx = x_end - x_start  # recalculate differentials
        dy = y_end - y_start  # recalculate differentials
        error = int(dx / 2.0)  # calculate error
        y_step = 1 if y_start < y_end else -1
        # iterate over bounding box generating points between start and end
        y = y_start
        points = []
        for x in range(x_start, x_end + 1):
            coord = [y, x] if is_steep else [x, y]
            points.append(coord)
            error -= abs(dy)
            if error < 0:
                y += y_step
                error += dx
        points = np.array(points).T

        # add value to the points
        self.occupancy_map[points[0], points[1]] += val

    def add_map_points(self, points_x, points_y, val):
        """
        Add a value to an array of points, input coordinates in meters
        points_x, points_y :  list of x and y coordinates in m
        val :  value to add to the cells of the points
        """
        x_px, y_px = self._conv_world_to_map(points_x, points_y)

        select = np.logical_and(np.logical_and(x_px >= 0, x_px < self.x_max_map),
                                np.logical_and(y_px >= 0, y_px < self.y_max_map))
        x_px = x_px[select]
        y_px = y_px[select]

        self.occupancy_map[x_px, y_px] += val

    def score(self, lidar, pose):
        """
        Computes the sum of log probabilities of laser end points in the map
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, position of the robot to evaluate, in world coordinates
        """
        # TODO for TP4

        # Suppression des points à la distance max du laser
        range_values = lidar.get_sensor_values()[lidar.get_sensor_values() < lidar.max_range]
        angle_values = lidar.get_ray_angles()[lidar.get_sensor_values() < lidar.max_range]

        # Positions des points détectés par le lidar
        lidar_global = np.zeros((2, range_values.shape[0]))
        # X positions of obstacles
        lidar_global[0] = pose[0] + np.cos(
            angle_values + pose[2]) * range_values
        # Y positions of obstacles
        lidar_global[1] = pose[1] + np.sin(
            angle_values + pose[2]) * range_values

        # Conversion des positions dans le repère de la carte
        x_map_raw, y_map_raw = self._conv_world_to_map(lidar_global[0], lidar_global[1])

        # Suppression des points hors carte
        x_map = [x for x, y in zip(x_map_raw, y_map_raw) if
                 0 <= x < self.x_max_map and 0 <= y < self.y_max_map]
        y_map = [y for x, y in zip(x_map_raw, y_map_raw) if
                 0 <= x < self.x_max_map and 0 <= y < self.y_max_map]

        # Return du score
        return np.sum(self.occupancy_map[x_map, y_map])

    def get_corrected_pose(self, odom, odom_pose_ref=None):
        """
        Compute corrected pose in map frame from raw odom pose + odom frame pose,
        either given as second param or using the ref from the object
        odom : raw odometry position
        odom_pose_ref : optional, origin of the odom frame if given,
                        use self.odom_pose_ref if not given
        """
        # TODO for TP4

        if odom_pose_ref is None:
            odom_pose_ref = self.odom_pose_ref

        corrected_pose = np.zeros(3)

        corrected_pose[0] = np.cos(odom_pose_ref[2]) * odom_pose_ref[0] - np.sin(odom_pose_ref[2]) * odom_pose_ref[1] + odom[0]
        corrected_pose[1] = np.sin(odom_pose_ref[2]) * odom_pose_ref[1] + np.cos(odom_pose_ref[2]) * odom_pose_ref[0] + odom[1]
        corrected_pose[2] = odom_pose_ref[2] + odom[2]

        return corrected_pose

    def localise(self, lidar, odom):
        """
        Compute the robot position wrt the map, and updates the odometry reference
        lidar : placebot object with lidar data
        odom : [x, y, theta] nparray, raw odometry position
        """
        # TODO for TP4

        # Params
        nb_wout_impr = 0  # Nombre de tirages sans améliorations
        N = 30  # Seuil de tirages sans améliorations qui arrête la boucle
        sigma_xy = 2  # Sigma pour les offsets des positions x et y
        sigma_theta = 0.2  # Sigma pour l'offset de theta

        best_score = self.score(lidar, self.get_corrected_pose(odom))

        while nb_wout_impr < N:
            # Tirage aléatoire des offsets
            xy_offset = np.random.normal(0, sigma_xy, 2)
            theta_offset = np.random.normal(0, sigma_theta, 1)

            # Addition des offsets à l'odom courante
            current_odom_ref = self.odom_pose_ref + np.concatenate((xy_offset, theta_offset))
            current_score = self.score(lidar, self.get_corrected_pose(odom, odom_pose_ref=current_odom_ref))
            # print(f"best score : {best_score:.2f}   current score : {current_score:.2f}")
            if current_score > best_score:  # Si on a trouvé une meilleure odom_ref
                best_score = current_score
                # MàJ de odom_pose_ref avec la meilleure position de référence de l'odométrie calculée
                self.odom_pose_ref = current_odom_ref
                nb_wout_impr = 0
            else:  # Si la boucle n'améliore plus le résultat
                nb_wout_impr += 1

        return best_score

    def update_map(self, lidar, pose):
        """
        Bayesian map update with new observation
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, corrected pose in world coordinates
        """
        # TP3
        # Parameters
        DIST_OF_PROX_REGION = 5 # Distance à l'obstacle à partir de laquelle la valeur d'occupation est next_occ_val
        MAP_VAL_LIMIT = 100
        occ_pt_prob, next_occ_pt_prob, empty_pt_prob = 0.95, 0.5, 0.2

        # Valeur d'occupation d'un point occupé
        occ_val = np.log(occ_pt_prob / (1 - occ_pt_prob))
        # Valeur d'occupation d'un point adjacent à un point occupé
        next_occ_val = np.log(next_occ_pt_prob / (1 - next_occ_pt_prob)) # Ca fait 0
        # Valeur d'occupation d'un point vide
        empty_val = np.log(empty_pt_prob / (1 - empty_pt_prob))

        # Conversion de polaire à local
        # Calcul des positions des obstacles
        obstacles = np.zeros((2, lidar.get_sensor_values().shape[0]))
        obstacles[0] = pose[0] + np.cos(
            lidar.get_ray_angles() + pose[2]) * lidar.get_sensor_values()
        obstacles[1] = pose[1] + np.sin(
            lidar.get_ray_angles() + pose[2]) * lidar.get_sensor_values()
        # Calcul des points d'entrée dans la zone à proximité de l'obstacle
        prox_region = np.zeros((2, lidar.get_sensor_values().shape[0]))
        prox_region[0] = pose[0] + np.cos(
            lidar.get_ray_angles() + pose[2]) * (lidar.get_sensor_values() - DIST_OF_PROX_REGION)
        prox_region[1] = pose[1] + np.sin(
            lidar.get_ray_angles() + pose[2]) * (lidar.get_sensor_values() - DIST_OF_PROX_REGION)

        # Map update
        # add_map_points et add_map_line vérifient déjà que les points sont dans la map
        for i in range(obstacles.shape[1]):
            # Ajout des points vides
            self.add_map_line(pose[0], pose[1], prox_region[0][i], prox_region[1][i],
                              empty_val)
            # Ajout des points adjacents aux points occupés
            self.add_map_line(prox_region[0][i], prox_region[1][i], obstacles[0][i], obstacles[1][i],
                              next_occ_val)

        # Ajout des points occupés
        self.add_map_points(obstacles[0], obstacles[1], occ_val)

        # Limitation des valeurs de la map pour une meilleure visualisation et un seuil pour update la map plus simple à choisir
        self.occupancy_map[self.occupancy_map < -MAP_VAL_LIMIT] = -MAP_VAL_LIMIT
        self.occupancy_map[self.occupancy_map > MAP_VAL_LIMIT] = MAP_VAL_LIMIT

    def get_neighbors(self, current):
        """
        Returns the neighbors of the current cell
        Parameters
        ----------
        current : point occrodinates to find the neigbors of
        Returns
        -------
        valid neighbors of the current point as a list of [x, y] coordinates
        """
        # Parameters
        OCCUPIED_THRESHOLD = 10  # Threshold to consider a cell occupied

        # Get sides and diagonals of the current cell
        neighbors = [[current[0] + x_offset, current[1] + y_offset] for x_offset in range(-1, 2) for y_offset in
                     range(-1, 2) if (x_offset != 0 or y_offset != 0)]

        #  Filter out cells outside the map
        neighbors = [neighbor for neighbor in neighbors if
                     0 <= neighbor[0] < self.y_max_map and 0 <= neighbor[1] < self.y_max_map]

        # Filter out occupied cells
        # No-data cells (cells with occupancy = 0) should be treated but as I don't have an exploration method, I didn't implement it
        neighbors = [neighbor for neighbor in neighbors if self.occupancy_map[neighbor[0], neighbor[1]] < OCCUPIED_THRESHOLD]

        return neighbors

    def reconstruct_path(self, came_from, current, start):
        """
        Reconstruct the path from generated by A*
        Parameters
        ----------
        came_from : np.array keeping all the computed links between the nodes
        current : point of arrival of A*
        start : start point of the trajectory

        Returns
        -------
        path : np.array in world coordinates
        """
        path = np.zeros((1, 2), dtype=int)
        path[0] = np.array(current)
        while not np.array_equal(current, start):
            current = np.array(came_from[current[0], current[1]])
            path = np.append(path, [current], axis=0)
        # Flip path as append is used instead of prepend
        # path = np.flip(path, axis=0)
        # Convert path to world coordinates
        x_path, y_path = self._conv_map_to_world(path[:, 0], path[:, 1])
        # Construct path as list of [x, y] world coordinates
        path = [[x, y] for x, y in zip(x_path, y_path)]
        return path

    def plan(self, start, goal):
        """
        Compute a path using A*, recompute plan if start or goal change
        start : [x, y, theta] nparray, start pose in world coordinates
        goal : [x, y, theta] nparray, goal pose in world coordinates
        """
        # TODO for TP5

        # Convert start and goal to map coordinates
        start_a = np.array(self._conv_world_to_map(start[0], start[1]))
        goal_a = np.array(self._conv_world_to_map(goal[0], goal[1]))
        # print(f"(map) start_a : {start_a}  goal_a : {goal_a}")

        # Initialization
        open_set = [[heuristic(start_a, goal_a),
                    start_a]]  # must be a list (no nd-array) to work with heapq, f score is kept in the set to sort it according to that value
        heapq.heapify(open_set)
        came_from = np.zeros(shape=self.occupancy_map.shape + (2,), dtype=int)

        g_score = np.ones_like(self.occupancy_map) * np.inf
        g_score[start_a[0], start_a[1]] = 0
        # f = g + h, f stocké dans open_set

        while open_set:
            [_, current] = heapq.heappop(open_set)

            if np.array_equal(current, goal_a): # Si on est arrivé
                path = self.reconstruct_path(came_from, current, start_a)
                return path

            for neighbor in self.get_neighbors(current):
                trial_g_score = g_score[current[0], current[1]] + np.sqrt(
                    (current[0] - neighbor[0]) ** 2 + (current[1] - neighbor[1]) ** 2)
                if trial_g_score < g_score[neighbor[0], neighbor[1]]:
                    came_from[neighbor[0], neighbor[1]] = current
                    g_score[neighbor[0], neighbor[1]] = trial_g_score
                    if neighbor not in [item[1] for item in open_set]:
                        heapq.heappush(open_set, [trial_g_score + heuristic(neighbor, goal_a), neighbor])

        print("Error : A* could not find path")
        return None

    def display(self, robot_pose):
        """
        Screen display of map and robot pose, using matplotlib
        robot_pose : [x, y, theta] nparray, corrected robot pose
        """

        plt.cla()
        plt.imshow(self.occupancy_map.T, origin='lower',
                   extent=[self.x_min_world, self.x_max_world, self.y_min_world, self.y_max_world])
        plt.clim(-4, 4)
        plt.axis("equal")

        delta_x = np.cos(robot_pose[2]) * 10
        delta_y = np.sin(robot_pose[2]) * 10
        plt.arrow(robot_pose[0], robot_pose[1], delta_x, delta_y,
                  color='red', head_width=5, head_length=10, )

        # plt.show()
        plt.pause(0.001)

    def display2(self, robot_pose):
        """
        Screen display of map and robot pose,
        using opencv (faster than the matplotlib version)
        robot_pose : [x, y, theta] nparray, corrected robot pose
        """

        img = cv2.flip(self.occupancy_map.T, 0)
        img = img - img.min()
        img = img / img.max() * 255
        img = np.uint8(img)
        img2 = cv2.applyColorMap(src=img, colormap=cv2.COLORMAP_JET)

        pt2_x = robot_pose[0] + np.cos(robot_pose[2]) * 20
        pt2_y = robot_pose[1] + np.sin(robot_pose[2]) * 20
        pt2_x, pt2_y = self._conv_world_to_map(pt2_x, -pt2_y)

        pt1_x, pt1_y = self._conv_world_to_map(robot_pose[0], -robot_pose[1])

        # print("robot_pose", robot_pose)
        pt1 = (int(pt1_x), int(pt1_y))
        pt2 = (int(pt2_x), int(pt2_y))
        cv2.arrowedLine(img=img2, pt1=pt1, pt2=pt2,
                        color=(0, 0, 255), thickness=2)
        cv2.imshow("map slam", img2)
        cv2.waitKey(1)

    def display_path(self, robot_pose, path):
        """
        Screen display of map and robot pose and path,
        using opencv (faster than the matplotlib version)
        robot_pose : [x, y, theta] nparray, corrected robot pose
        path : [[x1, y1], [x2, y2], ...] list of points
        """
        img = cv2.flip(self.occupancy_map.T, 0)
        img = img - img.min()
        img = img / img.max() * 255
        img = np.uint8(img)
        img2 = cv2.applyColorMap(src=img, colormap=cv2.COLORMAP_JET)

        COLOR = (0, 255, 0) # Path color
        RADIUS = 2

        # Draw the points on the image
        for point in path:
            cv2.circle(img2, (self._conv_world_to_map(point[0], -point[1])), RADIUS, COLOR, -1)  # -1 for filled circle

        pt2_x = robot_pose[0] + np.cos(robot_pose[2]) * 20
        pt2_y = robot_pose[1] + np.sin(robot_pose[2]) * 20
        pt2_x, pt2_y = self._conv_world_to_map(pt2_x, -pt2_y)

        pt1_x, pt1_y = self._conv_world_to_map(robot_pose[0], -robot_pose[1])

        # print("robot_pose", robot_pose)
        pt1 = (int(pt1_x), int(pt1_y))
        pt2 = (int(pt2_x), int(pt2_y))
        cv2.arrowedLine(img=img2, pt1=pt1, pt2=pt2,
                        color=(0, 0, 255), thickness=2)
        cv2.imshow("map slam", img2)
        cv2.waitKey(1)

    def save(self, filename):
        """
        Save map as image and pickle object
        filename : base name (without extension) of file on disk
        """

        plt.imshow(self.occupancy_map.T, origin='lower',
                   extent=[self.x_min_world, self.x_max_world,
                           self.y_min_world, self.y_max_world])
        plt.clim(-4, 4)
        plt.axis("equal")
        plt.savefig(filename + '.png')

        with open(filename + ".p", "wb") as fid:
            pickle.dump({'occupancy_map': self.occupancy_map,
                         'resolution': self.resolution,
                         'x_min_world': self.x_min_world,
                         'x_max_world': self.x_max_world,
                         'y_min_world': self.y_min_world,
                         'y_max_world': self.y_max_world}, fid)

    def load(self, filename):
        """
        Load map from pickle object
        filename : base name (without extension) of file on disk
        """
        # TODO
