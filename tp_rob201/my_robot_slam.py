"""
Robot controller definition
Complete controller including SLAM, planning, path following
"""
import numpy as np
from place_bot.entities.lidar import LidarParams
from place_bot.entities.odometer import OdometerParams
from place_bot.entities.robot_abstract import RobotAbstract

from control import reactive_obst_avoid, potential_field_control
from tiny_slam import TinySlam


# Definition of our robot controller
class MyRobotSlam(RobotAbstract):
    """A robot controller including SLAM, path planning and path following"""

    def __init__(self,
                 lidar_params: LidarParams = LidarParams(),
                 odometer_params: OdometerParams = OdometerParams()):
        # Passing parameter to parent class
        super().__init__(should_display_lidar=False,
                         lidar_params=lidar_params,
                         odometer_params=odometer_params)

        # step counter to deal with init and display
        self.counter = 0

        # Init SLAM object
        self._size_area = (800, 800)
        self.tiny_slam = TinySlam(x_min=- self._size_area[0],
                                  x_max=self._size_area[0],
                                  y_min=- self._size_area[1],
                                  y_max=self._size_area[1],
                                  resolution=2)

        # storage for pose after localization
        self.corrected_pose = np.array([0, 0, 0])

        # Added for conveniance
        self.path = None
        self.goal = np.array([0, 0, 0])


    def control(self):
        """
        Main control function executed at each time step
        """
        # Params
        SCORE_TO_UPDATE_MAP = 10  # 20 with simplest lidar model and 600 for intermediate lidar model
        NB_ITER_TO_BACK = 350  # Nombre d'itérations avant retour au bercail
        THRESHOLD_DIST = 20  # Distance à laquelle on considère que le robot est arrivé à destination (peut être temporaire)

        best_score = self.tiny_slam.localise(self.lidar(), self.odometer_values())
        # print(f"best_score : {best_score:.2f}")
        self.corrected_pose = self.tiny_slam.get_corrected_pose(self.odometer_values())
        command = {"forward": 0, "rotation": 0}

        if best_score >= SCORE_TO_UPDATE_MAP or self.counter == 0:
            # Update occupancy grid map
            self.tiny_slam.update_map(self.lidar(), self.odometer_values())

        # Au bout de NB_ITER_TO_BACK itérations, on lance le path planning
        if self.counter > NB_ITER_TO_BACK:
            if self.path is None:  # Si pas de path déjà calculé
                self.path = self.tiny_slam.plan(self.corrected_pose, self.goal)

            if self.path: # Si path non vide, on continue à suivre le chemin
                if np.linalg.norm(self.corrected_pose[:2] - self.path[-1]) < THRESHOLD_DIST:  # Si assez proche du point à atteindre, on passe au suivant
                    self.path.pop()

                command = potential_field_control(self.lidar(), self.corrected_pose, np.append(self.path[-1], 0)) if self.path else command
                self.tiny_slam.display_path(self.odometer_values(), self.path)

            # Si path vide, on est arrivé à destination

        else:  # Partie d'exploration
            command = reactive_obst_avoid(self.lidar())
            self.tiny_slam.display2(self.odometer_values())

        self.counter += 1
        return command
