"""
Robot controller definition
Complete controller including SLAM, planning, path following
"""
import numpy as np

from place_bot.entities.robot_abstract import RobotAbstract
from place_bot.entities.odometer import OdometerParams
from place_bot.entities.lidar import LidarParams

from tiny_slam import TinySlam

from control import reactive_obst_avoid


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

    def control(self):
        """
        Main control function executed at each time step
        """
        self.counter += 1

        # Compute new command speed to perform obstacle avoidance
        command = reactive_obst_avoid(self.lidar())

        # Update occupancy grid map
        self.tiny_slam.update_map(self.lidar(), self.odometer_values())

        return command
