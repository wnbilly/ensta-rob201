import numpy as np

from place_bot.entities.robot_abstract import RobotAbstract
from place_bot.entities.odometer import OdometerParams
from place_bot.entities.lidar import LidarParams

from control import reactive_obst_avoid
from control import wall_follow
from control import potential_field_control

from tiny_slam import TinySlam

class MyRobot(RobotAbstract):
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

        # storage for pose after localization
        self.corrected_pose = np.array([0, 0, 0])

    def control(self):
        """
        Main control function executed at each time step
        """

        # Compute new command speed to perform obstacle avoidance
        # command = reactive_obst_avoid(self.lidar())
        # command = wall_follow((self.lidar()))

        # goal = np.array([240, 20, 0]) # Unreachable with potential
        goal = np.array([370, -260, 0]) # Simple goal w/out walls to avoid
        # goal = np.array([240, 220, 0])  # Reachable via the closest door to the spawn point
        pose = np.array([self.position[0], self.position[1], self.angle])
        command = potential_field_control(self.lidar(), pose, goal)

        return command
