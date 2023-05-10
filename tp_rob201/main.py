""" A simple SLAM demonstration using the "placebot" robot simulator """

import sys

from place_bot.entities.lidar import LidarParams
from place_bot.entities.odometer import OdometerParams
from place_bot.simu_world.simulator import Simulator

from my_robot_slam import MyRobotSlam
from robot import MyRobot
from worlds.my_world import MyWorld

if __name__ == '__main__':
    lidar_params = LidarParams()
    lidar_params.noise_enable = True
    # lidar_params.fov = 360
    # lidar_params.resolution = 361
    # lidar_params.max_range = 600
    # lidar_params.std_dev_noise = 2.5

    odometer_params = OdometerParams()
    # odometer_params.param1 = 0.3  # 0.3  # meter/meter, influence of translation to translation
    # odometer_params.param2 = 0.1  # 0.1  # meter/degree, influence of rotation to translation
    # odometer_params.param3 = 0.04  # 0.04 # degree/meter, influence of translation to rotation
    # odometer_params.param4 = 0.01  # 0.01 # degree/degree, influence of rotation to rotation

    USE_SLAM = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    print(f"To use SLAM, add 1 as argument, 0 otherwise. Using SLAM: {USE_SLAM}")

    my_robot = MyRobotSlam(lidar_params=lidar_params, odometer_params=odometer_params) if USE_SLAM else MyRobot(
        lidar_params=lidar_params, odometer_params=odometer_params)
    my_world = MyWorld(robot=my_robot)
    simulator = Simulator(the_world=my_world,
                          use_keyboard=False,
                          draw_lidar=True,
                          use_mouse_measure=True)

    simulator.run()

    if USE_SLAM:
        my_robot.tiny_slam.display(my_robot.corrected_pose)
