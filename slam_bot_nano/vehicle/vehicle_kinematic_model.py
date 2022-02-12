from math import atan2, sin, cos

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

normalise_angle = lambda angle : atan2(sin(angle), cos(angle))

class VehicleKinematicModel:

    def __init__(self):
        self.moving_velocity = 1 # m/s
        # self.l_steer_angle = 0.131019 # radians (75.06835736 degrees)
        # self.r_steer_angle = -0.1289959 # radians (73.909206445 degrees)
        self.l_steer_angle = 0.08 # radians (75.06835736 degrees)
        self.r_steer_angle = -0.11 # radians (73.909206445 degrees)
        self.wheel_base_length = 0.15 # m
        # self.wheel_base_length = 0.25 # m

        self.yaw = np.pi/2.0
        self.pos = np.zeros((2, 1))

    def update(self, dt, lat_control=0, long_control=0):
        if long_control == 1:
            velocity = self.moving_velocity
        elif long_control == -1:
            velocity = -1 * self.moving_velocity
        elif long_control == 0:
            velocity = 0

        if lat_control == 1:
            steering_angle = self.r_steer_angle
        elif lat_control == -1:
            steering_angle = self.l_steer_angle
        elif lat_control == 0:
            steering_angle = 0

        angular_velocity = velocity * np.tan(steering_angle) / self.wheel_base_length
        new_x = self.pos[0] + velocity * cos(self.yaw) * dt
        new_y = self.pos[1] + velocity * sin(self.yaw) * dt
        new_yaw = normalise_angle(self.yaw + angular_velocity * dt)

        self.yaw = new_yaw
        self.pos = np.array([new_x, new_y])
