import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


class ControlsOdometry:

    def __init__(self):
        inches_to_m = 0.0254

        self.wheel_circumference = 2.8 * np.pi * inches_to_m
        self.rps = 0.7958

        points_l = np.array([[-0.6, 1.65], [-1.4, 3.8], [-3, 6.5], [-5.2, 9.2], [-7.55, 11.15]]) * inches_to_m
        points_r = np.array([[1, 3.1], [2, 5.25], [3, 6.9], [4, 8.15], [5, 9.25], [6, 10.1]]) * inches_to_m

        xl, yl = points_l[:, 0].flatten(), points_l[:, 1].flatten()
        xr, yr = points_r[:, 0].flatten(), points_r[:, 1].flatten()

        self.fl = interp1d(xl, yl, kind='linear', fill_value="extrapolate")
        self.fr = interp1d(xr, yr, kind='linear', fill_value="extrapolate")

        self.cur_R = np.eye(3,3) # current rotation
        self.cur_t = np.zeros((3,1)) # current translation

        self.last_control = None
        self.last_timestamp = None

    def update(self, timestamp, new_control=None):
        if new_control is not None:
            pass
        else:
            pass
