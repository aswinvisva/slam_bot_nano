import os
import curses
import sys
import threading

import termios, fcntl, sys, os
import numpy as np
import cv2
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
plt.switch_backend('TkAgg')
from scipy.spatial.transform import Rotation as R

from slam_bot_nano.sensors.stereo_camera import *
from slam_bot_nano.slam.camera_frame import CameraFrame, ORBFeatureExtractor
from slam_bot_nano.slam.math_utils import *
from slam_bot_nano.slam.visual_odometry import VisualOdometry
from slam_bot_nano.slam.lidar_odometry import LidarOdometry
from slam_bot_nano.slam.map import PointMap
from slam_bot_nano.slam.depth_estimation import get_depth_est
from display import Mplot3d
from slam_bot_nano.vehicle.vehicle_kinematic_model import VehicleKinematicModel

RMAX = 32.0


class Replay:
    def __init__(self, data_path, skip_every=1):
        self.subfolders = sorted([f.path for f in os.scandir(data_path) if f.is_dir()], key=lambda k: float(os.path.basename(k)))
        self.idx = 0
        self.skip_every=skip_every
        self.prev_ts = None

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= len(self.subfolders):
            raise StopIteration

        data_dir = self.subfolders[self.idx]
        ts = float(os.path.basename(data_dir))
        dt = None

        if self.prev_ts is not None:
            dt = ts - self.prev_ts

        self.prev_ts = ts

        with np.load(os.path.join(data_dir, 'lidar.npz')) as data:
            angle = data['angle']
            ran = data['ran']
            intensity = data['intensity']

        with np.load(os.path.join(data_dir, 'controls.npz')) as data:
            long_control = data['long']
            lat_control = data['lat']

        self.idx += self.skip_every

        return (long_control, lat_control, dt), (angle, ran, intensity)

class ControlLoop:
    def __init__(self, path):
        self.replay = Replay(path)

        self.lidar_grabbed = False

        self.plt3d = Mplot3d(title='3D trajectory')

        self.lidar_frame = None

        self.fig = plt.figure()
        self.fig.canvas.set_window_title('YDLidar LIDAR Monitor')
        self.lidar_polar = plt.subplot(polar=True)
        self.lidar_polar.autoscale_view(True,True,True)
        self.lidar_polar.set_rmax(RMAX)
        self.lidar_polar.grid(True)

        self.trajectory_fig = plt.figure()
        self.trajectory_fig.canvas.set_window_title('Trajectory')
        self.trajectory_sublot = plt.subplot()
        self.trajectory_sublot.autoscale_view(True,True,True)
        self.trajectory_sublot.grid(True)

        self.t0_est = None
        self.traj3d_est = []
        self.traj3d_gt = []
        self.cur_t_gt = None

        self.cur_frame = None
        self.cur_lidar_frame = None
        self.trajectory_frame = None
        self.map_frame = None

        self.lo = LidarOdometry()
        self.vkm = VehicleKinematicModel()

        self.frame_shape = (360, 640, 3)

        self.map = PointMap()

        self.r270_matrix = R.from_euler('xyz', [0, 0, 90], degrees=True).as_matrix()

        self.idx = 0

    def start(self):
        for (long_control, lat_control, dt), (angle, ran, intensity) in self.replay:
            self.recent_long_control = long_control
            self.recent_lat_control = lat_control
            self.dt = dt
            self.cur_lidar_frame = (ran, angle)
            self.fig.canvas.draw()
            self.lidar_polar.clear()
            self.lidar_polar.scatter(angle, ran, c=intensity, cmap='hsv', alpha=0.95)
            lidar_frame = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype='uint8')
            self.lidar_frame = lidar_frame.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            self.lidar_frame = cv2.resize(self.lidar_frame, (self.frame_shape[1], self.frame_shape[0]))

            self.process_data()
            self.display_info()

            if self.idx % 50 == 0:
                self.map.save("slam_bot_nano/data", "temp_map")

            self.idx += 1

        self.plt3d.quit()
        self.map.save("slam_bot_nano/data", "final_map")

    def process_data(self):
        shape = self.frame_shape

        self.lo.update(self.cur_lidar_frame)
        self.cur_R, self.cur_t = self.lo.cur_R.get(), self.lo.cur_t.get()

        if self.dt is not None:
            self.vkm.update(self.dt, self.recent_lat_control, self.recent_long_control)
            pos = self.vkm.pos
            self.cur_t_gt = np.zeros((3, 1))
            self.cur_t_gt[0] = pos[0]
            self.cur_t_gt[1] = pos[1]

        # self.cur_R, self.cur_t = self.lo.cur_R, self.lo.cur_t
        self.map.update(self.cur_lidar_frame, self.idx, self.cur_R, self.cur_t)
        self.map_frame = self.map.plot()

        if self.cur_t is not None and self.cur_t_gt is not None:
            self.traj3d_est.append(self.cur_t)
            self.traj3d_gt.append(self.cur_t_gt)

            self.plt3d.drawTraj(self.traj3d_est,'estimated',color='g',marker='.')
            self.plt3d.drawTraj(self.traj3d_gt,'gt',color='r',marker='.')

            self.plt3d.refresh()

            self.trajectory_fig.canvas.draw()
            self.trajectory_sublot.clear()
            self.trajectory_sublot.scatter(np.array(self.traj3d_est)[:, 0], np.array(self.traj3d_est)[:, 1], c=[idx/len(self.traj3d_est) for idx in range(len(self.traj3d_est))],cmap='Reds', alpha=0.95)
            trajectory_frame = np.frombuffer(self.trajectory_fig.canvas.tostring_rgb(), dtype='uint8')
            self.trajectory_frame = trajectory_frame.reshape(self.trajectory_fig.canvas.get_width_height()[::-1] + (3,))
            self.trajectory_frame = cv2.resize(self.trajectory_frame, (self.frame_shape[1], self.frame_shape[0]))

    def display_info(self):
        if self.trajectory_frame is None:
            self.trajectory_frame = np.zeros(self.frame_shape, dtype=np.uint8)

        if self.map_frame is None:
            self.map_frame = np.zeros(self.frame_shape, dtype=np.uint8)

        images = np.hstack((self.trajectory_frame, self.lidar_frame, self.map_frame))

        cv2.imshow("Data", images)
        cv2.waitKey(1)

if __name__ == "__main__":
    cl = ControlLoop("slam_bot_nano/data/runs/home_session_lidar_only")
    cl.start()
