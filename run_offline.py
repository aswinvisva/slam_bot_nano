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
from slam_bot_nano.slam.depth_estimation import get_depth_est
from display import Mplot3d

RMAX = 32.0


class Replay:
    def __init__(self, data_path, skip_every=1):
        self.subfolders = sorted([f.path for f in os.scandir(data_path) if f.is_dir()], key=lambda k: float(os.path.basename(k)))
        self.idx = 0
        self.skip_every=skip_every

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= len(self.subfolders):
            raise StopIteration

        data_dir = self.subfolders[self.idx]
        left_frame = cv2.imread(os.path.join(data_dir, 'left_frame.jpg'))
        right_frame = cv2.imread(os.path.join(data_dir, 'right_frame.jpg'))
        with np.load(os.path.join(data_dir, 'lidar.npz')) as data:
            angle = data['angle']
            ran = data['ran']
            intensity = data['intensity']

        self.idx += self.skip_every

        return (left_frame, right_frame), (angle, ran, intensity)

class ControlLoop:
    def __init__(self, path):
        self.replay = Replay(path)

        self.left_grabbed, self.right_grabbed, self.lidar_grabbed = False, False, False
        self.left_frame, self.right_frame = None, None

        self.left_camera = StereoCamera(0)
        self.right_camera = StereoCamera(1)

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

        self.trajectory_lidar_fig = plt.figure()
        self.trajectory_lidar_fig.canvas.set_window_title('Trajectory Lidar')
        self.trajectory_lidar_subplot = plt.subplot()
        self.trajectory_lidar_subplot.autoscale_view(True,True,True)
        self.trajectory_lidar_subplot.grid(True)

        self.t0_est = None
        self.traj3d_est = []
        self.traj3d_est_lidar = []

        self.cur_frame = None
        self.cur_lidar_frame = None
        self.matched_image = None
        self.trajectory_frame = None
        self.trajectory_frame_lidar = None

        self.vo = VisualOdometry(self.left_camera)
        self.lo = LidarOdometry()

        self.frame_shape = (360, 640, 3)

        self.map = []

        self.R_matrix = R.from_euler('z', 270, degrees=True).as_matrix()

    def start(self):
        for (left_frame, right_frame), (angle, ran, intensity) in self.replay:
            self.left_frame = left_frame
            self.right_frame = right_frame
            self.cur_lidar_frame = (ran, angle)
            self.fig.canvas.draw()
            self.lidar_polar.clear()
            self.lidar_polar.scatter(angle, ran, c=intensity, cmap='hsv', alpha=0.95)
            lidar_frame = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype='uint8')
            self.lidar_frame = lidar_frame.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            self.lidar_frame = cv2.resize(self.lidar_frame, (self.frame_shape[1], self.frame_shape[0]))

            self.process_data()
            self.display_info()

        self.plt3d.quit()

    def process_data(self):
        shape = self.frame_shape

        self.vo.update(self.left_frame, 0)
        self.lo.update(self.cur_lidar_frame)
        self.depth_img = get_depth_est(self.left_frame, self.right_frame)
        self.depth_img = cv2.resize(self.depth_img, (shape[1], shape[0]))

        self.cur_R, self.cur_t, self.matched_image = self.vo.cur_R, self.vo.cur_t, self.vo.matched_image
        self.cur_R_lidar, self.cur_t_lidar = self.lo.cur_R, self.lo.cur_t

        if self.matched_image is not None and self.matched_image.size > 0:
            self.matched_image = cv2.resize(self.matched_image, (shape[1], shape[0]))

        if self.cur_R is not None and self.cur_t is not None:
            self.traj3d_est.append(self.cur_t)
            self.traj3d_est_lidar.append(self.cur_t_lidar)

            points = self.R_matrix @ np.array(self.traj3d_est)

            self.plt3d.drawTraj(self.traj3d_est,'estimated',color='g',marker='.')
            self.plt3d.refresh()

            self.trajectory_fig.canvas.draw()
            self.trajectory_sublot.clear()
            self.trajectory_sublot.scatter(points[:, 0], points[:, 1], c=[idx/len(self.traj3d_est) for idx in range(len(self.traj3d_est))],cmap='Blues', alpha=0.95)
            trajectory_frame = np.frombuffer(self.trajectory_fig.canvas.tostring_rgb(), dtype='uint8')
            self.trajectory_frame = trajectory_frame.reshape(self.trajectory_fig.canvas.get_width_height()[::-1] + (3,))
            self.trajectory_frame = cv2.resize(self.trajectory_frame, (self.frame_shape[1], self.frame_shape[0]))

            self.trajectory_lidar_fig.canvas.draw()
            self.trajectory_lidar_subplot.clear()
            self.trajectory_lidar_subplot.scatter(np.array(self.traj3d_est_lidar)[:, 0], np.array(self.traj3d_est_lidar)[:, 1], c=[idx/len(self.traj3d_est_lidar) for idx in range(len(self.traj3d_est_lidar))],cmap='Reds', alpha=0.95)
            trajectory_frame_lidar = np.frombuffer(self.trajectory_lidar_fig.canvas.tostring_rgb(), dtype='uint8')
            self.trajectory_frame_lidar = trajectory_frame_lidar.reshape(self.trajectory_lidar_fig.canvas.get_width_height()[::-1] + (3,))
            self.trajectory_frame_lidar = cv2.resize(self.trajectory_frame_lidar, (self.frame_shape[1], self.frame_shape[0]))

    def display_info(self):

        if self.matched_image is None:
            self.matched_image = np.zeros(self.frame_shape, dtype=np.uint8)

        if self.depth_img is None:
            self.depth_img = np.zeros(self.frame_shape, dtype=np.uint8)

        if self.trajectory_frame is None:
            self.trajectory_frame = np.zeros(self.frame_shape, dtype=np.uint8)

        if self.trajectory_frame_lidar is None:
            self.trajectory_frame_lidar = np.zeros(self.frame_shape, dtype=np.uint8)

        images = np.hstack((self.left_frame, self.right_frame, self.lidar_frame))
        tmp = np.hstack((self.matched_image, self.trajectory_frame, self.depth_img))
        images = np.vstack((images, tmp))
        tmp = np.hstack((self.trajectory_frame_lidar, np.zeros((self.frame_shape[0], self.frame_shape[1]*2, self.frame_shape[2]), dtype=np.uint8)))
        images = np.vstack((images, tmp))

        cv2.imshow("Data", images)
        cv2.waitKey(1)

if __name__ == "__main__":
    cl = ControlLoop("slam_bot_nano/data/runs/home_session")
    cl.start()
