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

from slam_bot_nano.sensors.stereo_camera import *
from slam_bot_nano.slam.camera_frame import CameraFrame, ORBFeatureExtractor
from slam_bot_nano.slam.math_utils import *
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

        self.map_fig = plt.figure()
        self.map_fig.canvas.set_window_title('2D Map')
        self.map_sublot = plt.subplot()
        self.map_sublot.autoscale_view(True,True,True)
        self.map_sublot.grid(True)

        self.t0_est = None
        self.traj3d_est = []
        self.poses = []
        self.cur_R = np.eye(3,3) # current rotation
        self.cur_t = np.zeros((3,1)) # current translation
        self.cum_R_inv = np.eye(3,3)
        self.cum_t_inv = np.zeros((3,1))
        self.prev_pos = np.zeros((3,1))
        self.prev_frame = None
        self.cur_frame = None
        self.prev_lidar_frame = None
        self.cur_lidar_frame = None
        self.matched_image = None
        self.trajectory_image = None

        self.frame_shape = (360, 640, 3)

        self.feature_extractor = ORBFeatureExtractor()

        self.map = []

    def start(self):
        for (left_frame, right_frame), (angle, ran, intensity) in self.replay:
            self.prev_frame = self.cur_frame
            self.prev_lidar_frame = self.cur_lidar_frame
            self.left_frame = left_frame
            self.right_frame = right_frame

            frame = CameraFrame(self.left_frame, self.feature_extractor)
            self.cur_frame = frame
            self.cur_lidar_frame = (ran, angle)

            self.fig.canvas.draw()
            self.lidar_polar.clear()
            self.lidar_polar.scatter(angle, ran, c=intensity, cmap='hsv', alpha=0.95)
            if self.prev_lidar_frame is not None:
                self.lidar_polar.scatter(self.prev_lidar_frame[1], self.prev_lidar_frame[0], cmap='Greens', alpha=0.95)
            lidar_frame = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype='uint8')
            self.lidar_frame = lidar_frame.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            self.lidar_frame = cv2.resize(self.lidar_frame, (self.frame_shape[1], self.frame_shape[0]))

            self.process_data()
            self.display_info()

        self.plt3d.quit()

    def process_data(self):
        prev_frame, cur_frame = self.prev_frame, self.cur_frame
        shape = self.frame_shape

        self.depth_img = get_depth_est(self.left_frame, self.right_frame)
        self.depth_img = cv2.resize(self.depth_img, (shape[1], shape[0]))

        if prev_frame is None:
            self.t0_est = np.array([self.cur_t[0], self.cur_t[1]])  # starting translation
            self.prev_pos = np.array([self.cur_t[0], self.cur_t[1]])
        else:
            idxs_ref, idxs_cur, matches = self.feature_extractor.match(prev_frame, cur_frame)

            R, t, self.matched_image, self.quiver_image = estimatePose(prev_frame,
                cur_frame,
                idxs_ref,
                idxs_cur,
                self.left_camera,
                matches=matches,
                R_i=self.cur_R.copy(),
                t_i=self.cur_t.copy())

            R, t = lidar_localization(
                self.prev_lidar_frame,
                self.cur_lidar_frame,
                self.map,
                cum_R=self.cur_R,
                cum_t=self.cur_t
            )

            if self.matched_image is not None and self.matched_image.size > 0:
                self.matched_image = cv2.resize(self.matched_image, (shape[1], shape[0]))
                self.quiver_image = cv2.resize(self.quiver_image, (shape[1], shape[0]))

            if R is not None and t is not None:
                self.cur_t = self.cur_t + self.cur_R.dot(t)
                self.cur_R = self.cur_R.dot(R)

        if (self.t0_est is not None):
            p = [self.cur_t[0]-self.t0_est[0], self.cur_t[1]-self.t0_est[1], 0]   # the estimated traj starts at 0

            self.traj3d_est.append(self.cur_t)
            self.plt3d.drawTraj(self.traj3d_est,'estimated',color='g',marker='.')
            self.plt3d.refresh()

            self.trajectory_fig.canvas.draw()
            self.trajectory_sublot.clear()
            self.trajectory_sublot.scatter(np.array(self.traj3d_est)[:, 0], np.array(self.traj3d_est)[:, 1], c=[idx/len(self.traj3d_est) for idx in range(len(self.traj3d_est))],cmap='Blues', alpha=0.95)
            trajectory_frame = np.frombuffer(self.trajectory_fig.canvas.tostring_rgb(), dtype='uint8')
            self.trajectory_frame = trajectory_frame.reshape(self.trajectory_fig.canvas.get_width_height()[::-1] + (3,))
            self.trajectory_frame = cv2.resize(self.trajectory_frame, (self.frame_shape[1], self.frame_shape[0]))

        if len(self.map) > 0:
            print(np.array(self.map).shape)

            self.map_fig.canvas.draw()
            self.map_sublot.clear()
            self.map_sublot.scatter(np.array(self.map)[:, 0], np.array(self.map)[:, 1], cmap='Blues', alpha=0.95)
            map_frame = np.frombuffer(self.map_fig.canvas.tostring_rgb(), dtype='uint8')
            self.map_frame = map_frame.reshape(self.map_fig.canvas.get_width_height()[::-1] + (3,))
            self.map_frame = cv2.resize(self.map_frame, (self.frame_shape[1], self.frame_shape[0]))

            cv2.imshow("ASDA", self.map_frame)
            cv2.waitKey(1)

    def display_info(self):

        if self.matched_image is None:
            self.matched_image = np.zeros(self.frame_shape, dtype=np.uint8)

        if self.depth_img is None:
            self.depth_img = np.zeros(self.frame_shape, dtype=np.uint8)

        if self.trajectory_frame is None:
            self.trajectory_frame = np.zeros(self.frame_shape, dtype=np.uint8)

        images = np.hstack((self.left_frame, self.right_frame, self.lidar_frame))
        tmp = np.hstack((self.matched_image, self.trajectory_frame, self.depth_img))
        images = np.vstack((images, tmp))

        cv2.imshow("Camera Images", images)
        cv2.waitKey(1)

if __name__ == "__main__":
    cl = ControlLoop("slam_bot_nano/data/runs/home_session")
    cl.start()
