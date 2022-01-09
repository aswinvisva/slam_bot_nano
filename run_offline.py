import os
import curses
import sys
import threading

import termios, fcntl, sys, os
import numpy as np
import cv2
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from slam_bot_nano.sensors.stereo_camera import * 
from slam_bot_nano.slam.camera_frame import CameraFrame, ORBFeatureExtractor
from slam_bot_nano.slam.math_utils import *

RMAX = 32.0


class Replay:
    def __init__(self, data_path):
        self.subfolders = sorted([f.path for f in os.scandir(data_path) if f.is_dir()], key=lambda k: float(os.path.basename(k)))
        self.idx = 0

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

        self.idx += 1

        return (left_frame, right_frame), (angle, ran, intensity)

class ControlLoop:
    def __init__(self, path):
        self.replay = Replay(path)

        self.left_grabbed, self.right_grabbed, self.lidar_grabbed = False, False, False
        self.left_frame, self.right_frame = None, None

        self.left_camera = StereoCamera(0)
        self.right_camera = StereoCamera(1)

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
        self.poses = []
        self.cur_R = np.eye(3,3) # current rotation 
        self.cur_t = np.zeros((3,1)) # current translation 
        self.prev_frame = None
        self.cur_frame = None
        self.matched_image = None
        self.trajectory_image = None

        self.frame_shape = (360, 640, 3)

        self.feature_extractor = ORBFeatureExtractor()

    def start(self):
        for (left_frame, right_frame), (angle, ran, intensity) in self.replay:
            self.prev_frame = self.cur_frame
            self.left_frame = left_frame
            self.right_frame = right_frame

            frame = CameraFrame(self.left_frame, self.feature_extractor)
            self.cur_frame = frame

            self.fig.canvas.draw()
            self.lidar_polar.clear()
            self.lidar_polar.scatter(angle, ran, c=intensity, cmap='hsv', alpha=0.95)
            lidar_frame = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype='uint8')
            self.lidar_frame = lidar_frame.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            self.lidar_frame = cv2.resize(self.lidar_frame, (self.frame_shape[1], self.frame_shape[0])) 

            self.process_data()
            self.display_info()

    def process_data(self):
        prev_frame, cur_frame = self.prev_frame, self.cur_frame
        shape = self.frame_shape

        if prev_frame is None:
            self.t0_est = np.array([self.cur_t[0], self.cur_t[1], self.cur_t[2]])  # starting translation 
        else:
            idxs_ref, idxs_cur, matches = self.feature_extractor.match(prev_frame, cur_frame)

            R, t, self.matched_image = estimatePose(prev_frame, cur_frame, idxs_ref, idxs_cur, self.left_camera, matches=matches)

            print("New controls")

            print("="*10)
            print(R)
            print("="*10)
            print(t)
            print("="*10)
            
            if self.matched_image is not None and self.matched_image.size > 0:
                self.matched_image = cv2.resize(self.matched_image, (shape[1] * 2, shape[0])) 

            if R is not None and t is not None:
                self.cur_t = self.cur_t + self.cur_R.dot(t) 
                self.cur_R = self.cur_R.dot(R)

        if (self.t0_est is not None):             
            p = [self.cur_t[0]-self.t0_est[0], self.cur_t[1]-self.t0_est[1], self.cur_t[2]-self.t0_est[2]]   # the estimated traj starts at 0
            self.traj3d_est.append(self.cur_t)
            self.poses.append(poseRt(self.cur_R, p))   

            traj3d_est_np = np.array(self.traj3d_est)

            self.trajectory_fig.canvas.draw()
            self.trajectory_sublot.clear()
            self.trajectory_sublot.scatter(traj3d_est_np[:, 0], traj3d_est_np[:, 1])
            trajectory_image = np.frombuffer(self.trajectory_fig.canvas.tostring_rgb(), dtype='uint8')
            self.trajectory_image = trajectory_image.reshape(self.trajectory_fig.canvas.get_width_height()[::-1] + (3,))

            self.trajectory_image = cv2.resize(self.trajectory_image, (shape[1], shape[0])) 

    def display_info(self):
        if self.matched_image is None or self.trajectory_image is None:
            images = np.hstack((self.left_frame, self.right_frame, self.lidar_frame))
        else:
            images = np.hstack((self.left_frame, self.right_frame, self.lidar_frame))
            tmp = np.hstack((self.matched_image, self.trajectory_image))
            images = np.vstack((images, tmp))
        
        cv2.imshow("Camera Images", images)
        cv2.waitKey(100)

if __name__ == "__main__":
    cl = ControlLoop("slam_bot_nano/data/runs/home_session")
    cl.start()
