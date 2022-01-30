import os
import curses
import sys
import threading
import time

from adafruit_servokit import ServoKit
import termios, fcntl, sys, os
from jetracer.nvidia_racecar import NvidiaRacecar
import numpy as np
import cv2
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from slam_bot_nano.vehicle.utils import init_bot
from slam_bot_nano.controls.keyboard_input import KeyboardInput
from slam_bot_nano.sensors.stereo_camera import StereoCamera
from slam_bot_nano.sensors.lidar import Lidar2D
from slam_bot_nano.client_server_com.image_transfer_service import ImageTransferClient
from slam_bot_nano.slam.camera_frame import CameraFrame, ORBFeatureExtractor
from slam_bot_nano.slam.math_utils import *
from slam_bot_nano.slam.visual_odometry import VisualOdometry
from slam_bot_nano.slam.lidar_odometry import LidarOdometry
from slam_bot_nano.vehicle.vehicle_kinematic_model import VehicleKinematicModel


RMAX = 32.0

class ControlLoop:

    def __init__(self, send_over_udp=False):
        self.car = init_bot()
        self.kb = KeyboardInput()
        self.left_camera = StereoCamera(0).start()
        self.right_camera = StereoCamera(1).start()
        self.lidar = Lidar2D()
        self.it = ImageTransferClient("192.168.2.212", 5000, "192.168.2.205")
        self.frames_without_controls = 0
        self.send_over_udp = send_over_udp

        self.camera_lock = threading.Lock()
        self.lidar_lock = threading.Lock()
        self.controls_lock = threading.Lock()

        self.left_grabbed, self.right_grabbed, self.lidar_grabbed = False, False, False
        self.left_frame, self.right_frame = None, None

        self.lidar_frame = None

        self.running = False

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
        self.cur_lidar_frame = None

        self.vo = VisualOdometry(self.left_camera)
        self.lo = LidarOdometry()
        self.vkm = VehicleKinematicModel()
        self.recent_lat_control = 0
        self.recent_long_control = 0
        self.recent_t = None

        self.frame_shape = (360, 640, 3)

        print("Init finished!")

    def start(self):
        if self.running:
            print('Control loop is already running')
            return

        self.running = True

        self.image_sensor_thread = threading.Thread(target=self.get_image_sensor_data, daemon=True)
        self.image_sensor_thread.start()

        self.lidar_sensor_thread = threading.Thread(target=self.get_lidar_data, daemon=True)
        self.lidar_sensor_thread.start()

        self.controls_thread = threading.Thread(target=self.get_controls, daemon=True)
        self.controls_thread.start()

        if self.send_over_udp:
            self.display_thread = threading.Thread(target=self.send_info, daemon=True)
        else:
            self.display_thread = threading.Thread(target=self.display_info, daemon=True)
            
        self.display_thread.start()

        self.processing_thread = threading.Thread(target=self.process_data, daemon=True)
        self.processing_thread.start()

        self.image_sensor_thread.join()
        self.lidar_sensor_thread.join()
        self.controls_thread.join()
        self.display_thread.join()
        self.processing_thread.join()

    def stop(self):
        self.running = False
        self.image_sensor_thread.join()
        self.lidar_sensor_thread.join()
        self.controls_thread.join()
        self.display_thread.join()

    def process_data(self):
        while self.running:   
            shape = self.frame_shape
            with self.camera_lock:
                if self.left_frame is not None:
                    self.vo.update(self.left_frame, 0)
                    self.depth_img = get_depth_est(self.left_frame, self.right_frame)
                    self.depth_img = cv2.resize(self.depth_img, (shape[1], shape[0]))
                
            with self.lidar_lock:
                if self.cur_lidar_frame is not None:
                    self.lo.update(self.cur_lidar_frame)
    
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

    def get_image_sensor_data(self):
        while self.running:   
            with self.camera_lock:
                self.left_grabbed, self.left_frame = self.left_camera.read()
                self.right_grabbed, self.right_frame = self.right_camera.read()

                self.left_frame = cv2.flip(self.left_frame, -1)
                self.right_frame = cv2.flip(self.right_frame, -1)

    def get_lidar_data(self):
        while self.running:   
            with self.lidar_lock:
                self.lidar_grabbed, angle, ran, intensity = self.lidar.get_points()
                self.lidar_frame = None
                self.cur_lidar_frame = (ran, angle)

                if self.lidar_grabbed:
                    self.fig.canvas.draw()
                    self.lidar_polar.clear()
                    self.lidar_polar.scatter(angle, ran, c=intensity, cmap='hsv', alpha=0.95)
                    lidar_frame = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype='uint8')
                    self.lidar_frame = lidar_frame.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
                    self.lidar_frame = cv2.resize(self.lidar_frame, (self.frame_shape[1], self.frame_shape[0])) 

    def get_controls(self):
        c = 'r'
        while self.running and c != 'q':
            cur_t = time.time()
            if self.recent_t is not None:
                dt = cur_t - self.recent_t
            else:
                dt = None
            self.recent_t = cur_t

            try:            
                c = self.kb.getch()
                if c:
                    self.frames_without_controls = 0

                    if c == 'd':
                        self.car.right()
                        self.recent_lat_control = 1
                    elif c == 'a':
                        self.car.left()
                        self.recent_lat_control = -1
                    elif c == 'w':
                        self.car.forward()
                        self.car.straight()
                        self.recent_lat_control = 0
                        self.recent_long_control = 1
                    elif c == 's':
                        self.car.backward()
                        self.car.straight()
                        self.recent_long_control = -1
                    elif c == 'b':
                        self.car.brake()
                else:
                    self.frames_without_controls += 1

                    if self.frames_without_controls > 500000:
                        self.car.straight()
                        self.car.brake()
            except IOError:
                self.car.straight()
                self.car.brake()

            if dt is not None:
                self.vkm.update(dt, self.recent_lat_control, self.recent_long_control)

        self.running = False

    def send_info(self):
        while self.running:
            with self.camera_lock:
                with self.lidar_lock:
                    images = np.hstack((self.left_frame, self.right_frame, self.lidar_frame))
                    self.it.sendall(images)

    def display_info(self):
        while self.running:
            with self.camera_lock:
                with self.lidar_lock:
                    if self.left_grabbed and self.right_grabbed and self.lidar_grabbed:
                        if self.matched_image is None or self.trajectory_image is None:
                            images = np.hstack((self.left_frame, self.right_frame, self.lidar_frame))
                        else:
                            images = np.hstack((self.left_frame, self.right_frame, self.lidar_frame))
                            tmp = np.hstack((self.matched_image, self.trajectory_image))
                            images = np.vstack((images, tmp))
                        
                        cv2.imshow("Camera Images", images)
                        cv2.waitKey(1)

if __name__ == "__main__":
    cl = ControlLoop()
    cl.start()
