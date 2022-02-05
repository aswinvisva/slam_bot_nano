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
        self.lidar = Lidar2D()
        self.it = ImageTransferClient("192.168.2.212", 5000, "192.168.2.205")
        self.frames_without_controls = 0
        self.send_over_udp = send_over_udp

        self.lidar_lock = threading.Lock()
        self.controls_lock = threading.Lock()

        self.left_grabbed, self.right_grabbed, self.lidar_grabbed = False, False, False

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
        self.trajectory_image = None
        self.cur_lidar_frame = None

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
                
            with self.lidar_lock:
                if self.cur_lidar_frame is not None:
                    self.lo.update(self.cur_lidar_frame)
    
            self.cur_R, self.cur_t = self.lo.cur_R, self.lo.cur_t

            if self.cur_R is not None and self.cur_t is not None:
                self.traj3d_est.append(self.cur_t)

                self.trajectory_fig.canvas.draw()
                self.trajectory_sublot.clear()
                self.trajectory_sublot.scatter(np.array(self.traj3d_est)[:, 0], np.array(self.traj3d_est)[:, 1], c=[idx/len(self.traj3d_est) for idx in range(len(self.traj3d_est))],cmap='Reds', alpha=0.95)
                trajectory_frame = np.frombuffer(self.trajectory_fig.canvas.tostring_rgb(), dtype='uint8')
                self.trajectory_frame = trajectory_frame.reshape(self.trajectory_fig.canvas.get_width_height()[::-1] + (3,))
                self.trajectory_image = cv2.resize(self.trajectory_frame, (self.frame_shape[1], self.frame_shape[0]))

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
                        self.recent_long_control = 0
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
            with self.lidar_lock:
                if self.lidar_grabbed:
                    if self.trajectory_image is not None:
                        images = np.hstack((self.lidar_frame, self.trajectory_image))
                    else:
                        images = self.lidar_frame
                    
                    cv2.imshow("Images", images)
                    cv2.waitKey(1)

if __name__ == "__main__":
    cl = ControlLoop()
    cl.start()
