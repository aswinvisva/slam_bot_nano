import os
import curses
import sys
import threading

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
from slam_bot_nano.slam.camera_frame import CameraFrame
from slam_bot_nano.slam.feature_extractor import ORBFeatureExtractor
from slam_bot_nano.slam.math_utils import *


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

        self.frames = []
        self.feature_extractor = ORBFeatureExtractor()

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

        self.image_sensor_thread.join()
        self.lidar_sensor_thread.join()
        self.controls_thread.join()
        self.display_thread.join()

    def stop(self):
        self.running = False
        self.image_sensor_thread.join()
        self.lidar_sensor_thread.join()
        self.controls_thread.join()
        self.display_thread.join()

    def get_image_sensor_data(self):
        while self.running:   
            with self.camera_lock:
                if len(self.frames) > 0:
                    prev_frame = self.frames[-1]
                else:
                    prev_frame = None

                self.left_grabbed, self.left_frame = self.left_camera.read()
                self.right_grabbed, self.right_frame = self.right_camera.read()

                self.left_frame = cv2.flip(self.left_frame, -1)
                self.right_frame = cv2.flip(self.right_frame, -1)

                frame = CameraFrame(self.left_frame, self.feature_extractor)

                if prev_frame is None:
                    frame.pose = np.array([0, 0])
                else:
                    matches = self.feature_extractor.match(prev_frame, frame)
                    F = compute_fundamental_matrix(prev_frame, frame, matches)
                    Rt = fundamental_to_rt(F)
                    frame.pose = np.dot(Rt, prev_frame.pose)

                self.frames.append(frame)

    def get_lidar_data(self):
        while self.running:   
            with self.lidar_lock:
                self.lidar_grabbed, angle, ran, intensity = self.lidar.get_points()
                self.lidar_frame = None

                if self.lidar_grabbed:
                    self.fig.canvas.draw()
                    self.lidar_polar.clear()
                    self.lidar_polar.scatter(angle, ran, c=intensity, cmap='hsv', alpha=0.95)
                    lidar_frame = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype='uint8')
                    self.lidar_frame = lidar_frame.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
                    self.lidar_frame = cv2.resize(self.lidar_frame, (self.left_frame.shape[1], self.left_frame.shape[0])) 

    def get_controls(self):
        c = 'r'
        while self.running and c != 'q':
            try:            
                c = self.kb.getch()

                if c:
                    self.frames_without_controls = 0

                    if c == 'd':
                        self.car.right()
                    elif c == 'a':
                        self.car.left()
                    elif c == 'w':
                        self.car.forward()
                        self.car.straight()
                    elif c == 's':
                        self.car.backward()
                        self.car.straight()
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
                        images = np.hstack((self.left_frame, self.right_frame, self.lidar_frame))
                        cv2.imshow("Camera Images", images)
                        cv2.waitKey(1)

if __name__ == "__main__":
    cl = ControlLoop()
    cl.start()
