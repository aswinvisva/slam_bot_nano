import os
import curses
import sys
import threading
import time
import signal

import cv2
import numpy as np

from slam_bot_nano.vehicle.utils import init_bot
from slam_bot_nano.controls.keyboard_input import KeyboardInput
from slam_bot_nano.sensors.stereo_camera import StereoCamera
from slam_bot_nano.sensors.lidar import Lidar2D
from slam_bot_nano.client_server_com.image_transfer_service import ImageTransferClient
from slam_bot_nano.slam.camera_frame import CameraFrame, ORBFeatureExtractor
from slam_bot_nano.slam.math_utils import *

def mkdir_p(mypath: str):
    """
    Creates a directory. equivalent to using mkdir -p on the command line
    :param str, mypath: Path to create
    """

    from errno import EEXIST
    from os import makedirs, path

    try:
        makedirs(mypath)
    except OSError as exc:  # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else:
            raise

def signal_handler(sig, frame):
    RUNNING = False

RUNNING = False

class Recorder:
    def __init__(self):
        self.car = init_bot()
        self.kb = KeyboardInput()
        self.lidar = Lidar2D()
        self.frames_without_controls = 0
        self.most_recent_control = np.zeros((2, 1))
        self.controls_lock = threading.Lock()

        signal.signal(signal.SIGINT, signal_handler)

    def start(self):
        global RUNNING

        if RUNNING:
            print("Already running!")
            return

        self.start_time = time.time()
        self.run_path = "slam_bot_nano/data/runs/%s" % str(self.start_time)
        mkdir_p(self.run_path)

        RUNNING = True

        self.record_thread = threading.Thread(target=self.record_data, daemon=True)
        self.record_thread.start()

        self.controls_thread = threading.Thread(target=self.get_controls, daemon=True)
        self.controls_thread.start()

        self.controls_thread.join()
        
    def record_data(self):
        while RUNNING:
            elapsed_time = time.time() - self.start_time

            lidar_grabbed, angle, ran, intensity = self.lidar.get_points() 

            if lidar_grabbed:
                data_path = os.path.join(self.run_path, str(elapsed_time))
                mkdir_p(data_path)

                np.savez(os.path.join(data_path, 'lidar.npz'), angle=angle, ran=ran, intensity=intensity) 

                with self.controls_lock:
                    np.savez(os.path.join(data_path, 'controls.npz'), long=self.most_recent_control[0], lat=self.most_recent_control[1]) 

    def get_controls(self):
        global RUNNING
        
        c = 'r'
        while RUNNING and c != 'q':
            with self.controls_lock:
                try:            
                    c = self.kb.getch()

                    if c:
                        self.frames_without_controls = 0

                        if c == 'd':
                            self.car.right()
                            self.most_recent_control[1] = 1
                        elif c == 'a':
                            self.car.left()
                            self.most_recent_control[1] = -1
                        elif c == 'w':
                            self.car.forward()
                            self.car.straight()
                            self.most_recent_control[0] = 1
                            self.most_recent_control[1] = 0
                        elif c == 's':
                            self.car.backward()
                            self.car.straight()
                            self.most_recent_control[0] = -1
                            self.most_recent_control[1] = 0
                        elif c == 'b':
                            self.car.brake()
                            self.most_recent_control[0] = 0
                    else:
                        self.frames_without_controls += 1

                        if self.frames_without_controls > 500000:
                            self.car.straight()
                            self.car.brake()
                            self.most_recent_control[0] = 0
                            self.most_recent_control[1] = 0
                except IOError:
                    self.car.straight()
                    self.car.brake()

        RUNNING = False

if __name__ == "__main__":
    cl = Recorder()
    cl.start()