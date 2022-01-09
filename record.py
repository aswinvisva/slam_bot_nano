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
        self.left_camera = StereoCamera(0).start()
        self.right_camera = StereoCamera(1).start()
        self.lidar = Lidar2D()
        self.frames_without_controls = 0

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

        self.controls_thread = threading.Thread(target=self.get_controls, daemon=True)
        self.controls_thread.start()

        self.record_thread = threading.Thread(target=self.record_data, daemon=True)
        self.record_thread.start()

        self.controls_thread.join()
        self.record_thread.join()
        
    def record_data(self):
        while RUNNING:
            elapsed_time = time.time() - self.start_time

            lidar_grabbed, angle, ran, intensity = self.lidar.get_points()                
            left_grabbed, left_frame = self.left_camera.read()
            right_grabbed, right_frame = self.right_camera.read()

            if left_grabbed and right_grabbed and lidar_grabbed:
                data_path = os.path.join(self.run_path, str(elapsed_time))
                mkdir_p(data_path)

                left_frame = cv2.flip(left_frame, -1)
                right_frame = cv2.flip(right_frame, -1)

                cv2.imwrite(os.path.join(data_path, 'left_frame.jpg'), left_frame) 
                cv2.imwrite(os.path.join(data_path, 'right_frame.jpg'), right_frame)
                np.savez(os.path.join(data_path, 'lidar.npz'), angle=angle, ran=ran, intensity=intensity) 

    def get_controls(self):
        global RUNNING
        
        c = 'r'
        while RUNNING and c != 'q':
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

        RUNNING = False

class Replay:
    def __init__(self, data_path):
        self.subfolders = sorted([f.path for f in os.scandir(folder) if f.is_dir()], key=lambda k: float(k))
        self.idx = 0

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

if __name__ == "__main__":
    cl = Recorder()
    cl.start()