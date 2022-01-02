import os
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
from slam_bot_nano.client_server_com import ImageTransferServer

def display_info():
    it = ImageTransferServer("192.168.2.212", 9000)
    
    while True:
        images = it.recvall()
        cv2.imshow("Camera Images", images)
        cv2.waitKey(1)

if __name__ == "__main__":
    display_info()
