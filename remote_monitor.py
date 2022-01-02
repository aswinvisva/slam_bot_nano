import os
import sys
import threading

import termios, fcntl, sys, os
import numpy as np
import cv2

from slam_bot_nano.client_server_com.image_transfer_service import ImageTransferServer

def display_info():
    it = ImageTransferServer()
    
    while True:
        images = it.recvall()

        if images is not None:
            cv2.imshow("Camera Images", images)
            cv2.waitKey(1)

if __name__ == "__main__":
    display_info()
