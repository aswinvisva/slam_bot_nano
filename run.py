import os
import curses
import sys

from adafruit_servokit import ServoKit
import termios, fcntl, sys, os
from jetracer.nvidia_racecar import NvidiaRacecar

from slam_bot_nano.vehicle.utils import init_bot
from slam_bot_nano.controls.keyboard_input import KeyboardInput
from slam_bot_nano.sensors.stereo_cameras import StereoCamera
from slam_bot_nano.client_server_com.image_transfer_service import ImageTransferService


def control_loop():
    car = init_bot()
    kb = KeyboardInput()
    left_camera = StereoCamera(0).start()
    right_camera = StereoCamera(1).start()

    host = '192.168.2.205'
    RemoteDisplay = ImageTransferService.ImageTransferService(host)

    print("Controls Ready!")

    # Check remote display is up
    print(RemoteDisplay.ping())

    c = 'r'

    n_controlless_timesteps = 0

    while c != 'q':
        left_grabbed, left_frame = left_camera.read()
        right_grabbed, right_frame = right_camera.read()

        if left_grabbed and right_grabbed:
            images = np.hstack((left_frame, right_frame))
            cv2.imshow("Camera Images", images)
            RemoteDisplay.sendImage(images)

        try:
            c = kb.getch()

            if c:
                n_controlless_timesteps = 0

                if c == 'd':
                    car.right()
                elif c == 'a':
                    car.left()
                elif c == 'w':
                    car.forward()
                    car.straight()
                elif c == 's':
                    car.backward()
                    car.straight()
                elif c == 'b':
                    car.brake()
            else:
                n_controlless_timesteps += 1

                if n_controlless_timesteps > 500:
                    car.straight()
                    car.brake()

        except IOError:
            car.straight()
            car.brake()

    car.straight()
    car.brake()

if __name__ == "__main__":
    control_loop()
