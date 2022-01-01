import os
import curses
import sys

from adafruit_servokit import ServoKit
import termios, fcntl, sys, os
from jetracer.nvidia_racecar import NvidiaRacecar

from slam_bot_nano.vehicle.utils import init_bot
from slam_bot_nano.controls.keyboard_input import KeyboardInput


def control_loop():
    car = init_bot()
    kb = KeyboardInput()

    print("Controls Ready!")

    c = 'r'

    n_controlless_timesteps = 0

    while c != 'q':
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
