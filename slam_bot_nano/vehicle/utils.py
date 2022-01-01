import os

from adafruit_servokit import ServoKit
from jetracer.nvidia_racecar import NvidiaRacecar

def init_bot():

    class Bot:
        def __init__(self):
            self.car = NvidiaRacecar()
            self.car.steering_gain = 1
            self.car.steering_offset = 0
            self.car.throttle_gain = 1
            self.car.steering_motor.set_pulse_width_range(750, 2750)

        def left(self):
            self.car.steering = 0.1

        def right(self):
            self.car.steering = 1

        def straight(self):
            self.car.steering = 0.55

        def forward(self):
            self.car.throttle = -0.3

        def backward(self):
            self.car.throttle = 0.3

        def brake(self):
            self.car.throttle = 0.0

    b = Bot()

    return b
