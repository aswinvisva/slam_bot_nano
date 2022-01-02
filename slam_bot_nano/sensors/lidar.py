import os
import termios, fcntl, sys, os

import ydlidar
import numpy as np


class Lidar2D:

    def __init__(self):
        ports = ydlidar.lidarPortList()
        port = "/dev/ydlidar"
        for key, value in ports.items():
            port = value
            
        self.laser = ydlidar.CYdLidar()
        self.laser.setlidaropt(ydlidar.LidarPropSerialPort, port)
        self.laser.setlidaropt(ydlidar.LidarPropSerialBaudrate, 115200)
        self.laser.setlidaropt(ydlidar.LidarPropLidarType, ydlidar.TYPE_TOF)
        self.laser.setlidaropt(ydlidar.LidarPropDeviceType, ydlidar.YDLIDAR_TYPE_SERIAL)
        self.laser.setlidaropt(ydlidar.LidarPropScanFrequency, 10.0)
        self.laser.setlidaropt(ydlidar.LidarPropSampleRate, 20)
        self.laser.setlidaropt(ydlidar.LidarPropSingleChannel, True)
        self.laser.setlidaropt(ydlidar.LidarPropMaxAngle, 180.0)
        self.laser.setlidaropt(ydlidar.LidarPropMinAngle, -180.0)
        self.laser.setlidaropt(ydlidar.LidarPropMaxRange, 32.0)
        self.laser.setlidaropt(ydlidar.LidarPropMinRange, 0.01)
        self.scan = ydlidar.LaserScan()
        ret = self.laser.initialize()

        if ret:
            self.laser.turnOn()
        else:
            raise IOError("Lidar was not initialized")

    def __exit__():
        self.laser.turnOff()
        self.laser.disconnecting()

    def get_points(self):
        r = self.laser.doProcessSimple(self.scan)
        if r:
            angle = []
            ran = []
            intensity = []
            for point in self.scan.points:
                a = (np.pi / 2.0) + ((np.pi / 2.0) - point.angle)
                angle.append(a)
                ran.append(point.range)
                intensity.append(point.intensity)

            return True, angle, ran, intensity

        return False, None, None, None
