import numpy as np
import cv2

from feature_extractor import ORBFeatureExtractor

class CameraFrame:

    def __init__(self, img, feature_extractor):
        self.feature_extractor = feature_extractor
        self._kp, self._des = self.feature_extractor.extract(img)
        self._pose = None

    @property
    def kp(self):
        return self._kp

    @property
    def des(self):
        return self._des

    @property
    def pose(self):
        return self._pose
