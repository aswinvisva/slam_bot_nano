import numpy as np
import cv2

class ORBFeatureExtractor:

    def __init__(self):
        self.extractor = cv2.ORB_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    def extract(self, img):
        kp, des = self.extractor.detectAndCompute(img,None)
        return kp, des

    def match(self, f1, f2):
        des1, des2 = f1.des, f2.des
        
        if des1 is None or des2 is None:
            return []
    
        matches = self.matcher.match(des1,des2)

        return matches

class CameraFrame:

    def __init__(self, img, feature_extractor):
        self.feature_extractor = feature_extractor
        self._kp, self._des = self.feature_extractor.extract(img)
        self._pose = None
        self.img = img

    @property
    def kp(self):
        return self._kp

    @property
    def des(self):
        return self._des

    @property
    def pose(self):
        return self._pose

    @pose.setter
    def pose(self, p):
        self._pose = p
