import numpy as np
import cv2

class ORBFeatureExtractor:

    def __init__(self):
        self.extractor = cv2.ORB_create()
        self.matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    
    def extract(self, img):
        kp, des = orb.detectAndCompute(img1,None)
        return kp, des

    def match(self, f1, f2):
        des1, des2 = f1.des, f2.des
        matches = bf.match(des1,des2)

        return matches
