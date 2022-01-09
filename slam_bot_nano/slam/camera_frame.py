from collections import defaultdict

import numpy as np
import cv2

class ORBFeatureExtractor:

    def __init__(self):
        self.extractor = cv2.ORB_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    def extract(self, img):
        kp, des = self.extractor.detectAndCompute(img,None)
        return kp, des

    def match(self, f1, f2):
        des1, des2 = f1.des, f2.des
        
        if des1 is None or des2 is None:
            return []
    
        matches = self.matcher.knnMatch(des1, des2, k=2)

        return self.goodMatchesOneToOne(matches, des1, des2)

    # input: des1 = query-descriptors, des2 = train-descriptors
    # output: idx1, idx2  (vectors of corresponding indexes in des1 and des2, respectively)
    # N.B.: this returns matches where each trainIdx index is associated to only one queryIdx index    
    def goodMatchesOneToOne(self, matches, des1, des2, ratio_test=0.7):
        len_des2 = len(des2)
        idx1, idx2 = [], []
        end_matches = []  
        
        if matches is not None:         
            float_inf = float('inf')
            dist_match = defaultdict(lambda: float_inf)   
            index_match = dict()  
            for m, n in matches:
                if m.distance > ratio_test * n.distance:
                    continue     
                dist = dist_match[m.trainIdx]
                if dist == float_inf: 
                    # trainIdx has not been matched yet
                    dist_match[m.trainIdx] = m.distance
                    idx1.append(m.queryIdx)
                    idx2.append(m.trainIdx)
                    end_matches.append(m)
                    index_match[m.trainIdx] = len(idx2)-1
                else:
                    if m.distance < dist: 
                        # we have already a match for trainIdx: if stored match is worse => replace it
                        #print("double match on trainIdx: ", m.trainIdx)
                        index = index_match[m.trainIdx]
                        assert(idx2[index] == m.trainIdx) 
                        idx1[index]=m.queryIdx
                        idx2[index]=m.trainIdx
                        end_matches[index] = m   

        assert len(idx1) == len(idx2) == len(end_matches)
        assert all([idx2[i] == end_matches[i].trainIdx for i in range(len(idx1))])
        assert all([idx1[i] == end_matches[i].queryIdx for i in range(len(idx1))])

        return idx1, idx2, end_matches

class CameraFrame:

    def __init__(self, img, feature_extractor):
        self.feature_extractor = feature_extractor
        self._kp_original, self._des = self.feature_extractor.extract(img)
        self._kp = np.array([x.pt for x in self._kp_original], dtype=np.float32)
        self._kp_original = np.array([x for x in self._kp_original])

        self._pose = None
        self.img = img

    @property
    def kp(self):
        return self._kp

    @property
    def cv_kp(self):
        return self._kp_original

    @property
    def des(self):
        return self._des

    @property
    def pose(self):
        return self._pose

    @pose.setter
    def pose(self, p):
        self._pose = p
