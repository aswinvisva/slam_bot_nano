import numpy as np
import cv2


def compute_fundamental_matrix(f1, f2, matches, distance_threshold=20):
    kp1, kp2 = f1.kp, f2.kp
    pts1, pts2 = [], []

    for i, (m) in enumerate(matches):
        if m.distance < distance_threshold:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1  = np.asarray(pts1)
    pts2 = np.asarray(pts2)

    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_8POINT)
    return F

def poseRt(R, t):
  ret = np.eye(4)
  ret[:3, :3] = R
  ret[:3, 3] = t
  return ret

def estimatePose(self, f1, f2):	
  kp_ref_u = f1.kp	
  kp_cur_u = f2.kp	        
  self.kpn_ref = self.cam.unproject_points(kp_ref_u)
  self.kpn_cur = self.cam.unproject_points(kp_cur_u)
  if kUseEssentialMatrixEstimation:
      # the essential matrix algorithm is more robust since it uses the five-point algorithm solver by D. Nister (see the notes and paper above )
      E, self.mask_match = cv2.findEssentialMat(self.kpn_cur, self.kpn_ref, focal=1, pp=(0., 0.), method=cv2.RANSAC, prob=kRansacProb, threshold=kRansacThresholdNormalized)
  else:
      # just for the hell of testing fundamental matrix fitting ;-) 
      F, self.mask_match = self.computeFundamentalMatrix(kp_cur_u, kp_ref_u)
      E = self.cam.K.T @ F @ self.cam.K    # E = K.T * F * K 
  #self.removeOutliersFromMask(self.mask)  # do not remove outliers, the last unmatched/outlier features can be matched and recognized as inliers in subsequent frames                          
  _, R, t, mask = cv2.recoverPose(E, self.kpn_cur, self.kpn_ref, focal=1, pp=(0., 0.))   
  return R,t  # Rrc, trc (with respect to 'ref' frame)
