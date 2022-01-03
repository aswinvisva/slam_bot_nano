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

# https://github.com/geohot/twitchslam/blob/master/helpers.py#L46
def fundamental_to_rt(F):
  W = np.mat([[0,-1,0],[1,0,0],[0,0,1]],dtype=float)
  U,d,Vt = np.linalg.svd(F)
  if np.linalg.det(U) < 0:
    U *= -1.0
  if np.linalg.det(Vt) < 0:
    Vt *= -1.0
  R = np.dot(np.dot(U, W), Vt)
  if np.sum(R.diagonal()) < 0:
    R = np.dot(np.dot(U, W.T), Vt)
  t = U[:, 2]

  # TODO: Resolve ambiguities in better ways. This is wrong.
  if t[2] < 0:
    t *= -1
  
  # TODO: UGLY!
  if os.getenv("REVERSE") is not None:
    t *= -1
  return np.linalg.inv(poseRt(R, t))
