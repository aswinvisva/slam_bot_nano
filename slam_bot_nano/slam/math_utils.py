import numpy as np
import cv2

# [4x4] homogeneous T from [3x3] R and [3x1] t             
def poseRt(R, t):
    ret = np.eye(4)
    ret[:3, :3] = R
    ret[:3, 3] = t
    return ret   

def estimatePose(
    self, 
    kps_ref_unmatched, 
    kps_cur_unmatched, 
    matches, 
    cam, 
    kUseEssentialMatrixEstimation=True, 
    kRansacProb = 0.999,
    kRansacThresholdNormalized = 0.0003,
    img_ref=None,
    img_cur=None):	

    kps_ref = []
    kps_cur = []
    for i,(m) in enumerate(matches):
        # if m.distance < 20:
        kps_cur.append(kps_cur_unmatched[m.trainIdx].pt)
        kps_ref.append(kps_ref_unmatched[m.queryIdx].pt)

    if len(kps_ref) == 0 or len(kps_cur) == 0:
        return None, None, None

    matched_image = None

    if img_ref is not None and img_cur is not None:
        s_m = sorted(matches, key=lambda k: k.distance)

        img_ref.img = np.array(img_ref.img)
        img_cur.img = np.array(img_cur.img)

        matched_image = cv2.drawMatches(img_ref.img,kps_ref_unmatched,img_cur.img,kps_cur_unmatched,s_m[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    kps_cur  = np.asarray(kps_cur)
    kps_ref = np.asarray(kps_ref)

    kp_ref_u = cam.undistort_points(kps_ref)	
    kp_cur_u = cam.undistort_points(kps_cur)	        
    kpn_ref = cam.unproject_points(kp_ref_u)
    kpn_cur = cam.unproject_points(kp_cur_u)

    if kUseEssentialMatrixEstimation:
        # the essential matrix algorithm is more robust since it uses the five-point algorithm solver by D. Nister (see the notes and paper above )
        E, mask_match = cv2.findEssentialMat(kpn_cur, kpn_ref, focal=1, pp=(0., 0.), method=cv2.RANSAC, prob=kRansacProb, threshold=kRansacThresholdNormalized)
    else:
        # just for the hell of testing fundamental matrix fitting ;-) 
        F, mask_match = computeFundamentalMatrix(kp_cur_u, kp_ref_u)
        E = cam.K.T @ F @ cam.K    # E = K.T * F * K 
    #self.removeOutliersFromMask(self.mask)  # do not remove outliers, the last unmatched/outlier features can be matched and recognized as inliers in subsequent frames                          
    
    if E is None or E.shape[0] != 3 or E.shape[1] != 3:
        return None, None, matched_image
    
    _, R, t, mask = cv2.recoverPose(E, kpn_cur, kpn_ref, focal=1, pp=(0., 0.))   
    return R,t,matched_image  # Rrc, trc (with respect to 'ref' frame) 
