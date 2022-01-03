import numpy as np
import cv2


def estimatePose(self, kps_ref_unmatched, kps_cur_unmatched, matches, cam, kUseEssentialMatrixEstimation=True):	

    kps_ref = []
    kps_cur = []
    for i,(m) in enumerate(matches):
        if m.distance < 20:
            kps_cur.append(kps_cur_unmatched[m.trainIdx].pt)
            kps_ref.append(kps_ref_unmatched[m.queryIdx].pt)

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
    _, R, t, mask = cv2.recoverPose(E, kpn_cur, self.kpn_ref, focal=1, pp=(0., 0.))   
    return R,t  # Rrc, trc (with respect to 'ref' frame) 
