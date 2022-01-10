import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib

def downsample_image(image, reduce_factor):
	for i in range(0,reduce_factor):
		#Check if image is color or grayscale
		if len(image.shape) > 2:
			row,col = image.shape[:2]
		else:
			row,col = image.shape

		image = cv2.pyrDown(image, dstsize= (col//2, row // 2))
	return image

def get_depth_est(img_L, img_R):
    img_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2GRAY)
    img_R = cv2.cvtColor(img_R, cv2.COLOR_BGR2GRAY)

    img_L = downsample_image(img_L, 2)
    img_R = downsample_image(img_R, 2)

    stereo = cv2.StereoSGBM_create(
        minDisparity= -1,
        numDisparities = 32,
        blockSize = 15,
        uniquenessRatio = 5,
        speckleWindowSize = 5,
        speckleRange = 5,
        disp12MaxDiff = 1,
    ) #32*3*win_size**2)

    disparity = stereo.compute(img_L,img_R)
    
    my_cm = matplotlib.cm.get_cmap('gray')
    normed_data = (disparity - np.min(disparity)) / (np.max(disparity) - np.min(disparity))
    disparity = np.float32(my_cm(normed_data))
    disparity = cv2.cvtColor(disparity, cv2.COLOR_RGBA2BGR)
    disparity = (disparity*255).astype(np.uint8)

    return disparity

# [4x4] homogeneous T from [3x3] R and [3x1] t             
def poseRt(R, t):
    ret = np.eye(4)
    ret[:3, :3] = R
    ret[:3, 3] = t
    return ret   

def drawFeatureTracks(img, kps_ref, kps_cur, mask_match):
    draw_img = img
    num_outliers = 0           

    for i,pts in enumerate(zip(kps_ref, kps_cur)):
        if mask_match[i]:
            p1, p2 = pts 
            a,b = p1.astype(int).ravel()
            c,d = p2.astype(int).ravel()
            cv2.line(draw_img, (a,b),(c,d), (0,255,0), 1)
            cv2.circle(draw_img,(a,b),1, (0,0,255),-1)   
        else:
            num_outliers+=1
    
    return draw_img

def drawMatches(img_ref, kps_ref, img_cur, kps_cur, matches):
    matched_image = cv2.drawMatches(img_ref,kps_ref,img_cur,kps_cur,matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return matched_image

def computeFundamentalMatrix(kps_ref, kps_cur, kRansacThresholdPixels = 0.1, kRansacProb = 0.999):
    F, mask = cv2.findFundamentalMat(kps_ref, kps_cur, cv2.FM_RANSAC)
    if F is None or F.shape == (1, 1):
        # no fundamental matrix found
        return None, None
    elif F.shape[0] > 3:
        # more than one matrix found, just pick the first
        F = F[0:3, 0:3]
    return np.matrix(F), mask 	

def estimatePose(
    ref_frame, 
    cur_frame,
    idxs_ref,
    idxs_cur, 
    cam, 
    kUseEssentialMatrixEstimation=False, 
    kRansacProb = 0.999,
    kRansacThresholdNormalized = 0.0003,
    draw_tracks=True,
    matches=None,
    R_i=None,
    t_i=None):

    kps_ref = np.asarray(ref_frame.kp[idxs_ref])
    kps_cur = np.asarray(cur_frame.kp[idxs_cur])	

    if len(kps_ref) == 0 or len(kps_cur) == 0:
        return None, None, None, None

    kp_ref_u = cam.undistort_points(kps_ref)	
    kp_cur_u = cam.undistort_points(kps_cur)	        
    kp_ref = cam.unproject_points(kp_ref_u)
    kp_cur = cam.unproject_points(kp_cur_u)

    if kUseEssentialMatrixEstimation:
        # the essential matrix algorithm is more robust since it uses the five-point algorithm solver by D. Nister (see the notes and paper above )
        E, mask_match = cv2.findEssentialMat(kp_ref, kp_cur, focal=1, pp=(0., 0.), method=cv2.RANSAC, prob=kRansacProb, threshold=kRansacThresholdNormalized)
    else:
        # just for the hell of testing fundamental matrix fitting ;-) 
        F, mask_match = computeFundamentalMatrix(kp_cur_u, kp_ref_u)

        if F is None:
            return None, None, None, None

        E = cam.K.T @ F @ cam.K    # E = K.T * F * K 
    #self.removeOutliersFromMask(self.mask)  # do not remove outliers, the last unmatched/outlier features can be matched and recognized as inliers in subsequent frames                          

    if E is None or E.shape[0] != 3 or E.shape[1] != 3:
        return None, None, None, None
    
    matched_image = None

    ref_frame.img = np.array(ref_frame.img)
    cur_frame.img = np.array(cur_frame.img)

    if draw_tracks:
        matched_image = drawFeatureTracks(cur_frame.img, kp_ref_u, kp_cur_u, mask_match)
    else:
        matched_image = drawMatches(ref_frame.img, ref_frame.cv_kp, cur_frame.img, cur_frame.cv_kp, matches)

    if R_i is not None and t_i is not None:
        _, R, t, mask = cv2.recoverPose(E, kp_ref, kp_cur, R_i, t_i, focal=1, pp=(0., 0.))   
    else:
        _, R, t, mask = cv2.recoverPose(E, kp_ref, kp_cur, focal=1, pp=(0., 0.))

    origin = np.array([[0, 0, 0],[0, 0, 0]]) # origin point

    fig = plt.figure()
    ax = plt.subplot()

    V = R.dot(t) 

    ax.quiver(*origin, V[0], V[1], color=['r','b','g'], scale=21)
    fig.canvas.draw()
    ax.clear()
    trajectory_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    trajectory_image = trajectory_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return R,t,matched_image, trajectory_image  # Rrc, trc (with respect to 'ref' frame) 
