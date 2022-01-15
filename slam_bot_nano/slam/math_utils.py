import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
from skimage.measure import ransac
from skimage.transform import EssentialMatrixTransform
from sklearn import linear_model, datasets
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors


class Intrinsics:
    """Camera intrinsics object
    """
    def __init__(self, mat):
        self._mat = mat

    @property
    def mat(self):
        """ (array, [3x3]): intrinsics matrix """
        return self._mat

    @mat.setter
    def mat(self, mat):
        self._mat = mat

    @property
    def inv_mat(self):
        """ (array, [3x3]): inverse intrinsics matrix """
        return np.linalg.inv(self._mat)

    @inv_mat.setter
    def inv_mat(self, mat):
        self._mat = np.linalg.inv(mat)

    @property
    def fx(self):
        """ float: focal length in x-direction """
        return self._mat[0, 0]

    @fx.setter
    def fx(self, value):
        self._mat[0, 0] = value

    @property
    def fy(self):
        """ float: focal length in y-direction """
        return self._mat[1, 1]

    @fy.setter
    def fy(self, value):
        self._mat[1, 1] = value

    @property
    def cx(self):
        """ float: principal point in x-direction """
        return self._mat[0, 2]

    @cx.setter
    def cx(self, value):
        self._mat[0, 2] = value

    @property
    def cy(self):
        """ float: principal point in y-direction """
        return self._mat[1, 2]

    @cy.setter
    def cy(self, value):
        self._mat[1, 2] = value

def compute_pose_2d2d(kp_ref, kp_cur, K, valid_cfg="flow", valid_cfg_thresh=0.01, reproj_thre=0.0003):
    """Compute the pose from view2 to view1
    
    Args:
        kp_ref (array, [Nx2]): keypoints for reference view
        kp_cur (array, [Nx2]): keypoints for current view
        cam_intrinsics (Intrinsics): camera intrinsics
    
    Returns:
        a dictionary containing
            - **pose** (SE3): relative pose from current to reference view
            - **best_inliers** (array, [N]): boolean inlier mask
    """
    cam_intrinsics = Intrinsics(K)
    principal_points = (cam_intrinsics.cx, cam_intrinsics.cy)

    # validity check
    valid_case = True

    # initialize ransac setup
    R = np.eye(3)
    t = np.zeros((3,1))
    best_Rt = [R, t]
    best_inlier_cnt = 0
    max_ransac_iter = 3
    best_inliers = np.ones((kp_ref.shape[0], 1)) == 1
    inliers = [False for i in range(len(kp_ref))]

    if valid_cfg == "flow":
        # check flow magnitude
        avg_flow = np.mean(np.linalg.norm(kp_ref-kp_cur, axis=1))
        valid_case = avg_flow > valid_cfg_thresh        
    elif valid_cfg == "homo_ratio":
        # Find homography
        H, H_inliers = cv2.findHomography(
                    kp_cur,
                    kp_ref,
                    method=cv2.RANSAC,
                    confidence=0.99,
                    ransacReprojThreshold=0.2,
                    )
    elif valid_cfg == "GRIC":
        if kp_cur.shape[0] > 10:
            H, H_inliers = cv2.findHomography(
                        kp_cur,
                        kp_ref,
                        method=cv2.RANSAC,
                        confidence=0.99,
                        ransacReprojThreshold=1,
                        )

            H_res = compute_homography_residual(H, kp_cur, kp_ref)
            H_gric = calc_GRIC(
                        res=H_res,
                        sigma=0.8,
                        n=kp_cur.shape[0],
                        model="HMat"
            )
        else:
            valid_case = False

    
    if valid_case:
        num_valid_case = 0
        for i in range(max_ransac_iter): # repeat ransac for several times for stable result
            # shuffle kp_cur and kp_ref (only useful when random seed is fixed)	
            new_list = np.arange(0, kp_cur.shape[0], 1)	
            np.random.shuffle(new_list)
            new_kp_cur = kp_cur.copy()[new_list]
            new_kp_ref = kp_ref.copy()[new_list]

            E, inliers = cv2.findEssentialMat(
                        new_kp_cur,
                        new_kp_ref,
                        focal=cam_intrinsics.fx,
                        pp=principal_points,
                        method=cv2.RANSAC,
                        prob=0.99,
                        threshold=reproj_thre,
                        )

            if E is None or E.shape[0] != 3 or E.shape[1] != 3:
                continue

            # check homography inlier ratio
            if valid_cfg == "homo_ratio":
                H_inliers_ratio = H_inliers.sum()/(H_inliers.sum()+inliers.sum())
                valid_case = H_inliers_ratio < valid_cfg_thresh
                # print("valid: {} ratio: {}".format(valid_case, H_inliers_ratio))

                # inlier check
                inlier_check = inliers.sum() > best_inlier_cnt
            elif valid_cfg == "flow":
                cheirality_cnt, R, t, _ = cv2.recoverPose(E, new_kp_cur, new_kp_ref,
                                        focal=cam_intrinsics.fx,
                                        pp=principal_points)
                valid_case = cheirality_cnt > kp_cur.shape[0]*0.1
                
                # inlier check
                inlier_check = inliers.sum() > best_inlier_cnt and cheirality_cnt > kp_cur.shape[0]*0.05               
            elif valid_cfg == "GRIC":
                # get F from E
                K = cam_intrinsics.mat
                F = np.linalg.inv(K.T) @ E @ np.linalg.inv(K)
                E_res = compute_fundamental_residual(F, new_kp_cur, new_kp_ref)

                E_gric = calc_GRIC(
                    res=E_res,
                    sigma=0.8,
                    n=kp_cur.shape[0],
                    model='EMat'
                )
                valid_case = H_gric > E_gric

                # inlier check
                inlier_check = inliers.sum() > best_inlier_cnt

            # save best_E
            if inlier_check:
                best_E = E
                best_inlier_cnt = inliers.sum()

                revert_new_list = np.zeros_like(new_list)
                for cnt, i in enumerate(new_list):
                    revert_new_list[i] = cnt
                best_inliers = inliers[list(revert_new_list)]
            num_valid_case += (valid_case * 1)

        major_valid = num_valid_case > (max_ransac_iter/2)
        if major_valid:
            cheirality_cnt, R, t, _ = cv2.recoverPose(best_E, kp_cur, kp_ref,
                                    focal=cam_intrinsics.fx,
                                    pp=principal_points,
                                    )
            # cheirality_check
            if cheirality_cnt > kp_cur.shape[0]*0.1:
                best_Rt = [R, t]

    R, t = best_Rt
    return R, t, inliers

def fundamentalToRt(F):
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
    return R, t.reshape((3,1))

def rigid_transform_3D(A, B):
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    # if num_rows != 3:
    #     raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    # if num_rows != 3:
    #     raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t

def icp(a, b, init_pose=(0,0,0), no_iterations = 13):
    '''
    The Iterative Closest Point estimator.
    Takes two cloudpoints a[x,y], b[x,y], an initial estimation of
    their relative pose and the number of iterations
    Returns the affine transform that transforms
    the cloudpoint a to the cloudpoint b.
    Note:
        (1) This method works for cloudpoints with minor
        transformations. Thus, the result depents greatly on
        the initial pose estimation.
        (2) A large number of iterations does not necessarily
        ensure convergence. Contrarily, most of the time it
        produces worse results.
    '''

    src = np.array(a, copy=True).astype(np.float32)
    dst = np.array(b, copy=True).astype(np.float32)

    #Initialise with the initial pose estimation
    Tr = np.array([[np.cos(init_pose[2]),-np.sin(init_pose[2]),init_pose[0]],
                   [np.sin(init_pose[2]), np.cos(init_pose[2]),init_pose[1]],
                   [0,                    0,                   1          ]])

    print(src.shape, Tr[0:2])

    src = cv2.transform(src, Tr[0:2])

    for i in range(no_iterations):
        #Find the nearest neighbours between the current source and the
        #destination cloudpoint
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto',
                                warn_on_equidistant=False).fit(dst[0])
        distances, indices = nbrs.kneighbors(src[0])

        #Compute the transformation between the current source
        #and destination cloudpoint
        T = cv2.estimateRigidTransform(src, dst[0, indices.T], False)
        #Transform the previous source and update the
        #current source cloudpoint
        src = cv2.transform(src, T)
        #Save the transformation from the actual source cloudpoint
        #to the destination
        Tr = np.dot(Tr, np.vstack((T,[0,0,1])))
    return Tr[0:2]

def lidar_localization(kp_ref, kp_cur):
    """
    Given the map at t-1 and the map at t, compute inliers and overall trajectory in the XY plane
    """ 
    kp_ref = pol2cart(*kp_ref)
    kp_cur = pol2cart(*kp_cur)

    x = np.vstack(kp_ref).T
    y = np.vstack(kp_cur).T

    cost = cdist(x, y, 'euclidean')

    row_ind, col_ind = linear_sum_assignment(cost)

    print(x[row_ind].shape, y[col_ind].shape)

    print("Total matching cost: %d" % cost[row_ind, col_ind].sum())

    # R, t = rigid_transform_3D(x[row_ind].T, y[col_ind].T)

    T = icp(x[row_ind], y[col_ind])

    print(T)

    return T

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, phi)   

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
    kUseEssentialMatrixEstimation=True, 
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

    R, t, mask_match = compute_pose_2d2d(kp_ref, kp_cur, cam.K)

    # if kUseEssentialMatrixEstimation:
    #     # the essential matrix algorithm is more robust since it uses the five-point algorithm solver by D. Nister (see the notes and paper above )
    #     E, mask_match = cv2.findEssentialMat(kp_ref, kp_ref, cameraMatrix=cam.K, method=cv2.RANSAC, prob=kRansacProb, threshold=kRansacThresholdNormalized)
    #     # print("Inlier:", kp_ref[np.where(mask_match == 1)[0]][0:5])
    # else:
    #     # just for the hell of testing fundamental matrix fitting ;-) 
    #     F, mask_match = computeFundamentalMatrix(kp_cur_u, kp_ref_u)

    #     if F is None:
    #         return None, None, None, None

    #     E = cam.K.T @ F @ cam.K    # E = K.T * F * K 
    # #self.removeOutliersFromMask(self.mask)  # do not remove outliers, the last unmatched/outlier features can be matched and recognized as inliers in subsequent frames                          

    # if E is None or E.shape[0] != 3 or E.shape[1] != 3:
    #     return None, None, None, None
    
    matched_image = None

    ref_frame.img = np.array(ref_frame.img)
    cur_frame.img = np.array(cur_frame.img)

    if draw_tracks:
        if mask_match is not None:
            matched_image = drawFeatureTracks(cur_frame.img, kp_ref_u, kp_cur_u, mask_match)
    else:
        matched_image = drawMatches(ref_frame.img, ref_frame.cv_kp, cur_frame.img, cur_frame.cv_kp, matches)

    # if R_i is not None and t_i is not None:
    #     _, R, t, mask = cv2.recoverPose(E, kp_ref, kp_cur, R_i, t_i, focal=1, pp=(0., 0.))   
    # else:
    #     _, R, t, mask = cv2.recoverPose(E, kp_ref, kp_cur, focal=1, pp=(0., 0.))

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
