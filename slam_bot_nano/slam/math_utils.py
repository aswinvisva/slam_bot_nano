import os
import math

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
from scipy.optimize import fmin_cg
import numpy as np
from sklearn.neighbors import KDTree


class Point_cloud:
    def __init__(self):
        """
        path : path to the ply files
        """

        self.all_eigenvalues = None
        self.all_eigenvectors = None
        self.n = 0
        self.kdtree = None
        self.points = None
        self.nn = None

    def save(self,path):
        """
        Write the point cloud to a file
        params:
            path: output path
        """
        write_ply(path,self.points,["x","y","z"])

    def init_from_ply(self,path):
        """
        Initialize point cloud from ply file
        params:
            path: input path
        """
        tmp = read_ply(path)
        self.points = np.vstack((tmp['x'],tmp['y'],tmp['z'])).T
        self._init()


    def init_from_transfo(self, initial, R = None,t = None):
        """
        Initialize a point cloud from another point cloud and a tranformation R x + t
        params:
            initial: input point cloud object
            R : rotation matrix to apply to initial
            t : transormation
        """
        if R is None:
            R = np.eye(3)
        if t is None:
            t = np.zeros(3)

        self.points = initial.points @ R.T + t
        self._init()

    def init_from_points(self,points):
        """
        Initialize a point cloud from a list of points
        params:
            points: numpy array
        """
        self.points = points.copy()
        self._init()

        return self

    def _init(self):
        """
        Common function for every initialization function
        """
        self.kdtree = KDTree(self.points)
        self.n  = self.points.shape[0]
        self.all_eigenvalues = None
        self.all_eigenvectors = None
        self.nn = None

    def transform(self,R,T):
        """
        Apply transformation Rx+t to the point cloud
        """
        self.points = self.points @ R.T + T
        if not self.all_eigenvectors is None:
            self.all_eigenvectors = self.all_eigenvectors @ R.T

    def neighborhood_PCA(self, radius = 0.005):
        """
        Returns the eigenvalues, eigenvectors for each points on his neighbors
        The neigbours are computed with query_radius of kdtree
        params:
            radius: radius to find the list of nearest neighbors
        returns:
            tuple(eigenvalues, eigenvectors) of dimension (n,3) and (n,3,3)
        """
        if self.nn is None:
            self.nn = self.kdtree.query_radius(self.points,r = radius, return_distance = False)

        all_eigenvalues = np.zeros((self.n, 3))
        all_eigenvectors = np.zeros((self.n, 3, 3))

        for i in range(self.n):
            if len(self.nn[i]) < 3:
                all_eigenvalues[i], all_eigenvectors[i] = (np.array([0,0,0]),np.eye(3))
            else:
                all_eigenvalues[i], all_eigenvectors[i] = np.linalg.eigh(np.cov(self.points[self.nn[i]].T))

        return all_eigenvalues, all_eigenvectors

    def get_eigenvectors(self, radius = 0.005):
        """
        Returns the eigenvectors on the neighbors PCA. Used for computing it only
        once
        params:
            radius: radius to find the list of nearest neighbors NB: if the PCA have
                already been performed once, the PCA is not done again even if
                radius is different than the original
        """
        if self.all_eigenvectors is None:
            self.all_eigenvalues,self.all_eigenvectors = self.neighborhood_PCA(radius)
        return self.all_eigenvectors

    def get_projection_matrix_point2plane(self, indexes = None):
        """
        Get the projection matrix on the plane defined at each point for point2plane
        params:
            indexes: integer np array, indexes for which the projection matrix
            must be computed
        Returns:
            array of Projection matrices on each of the normals
        """
        if indexes is None:
            indexes = np.arange(self.n)

        all_eigenvectors = self.get_eigenvectors()
        normals = all_eigenvectors[:,:,0]
        normals = normals / np.linalg.norm(normals, axis = 1, keepdims = True)
        return np.array([normals[i,:,None]*normals[i,None,:] for i in indexes])

    def get_covariance_matrices_plane2plane(self, epsilon  = 1e-3,indexes = None):
        """
        Returns C_A covariance matrix used for plane2plane
        params:
            epsilon: value of the covariance in the normal direction
            indexes: integer np array, indexes for which the covariance matrix
            must be computed
        returns: array of covariance matrices for each point of indexes
        """
        if indexes is None:
            indexes = np.arange(self.n)
        d = 3
        new_n = indexes.shape[0]
        cov_mat = np.zeros((new_n,d,d))
        all_eigenvectors = self.get_eigenvectors()
        dz_cov_mat = np.eye(d)
        dz_cov_mat[0,0] = epsilon
        for i in range(new_n):
            U = all_eigenvectors[indexes[i]]
            cov_mat[i,:,:] = U @ dz_cov_mat @ U.T

        return cov_mat

def elementary_rot_mat(theta):
    """
    Returns the 3 rotations around each axe
    """


    R_x = np.array([[1,         0,                 0                ],
                    [0,         np.cos(theta[0]), -np.sin(theta[0]) ],
                    [0,         np.sin(theta[0]), np.cos(theta[0])  ]
                    ])



    R_y = np.array([[np.cos(theta[1]),    0,      np.sin(theta[1])  ],
                    [0,                   1,      0                 ],
                    [-np.sin(theta[1]),   0,      np.cos(theta[1])  ]
                    ])

    R_z = np.array([[np.cos(theta[2]),    -np.sin(theta[2]),    0],
                    [np.sin(theta[2]),    np.cos(theta[2]),     0],
                    [0,                   0,                    1]
                    ])

    return R_x, R_y, R_z

def rot_mat(theta) :
    """
    Returns the Rotation matrix for the rotation parametrized with theta
    Convention rotation around X then around Y then around Z
    """
    R_x,R_y,R_z = elementary_rot_mat(theta)
    R = R_z @ R_y @ R_x

    return R


def grad_rot_mat(theta):
    """
    Computes the gradient of the rotation matrix w.r.t the X,Y,Z euler angles
    Returns res[i,j,k] = dR_jk/theta_i
    """
    res = np.zeros((3,3,3))

    R_x,R_y,R_z = elementary_rot_mat(theta)

    g_x = np.array([[0,         0,                  0                ],
                    [0,         -np.sin(theta[0]), -np.cos(theta[0]) ],
                    [0,         np.cos(theta[0]),  -np.sin(theta[0]) ]
                    ])

    g_y = np.array([[-np.sin(theta[1]),   0,      np.cos(theta[1])  ],
                    [0,                   0,      0                 ],
                    [-np.cos(theta[1]),   0,      -np.sin(theta[1]) ]
                    ])

    g_z = np.array([[-np.sin(theta[2]),   -np.cos(theta[2]),    0],
                    [np.cos(theta[2]),    -np.sin(theta[2]),    0],
                    [0,                   0,                    0]
                    ])

    res[0,:,:] = R_z @ R_y @ g_x
    res[1,:,:] = R_z @ g_y @ R_x
    res[2,:,:] = g_z @ R_y @ R_x

    return res

def best_transform(data, ref, method = "point2point", indexes_d = None, indexes_r = None, verbose = True):
    """
    Returns the best transformation computed for the two aligned point clouds
    params:
        data: point cloud to align (shape n*3)
        ref: reference point cloud (shape n*3)
        method: must be one of : point2point, point2plane, plane2plane
        indexes_d: integer array Indexes and order to take into account in data
        indexes_r: integer array Indexes and order to take into account in ref
        verbose: Whether to plot the result of the iterations of conjugate gradient in plane2plane
    Returns:
        R: a rotation matrix (shape 3*3)
        t: translation (length 3 vector)
    """

    if indexes_d is None:
        indexes_d = np.arange(data.shape[0])
    if indexes_r is None:
        indexes_r = np.arange(ref.shape[0])

    assert(indexes_d.shape == indexes_r.shape)
    n = indexes_d.shape[0]
    if method == "point2point":
        x0 = np.zeros(6)
        M = np.array([np.eye(3) for i in range(n)])
        f = lambda x: loss(x,data.points[indexes_d],ref.points[indexes_r],M)
        df = lambda x: grad_loss(x,data.points[indexes_d],ref.points[indexes_r],M)

        x = fmin_cg(f = f,x0 = x0,fprime = df, disp = False)

    elif method == "point2plane":
        x0 = np.zeros(6)
        M = ref.get_projection_matrix_point2plane(indexes = indexes_r)
        f = lambda x: loss(x,data.points[indexes_d],ref.points[indexes_r],M)
        df = lambda x: grad_loss(x,data.points[indexes_d],ref.points[indexes_r],M)

        x = fmin_cg(f = f,x0 = x0,fprime = df, disp = False)

    elif method == "plane2plane":
        cov_data = data.get_covariance_matrices_plane2plane(indexes = indexes_d)
        cov_ref = ref.get_covariance_matrices_plane2plane(indexes = indexes_r, epsilon = 0.01)

        last_min = np.inf
        cpt = 0
        n_iter_max = 50
        x = np.zeros(6)
        tol = 1e-6
        while True:
            cpt = cpt+1
            R = rot_mat(x[3:])
            M = np.array([np.linalg.inv(cov_ref[i] + R @ cov_data[i] @ R.T) for i in range(n)])

            f = lambda x: loss(x,data.points[indexes_d],ref.points[indexes_r],M)
            df = lambda x: grad_loss(x,data.points[indexes_d],ref.points[indexes_r],M)

            out = fmin_cg(f = f, x0 = x, fprime = df, disp = False, full_output = True)

            x = out[0]
            f_min = out[1]
            if verbose:
                print("\t\t EM style iteration {} with loss {}".format(cpt,f_min))

            if last_min - f_min < tol:
                if verbose:
                    print("\t\t\t Stopped EM because not enough improvement or not at all")
                break
            elif cpt >= n_iter_max:
                if verbose:
                    print("\t\t\t Stopped EM because maximum number of iterations reached")
                break
            else:
                last_min = f_min

    else:
        print("Error, unknown method : {}".format(method))
        return

    t = x[0:3]
    R = x[3:]

    return rot_mat(R),t

def loss(x,a,b,M):
    """
    loss for parameter x
    params:
        x : length 6 vector of transformation parameters
            (t_x,t_y,t_z, theta_x, theta_y, theta_z)
        a : data to align n*3
        b : ref point cloud n*3 a[i] is the nearest neibhor of Rb[i]+t
        M : central matrix for each data point n*3*3 (cf loss equation)
    returns:
        Value of the loss function
    """
    t = x[:3]
    R = rot_mat(x[3:])
    residual = b - a @ R.T -t[None,:] # shape n*d
    tmp = np.sum(M * residual[:,None,:], axis = 2) # shape n*d
    return np.sum(residual * tmp)

def grad_loss(x,a,b,M):
    """
    Gradient of the loss loss for parameter x
    params:
        x : length 6 vector of transformation parameters
            (t_x,t_y,t_z, theta_x, theta_y, theta_z)
        a : data to align n*3
        b : ref point cloud n*3 a[i] is the nearest neibhor of Rb[i]+t
        M : central matrix for each data point n*3*3 (cf loss equation)
    returns:
        Value of the gradient of the loss function
    """
    t = x[:3]
    R = rot_mat(x[3:])
    g = np.zeros(6)
    residual = b - a @ R.T -t[None,:] # shape n*d
    tmp = np.sum(M * residual[:,None,:], axis = 2) # shape n*d

    g[:3] = - 2*np.sum(tmp, axis = 0)

    grad_R = - 2* (tmp.T @ a) # shape d*d
    grad_R_euler = grad_rot_mat(x[3:]) # shape 3*d*d
    g[3:] = np.sum(grad_R[None,:,:] * grad_R_euler, axis = (1,2)) # chain rule
    return g

def ICP(data,ref,method, exclusion_radius = 0.5, sampling_limit = None, verbose = True):
    """
    Full algorithm
    Aligns the two point cloud by iteratively matching the closest points
    params:
        data: point cloud to align (shape N*3)
        ref:
        method: one of point2point, point2plane, plane2plane
        exclusion_radius: threshold to discard pairs of point with too high distance
        sampling_limit: number of point to consider for huge point clouds
        verbose: whether to plot the results of the iterations and verbose of intermediate functions
    returns:
        R: rotation matrix (shape 3*3)
        T: translation (length 3)
        rms_list: list of rms at the end of each ICP iteration
    """

    data_aligned = Point_cloud()
    data_aligned.init_from_transfo(data)

    rms_list = []
    cpt = 0
    max_iter = 50
    dist_threshold = exclusion_radius
    RMS_threshold = 1e-4
    diff_thresh = 1e-3
    rms = np.inf
    while(True):
        if sampling_limit is None:
            samples = np.arange(data.n)
        else:
            samples = np.random.choice(data.n,size = sampling_limit,replace = False)

        dist,neighbors = ref.kdtree.query(data_aligned.points[samples], return_distance = True)

        dist = dist.flatten()
        neighbors = neighbors.flatten()

        indexes_d = samples[dist < dist_threshold]
        indexes_r = neighbors[dist < dist_threshold]

        R, T = best_transform(data, ref, method, indexes_d, indexes_r, verbose = verbose)
        data_aligned.init_from_transfo(data, R,T)
        new_rms = np.sqrt(np.mean(np.sum((data_aligned.points[samples]-ref.points[neighbors])**2,axis = 0)))
        rms_list.append(new_rms)
        if verbose:
            print("Iteration {} of ICP complete with RMS : {}".format(cpt+1,new_rms))

        if new_rms < RMS_threshold :
            if verbose:
                print("\t Stopped because very low rms")
            break
        elif rms - new_rms < 0:

            if verbose:
                print("\t Stopped because increasing rms")
            break
        elif rms-new_rms < diff_thresh:
            if verbose:
                print("\t Stopped because convergence of the rms")
            break
        elif cpt >= max_iter:
            if verbose:
                print("\t Max iter reached")
            break
        else:
            rms = new_rms
            cpt = cpt+1

    return R,T, indexes_d, indexes_r


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

def lidar_localization(kp_ref, kp_cur, slam_map, cum_R=None, cum_t=None):
    """
    Given the map at t-1 and the map at t, compute inliers and overall trajectory in the XY plane
    """
    kp_ref = pol2cart(*kp_ref)
    kp_cur = pol2cart(*kp_cur)

    x = np.vstack(kp_ref).T
    y = np.vstack(kp_cur).T

    x = np.append(x, np.zeros((x.shape[0], 1)), axis=1)
    y = np.append(y, np.zeros((y.shape[0], 1)), axis=1)

    x = Point_cloud().init_from_points(x)
    y = Point_cloud().init_from_points(y)

    R, t, row_indices, col_indices = ICP(x, y, "point2point")
    t = t.reshape((3,1))

    cum_t = cum_t + cum_R.dot(t)
    cum_R = cum_R.dot(R)

    cum_t = t.reshape((3,))

    Rt = poseRt(cum_R, cum_t)
    Rt_inv = np.linalg.inv(Rt)
    R_inv, t_inv = Rtpose(Rt_inv)

    transformed_point_cloud = Point_cloud()
    transformed_point_cloud.init_from_transfo(y, R=R_inv, t=t_inv)
    points = transformed_point_cloud.points.tolist()
    slam_map.extend(points)

    return R, t

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

def Rtpose(Rt):
    R = Rt[:3, :3]
    t = Rt[:3, 3]
    return R, t

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
