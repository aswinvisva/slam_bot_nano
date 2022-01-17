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

from slam_bot_nano.slam.math_utils import *

STAGE_FIRST_FRAME = 0
STAGE_DEFAULT_FRAME = 1


class LidarOdometry:
    def __init__(self):
        self.cur_R = np.eye(3,3) # current rotation
        self.cur_t = np.zeros((3,1)) # current translation
        self.last_frame = None
        self.new_frame = None
        self.frame_stage = 0

    def processFirstFrame(self):
        self.frame_stage = STAGE_DEFAULT_FRAME

    def processFrame(self):
        R, t, row_indices, col_indices = ICP(self.last_frame, self.new_frame, "point2point")
        t = t.reshape((3,1))

        self.cur_t = self.cur_t + self.cur_R.dot(t)
        self.cur_R = R.dot(self.cur_R)

    def update(self, point_cloud):
        point_cloud = pol2cart(*point_cloud)
        point_cloud = np.vstack(point_cloud).T
        point_cloud = np.append(point_cloud, np.zeros((point_cloud.shape[0], 1)), axis=1)
        point_cloud = Point_cloud().init_from_points(point_cloud)

        self.new_frame = point_cloud
        if(self.frame_stage == STAGE_DEFAULT_FRAME):
            self.processFrame()
        elif(self.frame_stage == STAGE_FIRST_FRAME):
            self.processFirstFrame()
        self.last_frame = self.new_frame

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

def best_transform(data, ref, method = "point2point", indexes_d = None, indexes_r = None, verbose = False):
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

def ICP(data,ref,method, exclusion_radius = 0.5, sampling_limit = None, verbose = False):
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
