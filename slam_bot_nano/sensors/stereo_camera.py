import cv2
import numpy as np
import threading

FILE_NAMES = {0: "../data/calib_left.npz", 1: "../data/calib_right.npz"}

class StereoCamera:

    def __init__(self, sensor_id):
        # Initialize instance variables
        # OpenCV video capture element
        self.video_capture = None
        # The last captured image from the camera
        self.frame = None
        self.grabbed = False
        # The thread where the video capture runs
        self.read_thread = None
        self.read_lock = threading.Lock()
        self.running = False

        self.sensor_id = sensor_id

        gstreamer_pipeline_string = self.gstreamer_pipeline()
        self.open(gstreamer_pipeline_string)

        with np.load(FILE_NAMES[sensor_id]) as data:
            self.K = data['mtx']

        self.Kinv = np.linalg.inv(self.K)

    #Opening the cameras
    def open(self, gstreamer_pipeline_string):
        gstreamer_pipeline_string = self.gstreamer_pipeline()
        try:
            self.video_capture = cv2.VideoCapture(
                gstreamer_pipeline_string, cv2.CAP_GSTREAMER
            )
            grabbed, frame = self.video_capture.read()
            print("Cameras are opened")

        except RuntimeError:
            self.video_capture = None
            print("Unable to open camera")
            print("Pipeline: " + gstreamer_pipeline_string)
            return
        # Grab the first frame to start the video capturing
        self.grabbed, self.frame = self.video_capture.read()

    #Starting the cameras
    def start(self):
        if self.running:
            print('Video capturing is already running')
            return None
        # create a thread to read the camera image
        if self.video_capture != None:
            self.running = True
            self.read_thread = threading.Thread(target=self.updateCamera, daemon=True)
            self.read_thread.start()
        return self

    def stop(self):
        self.running = False
        self.read_thread.join()

    def updateCamera(self):
        # This is the thread to read images from the camera
        while self.running:
            try:
                grabbed, frame = self.video_capture.read()
                with self.read_lock:
                    self.grabbed = grabbed
                    self.frame = frame
            except RuntimeError:
                print("Could not read image from camera")

    def read(self):
        with self.read_lock:
            if self.grabbed:
                frame = self.frame.copy()
                grabbed = self.grabbed
            else:
                return False, None
        return grabbed, frame

    def release(self):
        if self.video_capture != None:
            self.video_capture.release()
            self.video_capture = None
        # Now kill the thread
        if self.read_thread != None:
            self.read_thread.join()

    # Currently there are setting frame rate on CSI Camera on Nano through gstreamer
    # Here we directly select sensor_mode 3 (1280x720, 59.9999 fps)
    def gstreamer_pipeline(self,
            sensor_mode=3,
            capture_width=1280,
            capture_height=720,
            display_width=640,
            display_height=360,
            framerate=30,
            flip_method=0,
    ):
        return (
                "nvarguscamerasrc sensor-id=%d sensor-mode=%d ! "
                "video/x-raw(memory:NVMM), "
                "width=(int)%d, height=(int)%d, "
                "format=(string)NV12, framerate=(fraction)%d/1 ! "
                "nvvidconv flip-method=%d ! "
                "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
                "videoconvert ! "
                "video/x-raw, format=(string)BGR ! appsink"
                % (
                    self.sensor_id,
                    sensor_mode,
                    capture_width,
                    capture_height,
                    framerate,
                    flip_method,
                    display_width,
                    display_height,
                )
        )

    def project(self, xcs):
        projs = self.K @ xcs.T     
        zs = projs[-1]      
        projs = projs[:2]/ zs   
        return projs.T, zs

    def unproject_points(self, pts):
        return np.dot(self.Kinv, add_ones(pts).T).T[:, 0:2]

    def undistort_points(self, pts):
        if self.is_distorted:
            #uvs_undistorted = cv2.undistortPoints(np.expand_dims(uvs, axis=1), self.K, self.D, None, self.K)   # =>  Error: while undistorting the points error: (-215:Assertion failed) src.isContinuous() 
            uvs_contiguous = np.ascontiguousarray(pts[:, :2]).reshape((pts.shape[0], 1, 2))
            uvs_undistorted = cv2.undistortPoints(uvs_contiguous, self.K, self.D, None, self.K)            
            return uvs_undistorted.ravel().reshape(uvs_undistorted.shape[0], 2)
        else:
            return pts 
