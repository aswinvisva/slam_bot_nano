'''
 Based on the following tutorial:
   http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_calib3d/py_calibration/py_calibration.html
'''

import numpy as np
import cv2
import glob

from stereo_camera import StereoCamera


def calibrate(sensor, sensor_name, n_frames=20):
    # Define the chess board rows and columns
    rows = 6
    cols = 9

    # Set the termination criteria for the corner sub-pixel algorithm
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)

    # Prepare the object points: (0,0,0), (1,0,0), (2,0,0), ..., (6,5,0). They are the same for all images
    objectPoints = np.zeros((rows * cols, 3), np.float32)
    objectPoints[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)

    # Create the arrays to store the object points and the image points
    objectPointsArray = []
    imgPointsArray = []

    # Loop over the image files
    for i in range(n_frames):
        # Load the image and convert it to gray scale
        grabbed = False
        
        while not grabbed:
            grabbed, img = sensor.read()

        img = cv2.flip(img, -1)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (rows, cols), None)

        # Make sure the chess board pattern was found in the image
        if ret:
            # Refine the corner position
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Add the object points and the image points to the arrays
            objectPointsArray.append(objectPoints)
            imgPointsArray.append(corners)

            # Draw the corners on the image
            cv2.drawChessboardCorners(img, (rows, cols), corners, ret)
        
        # Display the image
        cv2.imshow('chess board', img)
        cv2.waitKey(500)

    # Calibrate the camera and save the results
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPointsArray, imgPointsArray, gray.shape[::-1], None, None)
    np.savez('calib_%s.npz' % sensor_name, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

    # Print the camera calibration error
    error = 0

    for i in range(len(objectPointsArray)):
        imgPoints, _ = cv2.projectPoints(objectPointsArray[i], rvecs[i], tvecs[i], mtx, dist)
        error += cv2.norm(imgPointsArray[i], imgPoints, cv2.NORM_L2) / len(imgPoints)

    print("Total error: ", error / len(objectPointsArray))

if __name__ == "__main__":
    left_camera = StereoCamera(0).start()
    right_camera = StereoCamera(1).start()

    calibrate(left_camera, "left")
    calibrate(right_camera, "right")