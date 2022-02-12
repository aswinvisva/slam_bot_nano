import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


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