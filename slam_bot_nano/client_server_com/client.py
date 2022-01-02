import ImageTransferService
import numpy as np
import cv2

if __name__ == "__main__":

    host = '0.0.0.0'
    src = ImageTransferService.ImageTransferService(host)

    # Check Redis is running 
    print(src.ping())

    while True:
        im = src.receiveImage()
        cv2.imshow('Image',im)
        cv2.waitKey(1)
