import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib

def view(path):
    with np.load(path) as data:
        occupancyGridVisited = data['occupancyGridVisited']
        occupancyGridTotal = data['occupancyGridTotal']

    ogMap = occupancyGridVisited / occupancyGridTotal
    ogMap = np.flipud(1 - ogMap)
    plt.imshow(ogMap, cmap='gray')
    plt.show()

if __name__ == "__main__":
    view("slam_bot_nano/data/final_map.npz")