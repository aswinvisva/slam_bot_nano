import cupy as cp_np

from slam_bot_nano.slam.math_utils import *


class PointMap:

    def __init__(self, grid_size=(250, 250), grid_unit_size=0.02):
        self._map = []
        self.map_fig = plt.figure()
        self.map_fig.canvas.set_window_title('Trajectory')
        self.map_subplot = plt.subplot()
        self.map_subplot.grid(True)
        self.frame_shape = (360, 640, 3)
        self.recent_R = cp_np.eye(3)
        self.recent_t = cp_np.zeros((3, 1))

        grid_x_size = int(grid_size[0] / grid_unit_size)
        grid_y_size = int(grid_size[1] / grid_unit_size)

        x = cp_np.linspace(-grid_x_size * grid_unit_size / 2, grid_x_size * grid_unit_size / 2, num=grid_x_size + 1)
        y = cp_np.linspace(-grid_y_size * grid_unit_size / 2, grid_y_size * grid_unit_size / 2, num=grid_y_size + 1)
        self.occupancyGridVisited = cp_np.ones((grid_x_size + 1, grid_y_size + 1))
        self.occupancyGridTotal = 2 * cp_np.ones((grid_x_size + 1, grid_y_size + 1))
        x = cp_np.linspace(-grid_x_size * grid_unit_size / 2, grid_x_size * grid_unit_size / 2, num=grid_x_size + 1)
        y = cp_np.linspace(-grid_y_size * grid_unit_size / 2, grid_y_size * grid_unit_size / 2, num=grid_y_size + 1)
        self.OccupancyGridX, self.OccupancyGridY = cp_np.meshgrid(x, y)

        self.mapXLim = cp_np.array([self.OccupancyGridX[0, 0], self.OccupancyGridX[0, -1]])
        self.mapYLim = cp_np.array([self.OccupancyGridY[0, 0], self.OccupancyGridY[-1, 0]])

        self.lidar_fov = 2 * cp_np.pi
        self.lidarMaxRange = 10
        self.unitGridSize = grid_unit_size

        self.wallThickness = 5 * grid_unit_size

    def spokesGrid(self, numSpokes):
        # 0th ray is at south, then counter-clock wise increases. Theta 0 is at east.
        numHalfElem = int(self.lidarMaxRange / self.unitGridSize)
        bearingIdxGrid = cp_np.zeros((2 * numHalfElem + 1, 2 * numHalfElem + 1))
        x = cp_np.linspace(-self.lidarMaxRange, self.lidarMaxRange, 2 * numHalfElem + 1)
        y = cp_np.linspace(-self.lidarMaxRange, self.lidarMaxRange, 2 * numHalfElem + 1)
        xGrid, yGrid = cp_np.meshgrid(x, y)
        bearingIdxGrid[:, numHalfElem + 1: 2 * numHalfElem + 1] = cp_np.rint((cp_np.pi / 2 + cp_np.arctan(
            yGrid[:, numHalfElem + 1: 2 * numHalfElem + 1] / xGrid[:, numHalfElem + 1: 2 * numHalfElem + 1]))
                / cp_np.pi / 2 * numSpokes - 0.5).astype(int)
        bearingIdxGrid[:, 0: numHalfElem] = cp_np.fliplr(cp_np.flipud(bearingIdxGrid))[:, 0: numHalfElem] + int(numSpokes / 2)
        bearingIdxGrid[numHalfElem + 1: 2 * numHalfElem + 1, numHalfElem] = int(numSpokes / 2)
        rangeIdxGrid = cp_np.sqrt(xGrid**2 + yGrid**2)
        return xGrid, yGrid, bearingIdxGrid, rangeIdxGrid

    def itemizeSpokesGrid(self, xGrid, yGrid, bearingIdxGrid, rangeIdxGrid, numSpokes):
        # Due to discretization, later theta added could lead to up to 1 deg discretization error
        radByX = []
        radByY = []
        radByR = []
        for i in range(numSpokes):
            idx = cp_np.argwhere(bearingIdxGrid == i)
            radByX.append(xGrid[idx[:, 0], idx[:, 1]])
            radByY.append(yGrid[idx[:, 0], idx[:, 1]])
            radByR.append(rangeIdxGrid[idx[:, 0], idx[:, 1]])
        return radByX, radByY, radByR
    
    def convertRealXYToMapIdx(self, x, y):
        #mapXLim is (2,) array for left and right limit, same for mapYLim
        xIdx = (cp_np.rint((x - self.mapXLim[0]) / self.unitGridSize)).astype(int)
        yIdx = (cp_np.rint((y - self.mapYLim[0]) / self.unitGridSize)).astype(int)
        return xIdx, yIdx

    def update(self, point_cloud, idx, R, t):
        self.recent_t = t
        point_cloud = cp_np.vstack(point_cloud).T
        point_cloud = point_cloud[cp_np.argsort(point_cloud[:, 1])]

        R = cp_np.array(R)
        t = cp_np.array(t)

        yaw = rot2euler(R)[2]
        numSamplesPerRev = point_cloud.shape[0]
        angularStep = self.lidar_fov / numSamplesPerRev
        numSpokes = int(cp_np.rint(2 * cp_np.pi / angularStep))
        spokesStartIdx = numSpokes - int(((numSpokes / 2 - numSamplesPerRev) / 2) % numSpokes)

        spokesOffsetIdxByTheta = int(cp_np.rint(yaw / (2 * cp_np.pi) * numSpokes))
        x, y = t[0], t[1]
        xGrid, yGrid, bearingIdxGrid, rangeIdxGrid = self.spokesGrid(numSpokes)
        radByX, radByY, radByR = self.itemizeSpokesGrid(xGrid, yGrid, bearingIdxGrid, rangeIdxGrid, numSpokes)
        emptyXList, emptyYList, occupiedXList, occupiedYList = [], [], [], []
        for i in range(numSpokes):
            spokeIdx = int(cp_np.rint((spokesStartIdx + spokesOffsetIdxByTheta + i) % numSpokes))
            xAtSpokeDir = radByX[spokeIdx]
            yAtSpokeDir = radByY[spokeIdx]
            rAtSpokeDir = radByR[spokeIdx]
            if point_cloud[i][0] < self.lidarMaxRange:
                emptyIdx = cp_np.argwhere(rAtSpokeDir < point_cloud[i][0] - self.wallThickness / 2)
            else:
                emptyIdx = []
            occupiedIdx = cp_np.argwhere(
                (rAtSpokeDir > point_cloud[i][0] - self.wallThickness / 2) & (rAtSpokeDir < point_cloud[i][0] + self.wallThickness / 2))
            xEmptyIdx, yEmptyIdx = self.convertRealXYToMapIdx(x + xAtSpokeDir[emptyIdx], y + yAtSpokeDir[emptyIdx])
            xOccupiedIdx, yOccupiedIdx = self.convertRealXYToMapIdx(x + xAtSpokeDir[occupiedIdx], y + yAtSpokeDir[occupiedIdx])

            if len(emptyIdx) != 0:
                self.occupancyGridTotal[yEmptyIdx, xEmptyIdx] += 1
            if len(occupiedIdx) != 0:
                self.occupancyGridVisited[yOccupiedIdx, xOccupiedIdx] += 2
                self.occupancyGridTotal[yOccupiedIdx, xOccupiedIdx] += 2

    def array(self):
        return cp_np.array([p._x for p in self._map if p.n_observations >= 0])

    def save(self, path, name):
        np.savez(os.path.join(path, f'{name}.npz'), occupancyGridVisited=self.occupancyGridVisited, occupancyGridTotal=self.occupancyGridTotal) 

    def plot(self):
        x, y = self.recent_t[0][0], self.recent_t[1][0]
        xRange = cp_np.array([x-10, x+10])
        yRange = cp_np.array([y-10, y+10])
        ogMap = self.occupancyGridVisited / self.occupancyGridTotal
        xIdx, yIdx = self.convertRealXYToMapIdx(xRange, yRange)

        self.map_subplot.set_xlim(xRange[0], xRange[1])
        self.map_subplot.set_ylim(yRange[0], yRange[1])

        self.map_fig.canvas.draw()
        self.map_subplot.clear()

        ogMap = ogMap[yIdx[0]: yIdx[1], xIdx[0]: xIdx[1]]
        ogMap = cp_np.flipud(1 - ogMap)
        xRange = xRange.get()
        yRange = yRange.get()
        ogMap = ogMap.get()
        self.map_subplot.imshow(ogMap, cmap='gray', extent=[xRange[0], xRange[1], yRange[0], yRange[1]])
        # ogMap = ogMap >= 0.5
        # self.map_subplot.matshow(ogMap, cmap='gray', extent=[xRange[0], xRange[1], yRange[0], yRange[1]])

        trajectory_frame = np.frombuffer(self.map_fig.canvas.tostring_rgb(), dtype='uint8')
        trajectory_frame = trajectory_frame.reshape(self.map_fig.canvas.get_width_height()[::-1] + (3,))
        trajectory_frame = cv2.resize(trajectory_frame, (trajectory_frame.shape[1], self.frame_shape[0]))

        return trajectory_frame
