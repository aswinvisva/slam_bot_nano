import numpy as np

class PointMap:

    def __init__(self):
        self._map = []

    def add_points(self, points):
        for p in points:
            point_P = Point(p)
            self.add_point(point_P)

    def add_point(self, p, dist_thresh=0.05):
        for point in self._map:
            if point.get_distance_to_point(p) < dist_thresh:
                point.add_observation()
                return False

        self._map.append(p)
        return True

class Point:

    def __init__(self, x):
        self.n_observations
        self._x = x

    def add_observation(self):
        self.n_observations += 1

    def remove_observation(self):
        self.n_observations -= 1

    def val(self):
        return self._x

    def get_distance_to_point(self, other):
        distance = np.linalg.norm(self.val() - other.val())
        return distance

