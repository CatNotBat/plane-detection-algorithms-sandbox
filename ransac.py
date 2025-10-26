import numpy as np

from utils import _get_plane_from_points, _get_points_closer_than_distance


def ransac(points: np.ndarray, iterations: int = 2000, distance_threshold: float = 0.05):
    best_plane = None
    best_inliers = np.empty((0, 3))

    for _ in range(iterations):
        inliers = 0
        indices = np.random.choice(points.shape[0], 3, replace=False)
        p1, p2, p3 = points[indices]
        plane = _get_plane_from_points(p1, p2, p3)

        if plane is None:
            continue  # Skip degenerate planes

        inliers = _get_points_closer_than_distance(plane, points, distance_threshold)
        if inliers.shape[0] > best_inliers.shape[0]:
            best_inliers = inliers
            best_plane = plane
    return best_inliers, best_plane
