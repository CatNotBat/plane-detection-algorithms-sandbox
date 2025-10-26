import time
import numpy as np
from ransac import ransac
from svd import svd
from wsvd import wsvd
from utils import _get_distance_from_plane


def ransac_svd(points: np.ndarray, iterations: int = 2000, distance_threshold: float = 0.05):
    filtered_points = ransac(points, iterations, distance_threshold)[0]
    normal = svd(filtered_points)[2][-1, :]

    if normal[2] < 0:
        normal = -normal

    a = normal[0]
    b = normal[1]
    c = normal[2]
    centroid = np.mean(filtered_points, axis=0)
    d = -centroid @ normal

    return a, b, c, d


def wls(points: np.ndarray, max_iterations: int = 100, convergence_threshold: float = 1e-6):
    """Weighted Least Squares plane fitting using Weighted SVD."""
    start_normal = svd(points)[2][-1, :]
    if start_normal[2] < 0:
        start_normal = -start_normal
    a = start_normal[0]
    b = start_normal[1]
    c = start_normal[2]
    centroid = np.mean(points, axis=0)
    d = -centroid @ start_normal
    current_plane = (a, b, c, d)

    for _ in range(max_iterations):
        distances = _get_distance_from_plane(current_plane, points)
        distances_from_median = np.abs(distances - np.median(distances))
        mad = np.median(distances_from_median)
        c = mad * 4.685
        weights = (1 - (distances / c) ** 2) ** 2
        weights[np.abs(distances / c) > 1] = 0.0
        _, _, vt, centroid = wsvd(points, weights)
        normal = vt[-1, :]
        if normal[2] < 0:
            normal = -normal
        a = normal[0]
        b = normal[1]
        c = normal[2]
        d = -centroid @ normal
        new_plane = (a, b, c, d)
        if np.linalg.norm(np.array(current_plane) - np.array(new_plane)) < convergence_threshold:
            break
        current_plane = new_plane

    return current_plane


if __name__ == "__main__":
    ITERATIONS = 5000
    DISTANCE_THRESHOLD = 0.05
    start_time = time.time()

    data_points = np.loadtxt("my_test_plane.xyz")
    plane = ransac_svd(data_points)

    print(f"Best Plane: {plane[0]}x + {plane[1]}y + {plane[2]}z + {plane[3]} time: {time.time() - start_time:.2f} seconds")
