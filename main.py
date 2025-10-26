import time
import numpy as np
from ransac import ransac
from svd import svd


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


if __name__ == "__main__":
    ITERATIONS = 5000
    DISTANCE_THRESHOLD = 0.05
    start_time = time.time()

    data_points = np.loadtxt("my_test_plane.xyz")
    plane = ransac_svd(data_points, ITERATIONS, DISTANCE_THRESHOLD)

    print(f"Best Plane: {plane[0]}x + {plane[1]}y + {plane[2]}z + {plane[3]} time: {time.time() - start_time:.2f} seconds")
