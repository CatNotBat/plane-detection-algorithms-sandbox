import time
import numpy as np
from ransac import ransac
from svd import svd

if __name__ == "__main__":
    ITERATIONS = 5000
    DISTANCE_THRESHOLD = 0.05
    start_time = time.time()

    points = np.loadtxt("my_test_plane.xyz")
    filtered_points = ransac(points, ITERATIONS, DISTANCE_THRESHOLD)[0]
    normal = svd(filtered_points)[2][-1, :]

    if normal[2] < 0:
        normal = -normal

    A = normal[0]
    B = normal[1]
    C = normal[2]
    centroid = np.mean(filtered_points, axis=0)
    D = -centroid @ normal

    print(f"Best Plane: {A}x + {B}y + {C}z + {D} time: {time.time() - start_time:.2f} seconds")
