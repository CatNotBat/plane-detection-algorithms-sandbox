import numpy as np


def svd(points: np.ndarray):
    """
    Perform Singular Value Decomposition (SVD) on the given set of points.

    Args:
        points (numpy.ndarray): A (N, 3) array of 3D points.

    Returns:
        tuple: U, S, Vt matrices from the SVD.
    """
    # Center the points by subtracting the mean
    centered_points = points - np.mean(points, axis=0)

    # Perform SVD
    U, S, Vt = np.linalg.svd(centered_points, full_matrices=False)

    return U, S, Vt