import numpy as np


def wsvd(points, weights):
    center_of_mass = np.sum(points * weights, axis=0) / np.sum(weights)
    centered_points = points - center_of_mass
    temp_matrix = centered_points * np.sqrt(weights[:, np.newaxis])

    U, S, Vt = np.linalg.svd(temp_matrix, full_matrices=False)

    return U, S, Vt
