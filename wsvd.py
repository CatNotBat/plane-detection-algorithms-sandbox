import numpy as np

def wsvd(points, weights):
    center_of_mass = np.sum(points * weights[:, np.newaxis], axis=0) / np.sum(weights)
    centered_points = points - center_of_mass
    temp_matrix = centered_points * np.sqrt(weights[:, np.newaxis])

    u, s, vt = np.linalg.svd(temp_matrix, full_matrices=False)

    return u, s, vt, center_of_mass
