import numpy as np

from utils import _get_plane_from_points, find_inliers_indices

def ransac(points: np.ndarray, iterations: int = 2000, distance_threshold: float = 0.05):
    # NOTE: You MUST initialize best_inliers_indices
    best_plane = None
    best_inliers_indices = np.empty(0, dtype=int) 

    # OPTIMIZATION: Use a smaller sample size for initial hypothesis testing
    # Use 100k points for RANSAC to run quickly
    SUB_SAMPLE_SIZE = min(100000, points.shape[0])
    
    for _ in range(iterations):
        # 1. Sample 3 points from the current ACTIVE set
        if SUB_SAMPLE_SIZE < 3:
            return None, np.empty(0, dtype=int), None # Not enough points left
            
        sample_indices = np.random.choice(points.shape[0], 3, replace=False)
        p1, p2, p3 = points[sample_indices]
        plane = _get_plane_from_points(p1, p2, p3)

        if plane is None:
            continue
        current_inliers_indices = find_inliers_indices(plane, points, distance_threshold)

        if current_inliers_indices.shape[0] > best_inliers_indices.shape[0]:
            best_inliers_indices = current_inliers_indices
            best_plane = plane
    # Return the points and indices corresponding to the best plane
    best_inliers = points[best_inliers_indices]
    return best_inliers, best_inliers_indices, best_plane