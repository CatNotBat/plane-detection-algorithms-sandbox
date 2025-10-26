import numpy as np

def _get_plane_from_points(p1: float, p2: float, p3: float):
    """Calculate plane coefficients (A, B, C, D) from three points."""
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    a, b, c = normal
    magnitude = np.sqrt(a**2 + b**2 + c**2)
    if magnitude < 1e-6:  # Epsilon for zero
        return None  # Signal a degenerate plane
    a /= magnitude
    b /= magnitude
    c /= magnitude
    d = -np.dot(normal, p1) / magnitude
    return a, b, c, d


def _get_points_closer_than_distance(plane: tuple, points: np.ndarray, distance_threshold: float):
    """Get points closer than a certain distance from the plane."""
    distances = _get_distance_from_plane(plane, points)
    close_points = points[distances < distance_threshold]
    return close_points


def _get_distance_from_plane(plane: tuple, points: np.ndarray):
    """Calculate the perpendicular distance from points to the plane."""
    a, b, c, d = plane
    numerator = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d)
    denominator = np.sqrt(a**2 + b**2 + c**2)
    distances = numerator / denominator
    return distances
