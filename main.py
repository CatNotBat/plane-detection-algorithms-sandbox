import time
import numpy as np
from ransac import ransac
from svd import svd
from wsvd import wsvd
from utils import _get_distance_from_plane
import open3d as o3d

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
        current_plane = new_plane

    return current_plane


if __name__ == "__main__":
    ITERATIONS = 30
    DISTANCE_THRESHOLD = 0.5
    MIN_POINTS_THRESHOLD = 50000
    start_time = time.time()
    FULL_DATA = np.loadtxt("bildstein_station5_xyz_intensity_rgb.txt", usecols=(0, 1, 2))
    segmented_planes = []
    current_active_indices = np.arange(FULL_DATA.shape[0], dtype=int)
    loop_count = 0
    while len(current_active_indices) > MIN_POINTS_THRESHOLD:
        loop_count += 1
        print(f"\n--- Segmentation Loop {loop_count}: {len(current_active_indices)} points remaining ---")

        points_subset = FULL_DATA[current_active_indices]
        
        best_inliers, relative_inlier_indices, best_plane = ransac(
            points_subset, ITERATIONS, DISTANCE_THRESHOLD
        )
        
        inlier_count = len(best_inliers)
        
        # 2. Check stopping condition
        if inlier_count < MIN_POINTS_THRESHOLD:
            print(f"Stopping. Found only {inlier_count} inliers (< {MIN_POINTS_THRESHOLD}).")
            break
        absolute_inlier_indices = current_active_indices[relative_inlier_indices]
        
        # 4. Save the found plane points and coefficients
        segmented_planes.append({
            'points': best_inliers,
            'indices': absolute_inlier_indices,
            'coefficients': best_plane,
        })
        
        is_inlier = np.zeros(len(current_active_indices), dtype=bool)
        is_inlier[relative_inlier_indices] = True
        
        current_active_indices = current_active_indices[~is_inlier]
        
        print(f"Plane {loop_count} found: {inlier_count} points. Remaining points: {len(current_active_indices)}")
        
        if loop_count == 1:
            DISTANCE_THRESHOLD = 0.15

    # --- VISUALIZATION ---
    print(f"\nSegmentation complete. Took {time.time() - start_time:.2f} seconds.")
    print(f"Total planes segmented: {len(segmented_planes)}")
    
    if segmented_planes:
        final_pcd = o3d.geometry.PointCloud()
        
        # Define a color map for visualization
        color_map = [
            [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0], 
            [0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [0.5, 0.5, 0.5], [0.8, 0.4, 0.0],
            [0.2, 0.8, 0.2]
        ]
        
        all_points = []
        all_colors = []

        for i, plane_data in enumerate(segmented_planes):
            points = plane_data['points']
            color = color_map[i % len(color_map)]
            colors = np.tile(color, (len(points), 1))
            
            all_points.append(points)
            all_colors.append(colors)

        # Add remaining unassigned points (noise) as black
        if len(current_active_indices) > 0:
            noise_points = FULL_DATA[current_active_indices]
            noise_colors = np.tile([0.1, 0.1, 0.1], (len(noise_points), 1))
            
            all_points.append(noise_points)
            all_colors.append(noise_colors)
            print(f"Showing {len(noise_points)} points as unsegmented noise.")

        if all_points:
            final_points = np.concatenate(all_points)
            final_colors = np.concatenate(all_colors)

            final_pcd.points = o3d.utility.Vector3dVector(final_points)
            final_pcd.colors = o3d.utility.Vector3dVector(final_colors)

            o3d.visualization.draw_geometries([final_pcd], window_name="Sequential RANSAC Segmentation")
        else:
            print("No significant planes were found.")