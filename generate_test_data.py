import numpy as np

import open3d as o3d


def generate_plane_test_data(num_inliers=1000, num_outliers=500, noise_std_dev=0.01):
    """
    Generates a 3D point cloud with a noisy plane and random outliers.

    Args:
        num_inliers (int): Number of points to generate for the plane.
        num_outliers (int): Number of random outlier points.
        noise_std_dev (float): The standard deviation of the Gaussian noise
                               added to the inlier points.

    Returns:
        numpy.ndarray: A (num_inliers + num_outliers, 3) array of points, shuffled.
    """
    
    print(f"Generating {num_inliers} inliers and {num_outliers} outliers...")

    # 1. Define a "ground truth" plane: Ax + By + Cz + D = 0
    #    Let's use a simple plane: 0.5x - 0.2y + 1.0z - 1.0 = 0
    #    So, z = (-0.5x + 0.2y + 1.0) / 1.0 = -0.5x + 0.2y + 1.0
    A, B, C, D = 0.5, -0.2, 1.0, -1.0
    
    # 2. Generate Inlier Points
    # Create random x, y coordinates
    inlier_xy = np.random.rand(num_inliers, 2) * 20 - 10  # Points spread over a 20x20 area
    
    # Calculate the corresponding z coordinate on the plane
    inlier_z = (-A * inlier_xy[:, 0] - B * inlier_xy[:, 1] - D) / C
    
    # Stack them into (x, y, z)
    inliers = np.hstack((inlier_xy, inlier_z.reshape(-1, 1)))
    
    # Add Gaussian noise to all (x, y, z) coordinates
    noise = np.random.normal(scale=noise_std_dev, size=inliers.shape)
    inliers += noise
    
    print(f"Inlier shape: {inliers.shape}")

    # 3. Generate Outlier Points
    # Random points in a larger 3D box
    outliers = np.random.rand(num_outliers, 3) * 30 - 15  # Points spread over a 30x30x30 cube
    print(f"Outlier shape: {outliers.shape}")

    # 4. Combine and Shuffle
    all_points = np.vstack((inliers, outliers))
    np.random.shuffle(all_points)
    
    print(f"Total points generated: {all_points.shape[0]}")
    return all_points

def save_points_to_xyz(points, filename="test_cloud.xyz"):
    """Saves a numpy array of points to a simple .xyz text file."""
    print(f"Saving points to {filename}...")
    np.savetxt(filename, points, fmt='%.6f', delimiter=' ')
    print("Done.")

if __name__ == "__main__":
    # Generate the data
    point_cloud = generate_plane_test_data(num_inliers=2000, num_outliers=5000, noise_std_dev=0.3)
    
    # Save it to a file
    # You can open this file in 3D viewers like MeshLab or CloudCompare
    # to see what it looks like.
    save_points_to_xyz(point_cloud, "my_test_plane.xyz")
    pcd = o3d.io.read_point_cloud("my_test_plane.xyz", format='xyz')

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])

    # In your main RANSAC script, you would load this data:
    # loaded_points = np.loadtxt("my_test_plane.xyz")
