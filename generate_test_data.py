import numpy as np
import open3d as o3d


def generate_plane_test_data(num_inliers=1000, num_outliers=500, noise_std_dev=0.01):
    """
    Generates a 3D point cloud with a noisy plane and random outliers.
    """

    print(f"Generating {num_inliers} inliers and {num_outliers} outliers...")

    a, b, c, d = 0.5, -0.2, 1.0, -1.0

    inlier_xy = np.random.rand(num_inliers, 2) * 20 - 10 

    inlier_z = (-a * inlier_xy[:, 0] - b * inlier_xy[:, 1] - d) / c
    inliers = np.hstack((inlier_xy, inlier_z.reshape(-1, 1)))
    noise = np.random.normal(scale=noise_std_dev, size=inliers.shape)
    inliers += noise
    print(f"Inlier shape: {inliers.shape}")
    outliers = np.random.rand(num_outliers, 3) * 30 - 15
    print(f"Outlier shape: {outliers.shape}")

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
    point_cloud = generate_plane_test_data(num_inliers=2000, num_outliers=5000, noise_std_dev=0.3)
    save_points_to_xyz(point_cloud, "my_test_plane.xyz")
    pcd = o3d.io.read_point_cloud("my_test_plane.xyz", format='xyz')
    o3d.visualization.draw_geometries([pcd])