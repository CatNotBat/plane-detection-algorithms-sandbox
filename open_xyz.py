import open3d as o3d
import numpy as np

if __name__ == "__main__":
    data_points = np.loadtxt("bildstein_station5_xyz_intensity_rgb.txt", usecols=(0, 1, 2))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data_points)
    o3d.visualization.draw_geometries([pcd]) # pylint: disable=E1101
