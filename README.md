# plane-detection-algorithms-sandbox

Comparing different algorithms for fitting planes to 3D point cloud data. Built to evaluate how each method handles noise and outliers on real LiDAR scans.

## Algorithms

- **RANSAC** — random sample consensus, iterative outlier rejection
- **SVD** — singular value decomposition on the full point set
- **Weighted SVD** — SVD with per-point weights (Tukey bisquare)
- **IRLS** — iteratively reweighted least squares using robust weights
- **RANSAC + SVD** — RANSAC for outlier removal, then SVD on inliers

## Usage

Expects a point cloud file (`.txt`, XYZ columns). The main script runs sequential RANSAC segmentation and visualizes each detected plane in a different color using Open3D.

```bash
pip install numpy open3d
python main.py
```

## What it looks like

The output is an Open3D window showing the segmented planes color-coded, with unsegmented noise in dark gray.
