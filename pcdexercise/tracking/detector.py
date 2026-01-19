import numpy as np
import open3d as o3d

class HumanDetector:
    """Detects human-like clusters in point clouds."""

    def __init__(self, eps=0.5, min_points=5):
        self.eps = eps
        self.min_points = min_points

    def extract_shape_descriptor(self, cluster):
        pts = np.asarray(cluster.points)
        if len(pts) < 5:
            return np.zeros(4)

        bbox = cluster.get_axis_aligned_bounding_box()
        ext = np.maximum(bbox.get_extent(), 1e-6)

        centered = pts - pts.mean(axis=0)
        aspect_ratio = ext[2] / max(ext[0], ext[1])
        head_density = float(np.mean(pts[:, 2] > np.percentile(pts[:, 2], 80)))
        var = np.var(centered, axis=0)

        return np.array([aspect_ratio, head_density, var[0], var[2]])

    def is_valid_human(self, cluster):
        pts = np.asarray(cluster.points)
        if len(pts) < 5:
            return False

        bbox = cluster.get_axis_aligned_bounding_box()
        ext = np.maximum(bbox.get_extent(), 1e-6)

        height = ext[2]
        width = max(ext[0], ext[1])
        density = len(pts) / np.prod(ext)

        return (
            0.4 <= height <= 2.2 and
            width <= 1.0 and
            density > 5.0
        )

    def detect(self, pcd):
        if pcd.is_empty():
            return []

        try:
            pcd, _ = pcd.remove_statistical_outlier(20, 1.5)
            _, inliers = pcd.segment_plane(0.15, 3, 1000)
            pcd = pcd.select_by_index(inliers, invert=True)
        except RuntimeError:
            pass

        labels = np.array(pcd.cluster_dbscan(self.eps, self.min_points))
        detections = []

        for lbl in np.unique(labels[labels >= 0]):
            cluster = pcd.select_by_index(np.where(labels == lbl)[0])
            if self.is_valid_human(cluster):
                detections.append({
                    "centroid": cluster.get_center(),
                    "shape": self.extract_shape_descriptor(cluster)
                })

        return detections
