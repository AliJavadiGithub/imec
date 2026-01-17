import os
import numpy as np
import open3d as o3d
from typing import List, Dict

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


class HumanDetector:
    """
    Stateless point-cloud-based human detector.
    """

    def detect(self, pcd_path: str) -> List[Dict]:
        if not os.path.exists(pcd_path):
            return []

        pcd = o3d.io.read_point_cloud(pcd_path)
        if pcd.is_empty():
            return []

        pcd = self._remove_noise(pcd)
        pcd = self._remove_ground(pcd)

        labels = np.array(pcd.cluster_dbscan(eps=0.5, min_points=5))
        detections = []

        for lbl in np.unique(labels[labels >= 0]):
            cluster = pcd.select_by_index(np.where(labels == lbl)[0])
            if self._is_human(cluster):
                detections.append({
                    "centroid": cluster.get_center(),
                    "shape": self._shape_descriptor(cluster)
                })

        return detections

    @staticmethod
    def _remove_noise(pcd):
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=20,
            std_ratio=1.5
        )
        return pcd

    @staticmethod
    def _remove_ground(pcd):
        if pcd.is_empty():
            return pcd

        _, inliers = pcd.segment_plane(
            distance_threshold=0.15,
            ransac_n=3,
            num_iterations=1000
        )
        return pcd.select_by_index(inliers, invert=True)

    @staticmethod
    def _is_human(cluster) -> bool:
        bbox = cluster.get_axis_aligned_bounding_box()
        ext = bbox.get_extent()
        return (
            0.7 < ext[2] < 2.1 and
            max(ext[0], ext[1]) < 1.0 and
            len(cluster.points) >= 5
        )

    @staticmethod
    def _shape_descriptor(cluster) -> np.ndarray:
        pts = np.asarray(cluster.points)
        if len(pts) < 5:
            return np.zeros(4)

        bbox = cluster.get_axis_aligned_bounding_box()
        ext = bbox.get_extent()

        centered = pts - pts.mean(axis=0)
        var = np.var(centered, axis=0)

        h_thresh = pts[:, 2].min() + 0.7 * ext[2]
        head_density = np.mean(pts[:, 2] > h_thresh)

        aspect_ratio = ext[2] / (max(ext[0], ext[1]) + 1e-6)

        return np.array([
            aspect_ratio,
            head_density,
            var[0],
            var[2]
        ])
