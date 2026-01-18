import numpy as np
import open3d as o3d
import os

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

class HumanDetector:

    def extract_shape_descriptor(self, cluster):
        pts = np.asarray(cluster.points)
        if len(pts) < 5:
            return np.zeros(4)

        centered_pts = pts - np.mean(pts, axis=0)
        bbox = cluster.get_axis_aligned_bounding_box()
        ext = bbox.get_extent()

        # Added safety for division by zero
        max_ext_xy = max(ext[0], ext[1])
        aspect_ratio = ext[2] / (max_ext_xy + 1e-6)
        
        height_threshold = np.min(pts[:, 2]) + 0.1 * ext[2]
        head_density = np.sum(pts[:, 2] > height_threshold) / len(pts)
        variance = np.var(centered_pts, axis=0)

        return np.array([
            aspect_ratio,
            head_density,
            variance[0],
            variance[2]
        ])

    def is_valid_human_geometry(self, cluster):
        bbox = cluster.get_axis_aligned_bounding_box()
        ext = bbox.get_extent()
        pts_count = len(cluster.points)

        return (
            0.1 < ext[2] < 2.1 and
            max(ext[0], ext[1]) < 1.0 and
            pts_count >= 5
        )

    def detect(self, pcd_path):
        detections = []

        if not os.path.exists(pcd_path):
            return detections

        pcd = o3d.io.read_point_cloud(pcd_path)
        # Check 1: Is the cloud empty at start?
        if pcd.is_empty() or len(pcd.points) < 3:
            return detections

        # Filter noise
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=20, std_ratio=1.5
        )

        # CHECK 2: Critical check for RANSAC requirement
        # ransac_n is 3, so we must have at least 3 points
        if len(pcd.points) < 3:
            # If we don't have enough points for a plane, 
            # we skip segmentation and treat the whole PCD as objects
            objects_pcd = pcd
        else:
            _, inliers = pcd.segment_plane(
                distance_threshold=0.15,
                ransac_n=3,
                num_iterations=1000
            )
            objects_pcd = pcd.select_by_index(inliers, invert=True)

        # Proceed with clustering if objects exist
        if objects_pcd.is_empty():
            return detections

        labels = np.array(
            objects_pcd.cluster_dbscan(eps=0.5, min_points=5)
        )

        if len(labels) == 0:
            return detections

        for label in np.unique(labels[labels >= 0]):
            idx = np.where(labels == label)[0]
            cluster = objects_pcd.select_by_index(idx)

            if self.is_valid_human_geometry(cluster):
                detections.append({
                    "centroid": cluster.get_center(),
                    "shape": self.extract_shape_descriptor(cluster)
                })

        return detections