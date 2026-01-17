import numpy as np
import open3d as o3d

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


class HumanDetector:

    def extract_shape_descriptor(self, cluster):
        pts = np.asarray(cluster.points)
        if len(pts) < 5:
            return np.zeros(4)

        centered_pts = pts - np.mean(pts, axis=0)
        bbox = cluster.get_axis_aligned_bounding_box()
        ext = bbox.get_extent()

        aspect_ratio = ext[2] / (max(ext[0], ext[1]) + 1e-6)
        height_threshold = np.min(pts[:, 2]) + 0.7 * ext[2]
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
            0.7 < ext[2] < 2.1 and
            max(ext[0], ext[1]) < 1.0 and
            pts_count >= 5
        )

    def detect(self, pcd_path):
        detections = []

        pcd = o3d.io.read_point_cloud(pcd_path)
        if pcd.is_empty():
            return detections

        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=20, std_ratio=1.5
        )

        _, inliers = pcd.segment_plane(
            distance_threshold=0.15,
            ransac_n=3,
            num_iterations=1000
        )

        objects_pcd = pcd.select_by_index(inliers, invert=True)

        labels = np.array(
            objects_pcd.cluster_dbscan(eps=0.5, min_points=5)
        )

        for label in np.unique(labels[labels >= 0]):
            idx = np.where(labels == label)[0]
            cluster = objects_pcd.select_by_index(idx)

            if self.is_valid_human_geometry(cluster):
                detections.append({
                    "centroid": cluster.get_center(),
                    "shape": self.extract_shape_descriptor(cluster)
                })

        return detections
