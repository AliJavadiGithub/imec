# tracking.py
"""
Robust Human Tracking in Point Cloud Space
----------------------------------------

Key guarantees:
- Input (.pcd files) and output (tracking_results.json)
- Clear separation of detection, tracking, and velocity estimation
- Conservative defaults tuned for mapAll / mapHumanOnly datasets

Dependencies:
- open3d
- numpy
- scipy
- filterpy

"""

import os
import re
import json
import logging
from typing import List, Dict, Tuple

import numpy as np
import open3d as o3d
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

# -----------------------------------------------------------------------------
# Global configuration
# -----------------------------------------------------------------------------
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def extract_timestamp_ms(filename: str) -> int:
    """Extract timestamp in milliseconds from filename."""
    match = re.search(r"(\d+)ms", filename)
    if not match:
        raise ValueError(f"Invalid filename format (no timestamp): {filename}")
    return int(match.group(1))


def safe_norm(x: np.ndarray) -> float:
    """Numerically safe norm."""
    return float(np.linalg.norm(x)) if x.size else 0.0

# -----------------------------------------------------------------------------
# Detection module
# -----------------------------------------------------------------------------

class HumanDetector:
    """
    Detects human-sized clusters from a point cloud using:
    - Noise removal
    - Ground plane subtraction
    - DBSCAN clustering
    - Simple geometric validation
    """

    def __init__(self):
        self.dbscan_eps = 0.5
        self.dbscan_min_points = 5

    @staticmethod
    def is_valid_human_geometry(cluster: o3d.geometry.PointCloud) -> bool:
        """Validate cluster size against human-like dimensions."""
        if cluster.is_empty():
            return False

        bbox = cluster.get_axis_aligned_bounding_box()
        ext = bbox.get_extent()  # [x, y, z]
        num_pts = len(cluster.points)

        # Human heuristics (meters)
        height_ok = 0.7 < ext[2] < 2.2
        width_ok = max(ext[0], ext[1]) < 1.0
        density_ok = num_pts >= 5

        return height_ok and width_ok and density_ok

    @staticmethod
    def extract_shape_descriptor(cluster: o3d.geometry.PointCloud) -> np.ndarray:
        """Lightweight shape descriptor for re-identification."""
        pts = np.asarray(cluster.points)
        if pts.shape[0] < 5:
            return np.zeros(4, dtype=np.float32)

        bbox = cluster.get_axis_aligned_bounding_box()
        ext = bbox.get_extent()

        aspect_ratio = ext[2] / (max(ext[0], ext[1]) + 1e-6)
        centered = pts - pts.mean(axis=0)
        var = np.var(centered, axis=0)

        head_thresh = np.min(pts[:, 2]) + 0.7 * ext[2]
        head_density = float(np.mean(pts[:, 2] > head_thresh))

        return np.array([
            aspect_ratio,
            head_density,
            var[0],
            var[2]
        ], dtype=np.float32)

    def detect(self, pcd_path: str) -> List[Dict]:
        """Run full detection pipeline on a .pcd file."""
        if not os.path.exists(pcd_path):
            return []

        pcd = o3d.io.read_point_cloud(pcd_path)
        if pcd.is_empty():
            return []

        # Noise filtering
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
        if pcd.is_empty():
            return []

        # Ground plane removal
        try:
            _, inliers = pcd.segment_plane(
                distance_threshold=0.15,
                ransac_n=3,
                num_iterations=1000
            )
            pcd = pcd.select_by_index(inliers, invert=True)
        except RuntimeError:
            # Plane fitting may fail on sparse frames
            pass

        if pcd.is_empty():
            return []

        # Clustering
        labels = np.array(pcd.cluster_dbscan(
            eps=self.dbscan_eps,
            min_points=self.dbscan_min_points
        ))

        detections = []
        for label in np.unique(labels):
            if label < 0:
                continue
            idx = np.where(labels == label)[0]
            cluster = pcd.select_by_index(idx)

            if not self.is_valid_human_geometry(cluster):
                continue

            detections.append({
                "centroid": cluster.get_center(),
                "shape": self.extract_shape_descriptor(cluster)
            })

        return detections

# -----------------------------------------------------------------------------
# Tracking primitives
# -----------------------------------------------------------------------------

class Track:
    """Single tracked human with Kalman state."""

    def __init__(self, track_id: int, centroid: np.ndarray, shape: np.ndarray):
        self.id = track_id
        self.kf = self._init_kalman(centroid)
        self.shape_history = [shape]
        self.age = 1
        self.hits = 1
        self.skipped = 0
        self.last_position = centroid.copy()
        self.is_static = False

    @staticmethod
    def _init_kalman(pos: np.ndarray) -> KalmanFilter:
        kf = KalmanFilter(dim_x=6, dim_z=3)

        # State: [x, y, z, vx, vy, vz]
        kf.F = np.eye(6)
        kf.H = np.block([[np.eye(3), np.zeros((3, 3))]])

        kf.x[:3] = pos.reshape(3, 1)
        kf.P *= 5.0
        kf.R *= 10.0
        kf.Q = np.eye(6) * 0.01

        return kf

    def predict(self, dt: float):
        for i in range(3):
            self.kf.F[i, i + 3] = dt
        self.kf.predict()
        self.age += 1

    def update(self, centroid: np.ndarray, shape: np.ndarray):
        self.kf.update(centroid)
        self.shape_history.append(shape)
        self.last_position = centroid.copy()
        self.hits += 1
        self.skipped = 0

    def velocity(self) -> np.ndarray:
        return self.kf.x[3:].flatten()

    def speed(self) -> float:
        return safe_norm(self.velocity())

    def avg_shape(self) -> np.ndarray:
        return np.mean(self.shape_history[-15:], axis=0)

# -----------------------------------------------------------------------------
# Multi-object tracker
# -----------------------------------------------------------------------------

class HumanTrackerMOT:
    """Multi-object tracker managing lifecycle, association and output."""

    def __init__(self):
        self.detector = HumanDetector()
        self.tracks: List[Track] = []
        self.lost: List[Track] = []
        self.next_id = 1
        self.history: List[Dict] = []

        # Hyperparameters
        self.min_hits = 3
        self.max_skip_dynamic = 20
        self.max_skip_static = 50
        self.gating_dist = 1.2
        self.dist_weight = 0.8
        self.shape_weight = 0.2

    # ------------------------------------------------------------------
    # Core update step
    # ------------------------------------------------------------------

    def update(self, frame_id: int, timestamp_ms: int, pcd_path: str):
        detections = self.detector.detect(pcd_path)

        # Time delta
        if self.history:
            prev_ts = self.history[-1]["timestamp_ms"]
        else:
            prev_ts = timestamp_ms - 33
        dt = max((timestamp_ms - prev_ts) / 1000.0, 1e-3)

        # Predict
        for t in self.tracks:
            t.predict(dt)

        # Association
        assigned_tracks, assigned_dets = set(), set()
        if self.tracks and detections:
            cost = np.full((len(self.tracks), len(detections)), 1e6)

            for i, trk in enumerate(self.tracks):
                for j, det in enumerate(detections):
                    d = safe_norm(trk.kf.x[:3].flatten() - det["centroid"])
                    if d > self.gating_dist:
                        continue
                    s = safe_norm(trk.avg_shape() - det["shape"])
                    cost[i, j] = self.dist_weight * d + self.shape_weight * s

            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if cost[r, c] >= self.gating_dist:
                    continue
                self.tracks[r].update(detections[c]["centroid"], detections[c]["shape"])
                assigned_tracks.add(r)
                assigned_dets.add(c)

        # Track maintenance
        active_tracks = []
        for i, trk in enumerate(self.tracks):
            if i not in assigned_tracks:
                trk.skipped += 1

            trk.is_static = trk.speed() < 0.15
            limit = self.max_skip_static if trk.is_static else self.max_skip_dynamic

            if trk.skipped <= limit:
                active_tracks.append(trk)
            else:
                self.lost.append(trk)

        self.tracks = active_tracks

        # New tracks / re-identification
        for j, det in enumerate(detections):
            if j in assigned_dets:
                continue

            reid = None
            for lt in self.lost:
                if safe_norm(lt.last_position - det["centroid"]) < 1.5:
                    reid = lt
                    break

            if reid:
                reid.kf.x[:3] = det["centroid"].reshape(3, 1)
                reid.shape_history.append(det["shape"])
                reid.skipped = 0
                self.tracks.append(reid)
                self.lost.remove(reid)
            else:
                self.tracks.append(Track(self.next_id, det["centroid"], det["shape"]))
                self.next_id += 1

        # Logging
        frame_out = []
        for trk in self.tracks:
            if trk.skipped == 0 and trk.hits >= self.min_hits:
                pos = trk.kf.x[:3].flatten()
                frame_out.append({
                    "id": int(trk.id),
                    "position": [round(float(p), 3) for p in pos],
                    "speed": round(trk.speed(), 3),
                    "status": "STATIC" if trk.is_static else "MOVING"
                })

        self.history.append({
            "frame_id": frame_id,
            "timestamp_ms": timestamp_ms,
            "detections": frame_out
        })

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    def finalize_results(self, output_path: str = "tracking_results.json"):
        id_counts = {}
        for f in self.history:
            for d in f["detections"]:
                id_counts[d["id"]] = id_counts.get(d["id"], 0) + 1

        valid_ids = {i for i, c in id_counts.items() if c > 10}
        cleaned = []

        for f in self.history:
            dets = [d for d in f["detections"] if d["id"] in valid_ids]
            if dets:
                f["detections"] = dets
                cleaned.append(f)

        with open(output_path, "w") as f:
            json.dump(cleaned, f, indent=4)

        logging.info(f"Tracking finished. Results written to {output_path}")

# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

def main():
    data_dir = "mapAll"
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    files = sorted(
        [f for f in os.listdir(data_dir) if f.endswith(".pcd")],
        key=extract_timestamp_ms
    )

    tracker = HumanTrackerMOT()

    for i, fname in enumerate(files):
        ts = extract_timestamp_ms(fname)
        tracker.update(i, ts, os.path.join(data_dir, fname))
        if i % 50 == 0:
            logging.info(f"Processing frame {i}/{len(files)}")

    tracker.finalize_results()


if __name__ == "__main__":
    main()
