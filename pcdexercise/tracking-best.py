"""
tracking.py
-----------
Robust multi-object human tracking in 3D point clouds.

- Detection: DBSCAN clustering + geometry validation
- Tracking: IMM Kalman Filter (CV + RW)
- Association: Hungarian + gating
- Re-ID: spatial proximity
- Output: tracking_results.json (unchanged schema)

Author: Refactored for clean architecture & safety
"""

import os
import re
import json
import numpy as np
import open3d as o3d
from typing import List, Dict
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter, IMMEstimator

# Silence Open3D spam
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

# =========================
# Utility
# =========================

def extract_timestamp_ms(filename: str) -> int:
    match = re.search(r"(\d+)ms", filename)
    if not match:
        raise ValueError(f"Invalid filename timestamp: {filename}")
    return int(match.group(1))


# =========================
# Detection
# =========================

class HumanDetector:
    """Detects human-like clusters from a point cloud."""

    def __init__(self):
        self.dbscan_eps = 0.5
        self.dbscan_min_points = 5

    def extract_shape_descriptor(self, cluster: o3d.geometry.PointCloud) -> np.ndarray:
        pts = np.asarray(cluster.points)
        if pts.shape[0] < 5:
            return np.zeros(4)

        centered = pts - pts.mean(axis=0)
        bbox = cluster.get_axis_aligned_bounding_box()
        ext = np.maximum(bbox.get_extent(), 1e-6)

        aspect_ratio = ext[2] / max(ext[0], ext[1])
        head_density = float(np.mean(pts[:, 2] > np.percentile(pts[:, 2], 80)))
        var = np.var(centered, axis=0)

        return np.array([aspect_ratio, head_density, var[0], var[2]])

    def is_valid_human_geometry(self, cluster: o3d.geometry.PointCloud) -> bool:
        pts = np.asarray(cluster.points)
        if pts.shape[0] < 5:
            return False

        bbox = cluster.get_axis_aligned_bounding_box()
        ext = np.maximum(bbox.get_extent(), 1e-6)

        height = ext[2]
        width = max(ext[0], ext[1])
        density = pts.shape[0] / np.prod(ext)

        return (
            0.4 <= height <= 2.2 and
            width <= 1.0 and
            density > 5.0
        )

    def detect(self, pcd: o3d.geometry.PointCloud) -> List[Dict]:
        """Returns list of detections: {centroid, shape}"""
        if pcd.is_empty():
            return []

        try:
            pcd, _ = pcd.remove_statistical_outlier(20, 1.5)
        except RuntimeError:
            return []

        if len(pcd.points) < 3:
            return []

        try:
            _, inliers = pcd.segment_plane(0.15, 3, 1000)
            pcd = pcd.select_by_index(inliers, invert=True)
        except RuntimeError:
            pass

        labels = np.array(pcd.cluster_dbscan(self.dbscan_eps, self.dbscan_min_points))
        detections = []

        for label in np.unique(labels[labels >= 0]):
            cluster = pcd.select_by_index(np.where(labels == label)[0])
            if self.is_valid_human_geometry(cluster):
                detections.append({
                    "centroid": cluster.get_center(),
                    "shape": self.extract_shape_descriptor(cluster)
                })

        return detections


# =========================
# Motion Model
# =========================

def build_imm_filter(initial_pos: np.ndarray) -> IMMEstimator:
    def base_kf(q_scale):
        kf = KalmanFilter(dim_x=6, dim_z=3)
        kf.F = np.eye(6)
        kf.H = np.block([[np.eye(3), np.zeros((3, 3))]])
        kf.x[:3] = initial_pos.reshape(3, 1)
        kf.P *= 5.0
        kf.R *= 10.0
        kf.Q = np.eye(6) * q_scale
        return kf

    kf_cv = base_kf(0.01)
    kf_rw = base_kf(0.1)

    mu = np.array([0.5, 0.5])
    trans = np.array([[0.95, 0.05], [0.05, 0.95]])

    return IMMEstimator([kf_cv, kf_rw], mu, trans)


# =========================
# Track
# =========================

class Track:
    def __init__(self, track_id: int, centroid: np.ndarray, shape: np.ndarray):
        self.id = track_id
        self.kf = build_imm_filter(centroid)
        self.shape_history = [shape]
        self.hits = 1
        self.age = 1
        self.skipped_frames = 0
        self.last_seen_pos = centroid
        self.is_static = False

    def predict(self, dt: float):
        for f in self.kf.filters:
            for i in range(3):
                f.F[i, i + 3] = dt
        self.kf.predict()
        self.age += 1

    def update(self, centroid: np.ndarray, shape: np.ndarray):
        self.kf.update(centroid)
        self.shape_history.append(shape)
        self.last_seen_pos = centroid
        self.skipped_frames = 0
        self.hits += 1

    def avg_shape(self) -> np.ndarray:
        return np.mean(self.shape_history[-15:], axis=0)

    def velocity(self) -> np.ndarray:
        return self.kf.x[3:].flatten()

    def confidence(self, eps=1e-6, p_max=50.0) -> float:
        P = self.kf.P[:3, :3]
        unc = np.sqrt(np.linalg.det(P) + eps)
        return float(np.clip(np.exp(-unc / p_max), 0, 1))


# =========================
# Tracker
# =========================

class HumanTrackerMOT:
    def __init__(self):
        self.tracks: List[Track] = []
        self.lost_tracks: List[Track] = []
        self.next_id = 1
        self.history = []

        self.detector = HumanDetector()

        self.min_hits = 3
        self.max_skip_dynamic = 20
        self.max_skip_static = 50
        self.dist_weight = 0.8
        self.shape_weight = 0.2
        self.gating_threshold = 1.2

    def associate(self, detections: List[Dict]):
        if not self.tracks or not detections:
            return {}, set()

        cost = np.full((len(self.tracks), len(detections)), 999.0)

        for i, t in enumerate(self.tracks):
            for j, d in enumerate(detections):
                dist = np.linalg.norm(t.kf.x[:3].flatten() - d["centroid"])
                if dist <= self.gating_threshold:
                    shape_dist = np.linalg.norm(t.avg_shape() - d["shape"])
                    cost[i, j] = self.dist_weight * dist + self.shape_weight * shape_dist

        rows, cols = linear_sum_assignment(cost)
        matches = {}

        for r, c in zip(rows, cols):
            if cost[r, c] < self.gating_threshold:
                matches[r] = c

        return matches, set(matches.values())

    def update(self, frame_id: int, timestamp_ms: int, pcd_path: str):
        if not os.path.exists(pcd_path):
            return

        pcd = o3d.io.read_point_cloud(pcd_path)
        detections = self.detector.detect(pcd)

        prev_ts = self.history[-1]["timestamp_ms"] if self.history else timestamp_ms - 33
        dt = max((timestamp_ms - prev_ts) / 1000.0, 1e-3)

        for t in self.tracks:
            t.predict(dt)

        matches, used_dets = self.associate(detections)

        for ti, di in matches.items():
            self.tracks[ti].update(detections[di]["centroid"], detections[di]["shape"])

        active = []
        for i, t in enumerate(self.tracks):
            if i not in matches:
                t.skipped_frames += 1

            speed = np.linalg.norm(t.velocity())
            t.is_static = speed < 0.15
            limit = self.max_skip_static if t.is_static else self.max_skip_dynamic

            if t.skipped_frames <= limit:
                active.append(t)
            else:
                self.lost_tracks.append(t)

        self.tracks = active

        for j, d in enumerate(detections):
            if j in used_dets:
                continue

            reid = False
            for lt in self.lost_tracks:
                if np.linalg.norm(lt.last_seen_pos - d["centroid"]) < 1.5:
                    lt.kf.x[:3] = d["centroid"].reshape(3, 1)
                    lt.skipped_frames = 0
                    self.tracks.append(lt)
                    self.lost_tracks.remove(lt)
                    reid = True
                    break

            if not reid:
                self.tracks.append(Track(self.next_id, d["centroid"], d["shape"]))
                self.next_id += 1

        results = []
        for t in self.tracks:
            if t.skipped_frames == 0 and t.hits >= self.min_hits:
                pos = t.kf.x[:3].flatten()
                vel = t.velocity()
                results.append({
                    "id": t.id,
                    "position": [round(p, 3) for p in pos],
                    "velocity": [round(v, 3) for v in vel],
                    "speed": round(float(np.linalg.norm(vel)), 3),
                    "status": "STATIC" if t.is_static else "MOVING",
                    "confidence": round(t.confidence(), 3)
                })

        self.history.append({
            "frame_id": frame_id,
            "timestamp_ms": timestamp_ms,
            "detections": results
        })

    def finalize_results(self, output="tracking_results.json"):
        id_counts = {}
        for f in self.history:
            for d in f["detections"]:
                id_counts[d["id"]] = id_counts.get(d["id"], 0) + 1

        valid = {k for k, v in id_counts.items() if v > 10}
        final = []

        for f in self.history:
            dets = [d for d in f["detections"] if d["id"] in valid]
            if dets:
                f["detections"] = dets
                final.append(f)

        with open(output, "w") as f:
            json.dump(final, f, indent=4)

        print(f"✅ Tracking Finished. Results in {output}")


# =========================
# CLI
# =========================

def get_user_choice():
    print("\nChoose Dataset:\n [1] Human Only\n [2] Entire Map")
    while True:
        c = input("Enter 1 or 2: ").strip()
        if c == "1":
            return "mapHumanOnly"
        if c == "2":
            return "mapAll"


def main():
    tracker = HumanTrackerMOT()
    base = os.path.dirname(os.path.abspath(__file__))
    dataset = get_user_choice()
    data_dir = os.path.join(base, dataset)

    files = sorted(
        [f for f in os.listdir(data_dir) if f.endswith(".pcd")],
        key=extract_timestamp_ms
    )

    for i, f in enumerate(files):
        ts = extract_timestamp_ms(f)
        tracker.update(i, ts, os.path.join(data_dir, f))
        if i % 50 == 0:
            print(f"Processing frame {i}/{len(files)}")

    tracker.finalize_results()


if __name__ == "__main__":
    main()
