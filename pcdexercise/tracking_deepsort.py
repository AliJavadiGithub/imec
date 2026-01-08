import os
import glob
import numpy as np
import open3d as o3d
from scipy.spatial.distance import cdist, cosine
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
import bisect

# --- Constants ---
MIN_POINTS_PER_FILE = 20  # Skip files with fewer points
MIN_POINTS_CLUSTER = 5   # Minimum points for DBSCAN cluster
EPS_CLUSTER = 0.15       # DBSCAN epsilon

# --- DeepSort-inspired Appearance Feature Extractor ---
class PointCloudDescriptor(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# --- Human Tracker with DeepSort-inspired Re-ID ---
class HumanTrackerDeepSort:
    def __init__(self, max_missed_frames=5, feature_dim=32):
        self.tracks = defaultdict(lambda: {
            "centroids": [],
            "timestamps": [],
            "missed_frames": 0,
            "features": []
        })
        self.next_id = 0
        self.max_missed_frames = max_missed_frames
        self.kalman_filters = {}
        self.feature_extractor = PointCloudDescriptor(output_dim=feature_dim)
        self.processed_timestamps = []

    def extract_features(self, cluster):
        points = np.asarray(cluster.points)
        if len(points) == 0:
            return np.zeros(32)
        points = (points - np.mean(points, axis=0)) / (np.std(points, axis=0) + 1e-6)
        points_tensor = torch.FloatTensor(points)
        with torch.no_grad():
            features = self.feature_extractor(points_tensor).mean(dim=0).numpy()
        return features

    def cluster_humans(self, pcd):
        labels = np.array(pcd.cluster_dbscan(EPS_CLUSTER, MIN_POINTS_CLUSTER))
        unique_labels = np.unique(labels)
        clusters = []
        for label in unique_labels:
            if label == -1:
                continue
            cluster = pcd.select_by_index(np.where(labels == label)[0])
            clusters.append(cluster)
        return clusters

    def get_centroid(self, cluster):
        points = np.asarray(cluster.points)
        return np.mean(points, axis=0) if len(points) > 0 else np.zeros(3)

    def init_kalman_filter(self, centroid):
        kf = KalmanFilter(dim_x=6, dim_z=3)
        kf.x = np.concatenate([centroid, [0, 0, 0]])
        kf.F = np.array([
            [1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        kf.P *= 1000
        kf.R = np.eye(3) * 0.1
        kf.Q = np.eye(6) * 0.01
        return kf

    def interpolate_missing_frames(self):
        """Interpolate missing frames for all tracks using Kalman Filter predictions."""
        if not self.processed_timestamps:
            return

        all_timestamps = sorted(set(self.processed_timestamps))

        for track_id, track in self.tracks.items():
            timestamps = track["timestamps"]
            missing_timestamps = sorted(set(all_timestamps) - set(timestamps))

            for ts in missing_timestamps:
                prev_ts = max([t for t in timestamps if t < ts], default=None)
                next_ts = min([t for t in timestamps if t > ts], default=None)

                if prev_ts is None or next_ts is None:
                    continue  # Cannot interpolate without both previous and next frames

                try:
                    prev_idx = timestamps.index(prev_ts)
                    next_idx = timestamps.index(next_ts)
                except ValueError:
                    continue  # Skip if indices are not found

                dt = (ts - prev_ts) / 1000  # Convert to seconds

                # Predict centroid using Kalman Filter
                if track_id in self.kalman_filters:
                    kf = self.kalman_filters[track_id]
                    steps = int(dt * 30)  # Assuming 30 FPS
                    for _ in range(steps):
                        kf.predict()
                    predicted_centroid = kf.x[:3]

                    # Insert predicted centroid
                    insert_idx = bisect.bisect_left(timestamps, ts)
                    track["centroids"].insert(insert_idx, predicted_centroid)
                    track["timestamps"].insert(insert_idx, ts)

                    # Interpolate feature linearly
                    if prev_idx < len(track["features"]) and next_idx < len(track["features"]):
                        alpha = (ts - prev_ts) / (next_ts - prev_ts)
                        interp_feature = (1 - alpha) * np.array(track["features"][prev_idx]) + alpha * np.array(track["features"][next_idx])
                        track["features"].insert(insert_idx, interp_feature)
                    else:
                        # If indices are out of range, use the last known feature
                        track["features"].insert(insert_idx, track["features"][-1])

    def associate_detections_deepsort(self, tracks, detections, features, max_distance=0.5, max_cosine_dist=0.4):
        if not tracks or not detections:
            return []
        track_centroids = np.array([track["centroids"][-1] for track in tracks.values()])
        motion_cost = cdist(track_centroids, detections, metric="euclidean")
        track_features = np.array([track["features"][-1] for track in tracks.values()])
        appearance_cost = cdist(track_features, features, metric=cosine)
        combined_cost = 0.5 * motion_cost + 0.5 * appearance_cost
        row_ind, col_ind = linear_sum_assignment(combined_cost)
        assignments = []
        for r, c in zip(row_ind, col_ind):
            if combined_cost[r, c] < max_distance + max_cosine_dist:
                assignments.append((list(tracks.keys())[r], c))
        return assignments

    def update_tracks(self, clusters, timestamp):
        self.processed_timestamps.append(timestamp)
        detections = []
        features = []
        for cluster in clusters:
            detections.append(self.get_centroid(cluster))
            features.append(self.extract_features(cluster))

        track_ids = list(self.tracks.keys())
        assignments = self.associate_detections_deepsort(self.tracks, detections, features)

        matched_track_ids = []
        for track_id, det_idx in assignments:
            centroid = detections[det_idx]
            feature = features[det_idx]
            self.tracks[track_id]["centroids"].append(centroid)
            self.tracks[track_id]["timestamps"].append(timestamp)
            self.tracks[track_id]["features"].append(feature)
            self.tracks[track_id]["missed_frames"] = 0
            if track_id not in self.kalman_filters:
                self.kalman_filters[track_id] = self.init_kalman_filter(centroid)
            self.kalman_filters[track_id].predict()
            self.kalman_filters[track_id].update(centroid)
            matched_track_ids.append(track_id)

        unmatched_dets = [i for i in range(len(detections)) if i not in [d for (_, d) in assignments]]
        for det_idx in unmatched_dets:
            centroid = detections[det_idx]
            feature = features[det_idx]
            self.tracks[self.next_id] = {
                "centroids": [centroid],
                "timestamps": [timestamp],
                "features": [feature],
                "missed_frames": 0
            }
            self.kalman_filters[self.next_id] = self.init_kalman_filter(centroid)
            self.next_id += 1

        unmatched_tracks = [tid for tid in track_ids if tid not in matched_track_ids]
        for track_id in unmatched_tracks:
            self.tracks[track_id]["missed_frames"] += 1
            if self.tracks[track_id]["missed_frames"] > self.max_missed_frames:
                del self.tracks[track_id]
                del self.kalman_filters[track_id]
            else:
                predicted_centroid = self.kalman_filters[track_id].x[:3]
                self.tracks[track_id]["centroids"].append(predicted_centroid)
                self.tracks[track_id]["timestamps"].append(timestamp)
                self.tracks[track_id]["features"].append(self.tracks[track_id]["features"][-1])

def load_and_check_pcd(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    if not pcd.has_points() or len(pcd.points) < MIN_POINTS_PER_FILE:
        print(f"Skipping sparse file: {file_path} (Points: {len(pcd.points)})")
        return None, None
    return pcd, int(os.path.basename(file_path).split("_")[1].split("ms")[0])

def visualize_tracks(tracks):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    for track_id, track in tracks.items():
        centroids = np.array(track["centroids"])
        ax.plot(centroids[:, 0], centroids[:, 1], centroids[:, 2], label=f"ID {track_id}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.title("Human Trajectories")
    plt.show()

def plot_velocity(tracks):
    plt.figure(figsize=(10, 5))
    for track_id, track in tracks.items():
        velocities = [np.linalg.norm(track["centroids"][i+1] - track["centroids"][i]) /
                      ((track["timestamps"][i+1] - track["timestamps"][i]) / 1000)
                      for i in range(len(track["centroids"])-1)]
        plt.plot(track["timestamps"][1:], velocities, label=f"ID {track_id}")
    plt.xlabel("Time (ms)")
    plt.ylabel("Speed (m/s)")
    plt.title("Human Speed Over Time")
    plt.legend()
    plt.show()

def main():
    tracker = HumanTrackerDeepSort()
    frame_paths = sorted(glob.glob("mapAll/*.pcd"))

    for frame_path in frame_paths:
        pcd, timestamp = load_and_check_pcd(frame_path)
        if pcd is None:
            continue
        clusters = tracker.cluster_humans(pcd)
        if clusters:
            tracker.update_tracks(clusters, timestamp)
        else:
            print(f"No clusters found in: {frame_path}")

    # Interpolate missing frames using Kalman Filter predictions
    tracker.interpolate_missing_frames()

    visualize_tracks(tracker.tracks)
    plot_velocity(tracker.tracks)

    for track_id, track in tracker.tracks.items():
        print(f"Track ID: {track_id}")
        print(f"Centroids: {track['centroids']}")
        print(f"Timestamps: {track['timestamps']}")
        print("---")

if __name__ == "__main__":
    main()
