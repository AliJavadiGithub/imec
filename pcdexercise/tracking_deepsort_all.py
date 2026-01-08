import json
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
MIN_POINTS_PER_FILE = 2
MIN_POINTS_CLUSTER = 5
EPS_CLUSTER = 1.0
MIN_HEIGHT = -0.5
MAX_HEIGHT = 1.5
MAX_SPEED = 5.0

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

    def filter_human_points(self, pcd):
        points = np.asarray(pcd.points)
        human_points_idx = np.where((points[:, 2] > MIN_HEIGHT) & (points[:, 2] < MAX_HEIGHT))[0]
        return pcd.select_by_index(human_points_idx)

    def cluster_humans(self, pcd):
        if len(np.asarray(pcd.points)) < MIN_POINTS_CLUSTER:
            return []
        try:
            labels = np.array(pcd.cluster_dbscan(EPS_CLUSTER, MIN_POINTS_CLUSTER))
            clusters = [pcd.select_by_index(np.where(labels == lab)[0]) for lab in np.unique(labels) if lab != -1]
            return clusters
        except:
            return []

    def get_centroid(self, cluster):
        points = np.asarray(cluster.points)
        return np.mean(points, axis=0) if len(points) > 0 else np.zeros(3)

    def init_kalman_filter(self, centroid):
        kf = KalmanFilter(dim_x=6, dim_z=3)
        kf.x = np.concatenate([centroid, [0, 0, 0]])
        kf.F = np.array([
            [1,0,0,1,0,0],
            [0,1,0,0,1,0],
            [0,0,1,0,0,1],
            [0,0,0,1,0,0],
            [0,0,0,0,1,0],
            [0,0,0,0,0,1]
        ])
        kf.H = np.eye(3,6)
        kf.P *= 1000
        kf.R = np.eye(3)*0.1
        kf.Q = np.eye(6)*0.01
        return kf

    def interpolate_missing_frames(self):
        if not self.processed_timestamps:
            return
        all_ts = sorted(set(self.processed_timestamps))
        for tid, track in self.tracks.items():
            timestamps = track["timestamps"]
            missing_ts = sorted(set(all_ts) - set(timestamps))
            for ts in missing_ts:
                prev_ts = max([t for t in timestamps if t < ts], default=None)
                next_ts = min([t for t in timestamps if t > ts], default=None)
                if prev_ts is None or next_ts is None:
                    continue
                prev_idx = timestamps.index(prev_ts)
                next_idx = timestamps.index(next_ts)
                dt = (ts - prev_ts)/1000
                if tid in self.kalman_filters:
                    kf = self.kalman_filters[tid]
                    steps = int(dt*30)
                    for _ in range(steps):
                        kf.predict()
                    pred = kf.x[:3]
                    insert = bisect.bisect_left(timestamps, ts)
                    track["centroids"].insert(insert, pred)
                    track["timestamps"].insert(insert, ts)
                    alpha = (ts-prev_ts)/(next_ts-prev_ts)
                    fprev = np.array(track["features"][prev_idx])
                    fnext = np.array(track["features"][next_idx])
                    track["features"].insert(insert, (1-alpha)*fprev + alpha*fnext)

    def associate_detections_deepsort(self, tracks, detections, features, max_distance=0.5, max_cosine_dist=0.4):
        if not tracks or not detections:
            return []
        track_centroids = np.array([track["centroids"][-1] for track in tracks.values()])
        motion_cost = cdist(track_centroids, detections, metric="euclidean")
        track_features = np.array([track["features"][-1] for track in tracks.values()])
        appearance_cost = cdist(track_features, features, metric=cosine)
        combined_cost = 0.5*motion_cost + 0.5*appearance_cost
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

        matched = []
        for tid, det_idx in assignments:
            c = detections[det_idx]
            f = features[det_idx]
            t = self.tracks[tid]
            t["centroids"].append(c)
            t["timestamps"].append(timestamp)
            t["features"].append(f)
            t["missed_frames"] = 0
            if tid not in self.kalman_filters:
                self.kalman_filters[tid] = self.init_kalman_filter(c)
            self.kalman_filters[tid].predict()
            self.kalman_filters[tid].update(c)
            matched.append(tid)

        unmatched_dets = [i for i in range(len(detections)) if i not in [d for (_, d) in assignments]]
        for det_idx in unmatched_dets:
            c = detections[det_idx]
            f = features[det_idx]
            self.tracks[self.next_id] = {
                "centroids":[c],
                "timestamps":[timestamp],
                "features":[f],
                "missed_frames":0
            }
            self.kalman_filters[self.next_id] = self.init_kalman_filter(c)
            self.next_id += 1

        unmatched_tracks = [tid for tid in track_ids if tid not in matched]
        for tid in unmatched_tracks:
            t = self.tracks[tid]
            t["missed_frames"] += 1
            if t["missed_frames"] > self.max_missed_frames:
                del self.tracks[tid]
                del self.kalman_filters[tid]
            else:
                pred = self.kalman_filters[tid].x[:3]
                t["centroids"].append(pred)
                t["timestamps"].append(timestamp)
                t["features"].append(t["features"][-1])

def load_and_check_pcd(path):
    pcd = o3d.io.read_point_cloud(path)
    if not pcd.has_points() or len(pcd.points) < MIN_POINTS_PER_FILE:
        return None, None
    timestamp = int(os.path.basename(path).split("_")[1].split("ms")[0])
    return pcd, timestamp

def visualize_tracks(tracks, env=None):
    if not tracks: return
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection="3d")
    if env:
        for pcd in env:
            pts = np.asarray(pcd.points)
            ax.scatter(pts[:,0], pts[:,1], pts[:,2], c='gray', s=1, alpha=0.3)
    for tid, t in tracks.items():
        c = np.array(t["centroids"])
        if len(c)>1: ax.plot(c[:,0],c[:,1],c[:,2], label=f"ID {tid}", linewidth=2)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.legend(); plt.show()

# --- Modified Speed Plot with Smoothing ---
def plot_velocity(tracks, smoothing_window=5):
    if not tracks:
        print("No tracks to plot velocity.")
        return

    plt.figure(figsize=(10,5))
    for track_id, track in tracks.items():
        centroids = np.array(track["centroids"])
        timestamps = np.array(track["timestamps"])
        if len(centroids)<2: continue
        order = np.argsort(timestamps)
        centroids = centroids[order]
        timestamps = timestamps[order]

        speeds = []
        for i in range(len(timestamps)-1):
            dt = timestamps[i+1] - timestamps[i]
            if dt <= 0:
                dt += 110000
            dist = np.linalg.norm(centroids[i+1]-centroids[i])
            speeds.append(min(dist/(dt/1000), MAX_SPEED))

        speeds = np.array(speeds)
        ts = timestamps[1:]

        # --- Smoothing ---
        if len(speeds) >= smoothing_window:
            kernel = np.ones(smoothing_window)/smoothing_window
            smoothed = np.convolve(speeds, kernel, mode='same')
        else:
            smoothed = speeds

        plt.plot(ts, smoothed, label=f"ID {track_id} (smoothed)")

    plt.xlabel("Time (ms)")
    plt.ylabel("Speed (m/s)")
    plt.title("Smoothed Human Speed Over Time")
    plt.ylim(0, MAX_SPEED+1)
    plt.legend()
    plt.show()

def main():
    tracker = HumanTrackerDeepSort()
    paths = sorted(glob.glob("mapAll/*.pcd"))
    env = []
    tracking_data = []

    for idx, path in enumerate(paths):
        pcd, ts = load_and_check_pcd(path)
        if pcd is None: continue
        human = tracker.filter_human_points(pcd)
        env.append(pcd)
        clusters = tracker.cluster_humans(human)
        if clusters:
            tracker.update_tracks(clusters, ts)
            dets = []
            for cl in clusters:
                dets.append({"position":tracker.get_centroid(cl).tolist(),"status":"MOVING"})
            tracking_data.append({"frame_id":idx,"timestamp":ts,"detections":dets})

    tracker.interpolate_missing_frames()

    with open("tracking_results.json","w") as f:
        json.dump(tracking_data, f, indent=4)

    visualize_tracks(tracker.tracks, env)
    plot_velocity(tracker.tracks)

if __name__ == "__main__":
    main()
