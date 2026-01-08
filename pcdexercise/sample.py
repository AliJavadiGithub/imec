import os
import glob
import numpy as np
import open3d as o3d
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_pcd_safe(file_path):
    try:
        pcd = o3d.io.read_point_cloud(file_path)
        if len(pcd.points) == 0:
            print(f"Warning: Empty PCD file {file_path}")
            return None
        return pcd
    except Exception as e:
        print(f"Warning: Failed to load {file_path}: {e}")
        return None

class HumanTracker:
    def __init__(self, max_missed_frames=5):
        self.tracks = {}
        self.next_id = 0
        self.max_missed_frames = max_missed_frames

    def _init_kalman_filter(self, initial_centroid):
        kf = KalmanFilter(dim_x=6, dim_z=3)
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
        kf.Q = np.eye(6) * 0.1
        kf.R = np.eye(3) * 1.0
        kf.x = np.concatenate([initial_centroid, [0, 0, 0]])
        return kf

    def cluster_humans(self, pcd, eps=0.15, min_points=20):
        if pcd is None:
            return []
        labels = np.array(pcd.cluster_dbscan(eps, min_points))
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
        return np.mean(points, axis=0)

    def associate_detections(self, tracks, detections, max_distance=0.5):
        if not tracks or not detections:
            return []
        track_centroids = np.array([track["kalman"].x[:3] for track in tracks.values()])
        cost_matrix = cdist(track_centroids, detections, metric="euclidean")
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assignments = []
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < max_distance:
                assignments.append((list(tracks.keys())[r], c))
        return assignments

    def update_tracks(self, detections, timestamp):
        for track_id in list(self.tracks.keys()):
            self.tracks[track_id]["kalman"].predict()
            self.tracks[track_id]["missed_frames"] += 1

        assignments = self.associate_detections(self.tracks, detections)
        matched_track_ids = set()
        for track_id, det_idx in assignments:
            self.tracks[track_id]["kalman"].update(detections[det_idx])
            self.tracks[track_id]["centroids"].append(detections[det_idx])
            self.tracks[track_id]["timestamps"].append(timestamp)
            self.tracks[track_id]["missed_frames"] = 0
            matched_track_ids.add(track_id)

        unmatched_detections = [i for i in range(len(detections)) if i not in [d for _, d in assignments]]
        for det_idx in unmatched_detections:
            new_id = self.next_id
            self.tracks[new_id] = {
                "kalman": self._init_kalman_filter(detections[det_idx]),
                "centroids": [detections[det_idx]],
                "timestamps": [timestamp],
                "missed_frames": 0
            }
            self.next_id += 1

        lost_tracks = [track_id for track_id in self.tracks if self.tracks[track_id]["missed_frames"] > self.max_missed_frames]
        for track_id in lost_tracks:
            del self.tracks[track_id]

def visualize_tracks(tracks):
    fig = plt.figure(figsize=(12, 6))
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

def main():
    tracker = HumanTracker()
    frame_paths = sorted(glob.glob("mapHumanOnly/*.pcd"))
    for frame_path in frame_paths:
        timestamp = int(os.path.basename(frame_path).split("_")[1].split("ms")[0])
        pcd = load_pcd_safe(frame_path)
        if pcd is None:
            continue
        clusters = tracker.cluster_humans(pcd)
        detections = [tracker.get_centroid(cluster) for cluster in clusters]
        tracker.update_tracks(detections, timestamp)

    visualize_tracks(tracker.tracks)

if __name__ == "__main__":
    main()
