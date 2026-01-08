#!/usr/bin/env python3
# ============================================================
# OC-SORT–style 3D Human Tracker for Point Clouds
# Input/Output behavior preserved from original script
# ============================================================

import json
import os
import glob
import bisect
import numpy as np
import open3d as o3d
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -------------------- Constants --------------------
MIN_POINTS_PER_FILE = 2
MIN_POINTS_CLUSTER = 5
EPS_CLUSTER = 1.0
MIN_HEIGHT = -0.5
MAX_HEIGHT = 1.5
MAX_SPEED = 5.0

MAX_MISSED_FRAMES = 8
CHI2_GATE = 9.21   # ~95% for 3 DoF
FRAME_RATE_HZ = 30.0

# -------------------- Kalman Filter --------------------
def init_kalman_filter(centroid):
    kf = KalmanFilter(dim_x=6, dim_z=3)
    kf.x = np.hstack([centroid, [0, 0, 0]])

    kf.F = np.eye(6)
    kf.H = np.zeros((3, 6))
    kf.H[:3, :3] = np.eye(3)

    kf.P *= 100.0
    kf.R = np.eye(3) * 0.05
    kf.Q = np.eye(6) * 0.01
    return kf

def update_F(kf, dt):
    kf.F = np.array([
        [1,0,0,dt,0,0],
        [0,1,0,0,dt,0],
        [0,0,1,0,0,dt],
        [0,0,0,1,0,0],
        [0,0,0,0,1,0],
        [0,0,0,0,0,1],
    ])

# -------------------- Track Class --------------------
class Track:
    def __init__(self, tid, centroid, timestamp):
        self.id = tid
        self.kf = init_kalman_filter(centroid)
        self.centroids = [centroid]
        self.timestamps = [timestamp]
        self.missed = 0
        self.last_timestamp = timestamp

    def predict(self, timestamp):
        dt = max((timestamp - self.last_timestamp) / 1000.0, 1e-3)
        update_F(self.kf, dt)
        self.kf.predict()
        self.last_timestamp = timestamp
        return self.kf.x[:3]

    def update(self, centroid, timestamp):
        self.kf.update(centroid)
        self.centroids.append(centroid)
        self.timestamps.append(timestamp)
        self.missed = 0
        self.last_timestamp = timestamp

    def mark_missed(self, timestamp):
        self.centroids.append(self.kf.x[:3].copy())
        self.timestamps.append(timestamp)
        self.missed += 1
        self.last_timestamp = timestamp

# -------------------- OC-SORT Tracker --------------------
class OCSORT3D:
    def __init__(self):
        self.tracks = {}
        self.next_id = 0
        self.all_timestamps = []

    # ---------- Point Cloud Processing ----------
    def filter_human_points(self, pcd):
        pts = np.asarray(pcd.points)
        idx = np.where((pts[:,2] > MIN_HEIGHT) & (pts[:,2] < MAX_HEIGHT))[0]
        return pcd.select_by_index(idx)

    def cluster_humans(self, pcd):
        if len(pcd.points) < MIN_POINTS_CLUSTER:
            return []
        labels = np.array(pcd.cluster_dbscan(EPS_CLUSTER, MIN_POINTS_CLUSTER))
        clusters = []
        for lab in np.unique(labels):
            if lab == -1:
                continue
            clusters.append(pcd.select_by_index(np.where(labels == lab)[0]))
        return clusters

    def centroid(self, cluster):
        return np.mean(np.asarray(cluster.points), axis=0)

    # ---------- Association ----------
    def associate(self, detections, timestamp):
        if not self.tracks:
            return [], list(range(len(detections))), []

        track_ids = list(self.tracks.keys())
        preds = []
        covs = []

        for tid in track_ids:
            pred = self.tracks[tid].predict(timestamp)
            preds.append(pred)
            covs.append(self.tracks[tid].kf.P[:3,:3])

        preds = np.array(preds)
        dets = np.array(detections)

        cost = np.zeros((len(preds), len(dets)))
        gating = np.full(cost.shape, False)

        for i in range(len(preds)):
            for j in range(len(dets)):
                diff = dets[j] - preds[i]
                maha = diff.T @ np.linalg.inv(covs[i]) @ diff
                cost[i,j] = maha
                if maha < CHI2_GATE:
                    gating[i,j] = True
                else:
                    cost[i,j] = 1e6

        row_ind, col_ind = linear_sum_assignment(cost)

        matches, unmatched_tracks, unmatched_dets = [], [], []
        matched_tracks = set()
        matched_dets = set()

        for r, c in zip(row_ind, col_ind):
            if gating[r,c]:
                matches.append((track_ids[r], c))
                matched_tracks.add(track_ids[r])
                matched_dets.add(c)

        for tid in track_ids:
            if tid not in matched_tracks:
                unmatched_tracks.append(tid)

        for i in range(len(detections)):
            if i not in matched_dets:
                unmatched_dets.append(i)

        return matches, unmatched_dets, unmatched_tracks

    # ---------- Update ----------
    def update(self, clusters, timestamp):
        self.all_timestamps.append(timestamp)
        detections = [self.centroid(c) for c in clusters]

        matches, unmatched_dets, unmatched_tracks = self.associate(detections, timestamp)

        for tid, det_idx in matches:
            self.tracks[tid].update(detections[det_idx], timestamp)

        for det_idx in unmatched_dets:
            self.tracks[self.next_id] = Track(self.next_id, detections[det_idx], timestamp)
            self.next_id += 1

        for tid in unmatched_tracks:
            tr = self.tracks[tid]
            tr.mark_missed(timestamp)
            if tr.missed > MAX_MISSED_FRAMES:
                del self.tracks[tid]

    # ---------- Interpolation ----------
    def interpolate_missing(self):
        all_ts = sorted(set(self.all_timestamps))
        for tr in self.tracks.values():
            ts = tr.timestamps
            cs = tr.centroids
            full_c, full_t = [], []
            for t in all_ts:
                if t in ts:
                    idx = ts.index(t)
                    full_c.append(cs[idx])
                    full_t.append(t)
                else:
                    i = bisect.bisect_left(ts, t)
                    if i == 0 or i == len(ts):
                        continue
                    t0, t1 = ts[i-1], ts[i]
                    c0, c1 = cs[i-1], cs[i]
                    alpha = (t - t0) / (t1 - t0)
                    full_c.append((1-alpha)*c0 + alpha*c1)
                    full_t.append(t)
            tr.centroids = full_c
            tr.timestamps = full_t

# -------------------- Utilities --------------------
def load_and_check_pcd(path):
    pcd = o3d.io.read_point_cloud(path)
    if not pcd.has_points() or len(pcd.points) < MIN_POINTS_PER_FILE:
        return None, None
    ts = int(os.path.basename(path).split("_")[1].split("ms")[0])
    return pcd, ts

def visualize_tracks(tracks, env=None):
    if not tracks:
        return
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection="3d")

    if env:
        for pcd in env:
            pts = np.asarray(pcd.points)
            ax.scatter(pts[:,0], pts[:,1], pts[:,2], c="gray", s=1, alpha=0.3)

    for tid, tr in tracks.items():
        c = np.array(tr.centroids)
        if len(c) > 1:
            ax.plot(c[:,0], c[:,1], c[:,2], linewidth=2, label=f"ID {tid}")

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.legend()
    plt.show()

def plot_velocity(tracks, smoothing_window=5):
    plt.figure(figsize=(10,5))
    for tid, tr in tracks.items():
        c = np.array(tr.centroids)
        t = np.array(tr.timestamps)
        if len(c) < 2:
            continue
        order = np.argsort(t)
        c = c[order]
        t = t[order]

        speeds = []
        for i in range(len(t)-1):
            dt = max((t[i+1]-t[i])/1000.0, 1e-3)
            speeds.append(min(np.linalg.norm(c[i+1]-c[i])/dt, MAX_SPEED))

        if len(speeds) >= smoothing_window:
            kernel = np.ones(smoothing_window)/smoothing_window
            speeds = np.convolve(speeds, kernel, mode="same")

        plt.plot(t[1:], speeds, label=f"ID {tid}")

    plt.xlabel("Time (ms)")
    plt.ylabel("Speed (m/s)")
    plt.ylim(0, MAX_SPEED+1)
    plt.legend()
    plt.show()

# -------------------- Main --------------------
def main():
    tracker = OCSORT3D()
    paths = sorted(glob.glob("mapAll/*.pcd"))
    env = []
    tracking_data = []

    for idx, path in enumerate(paths):
        pcd, ts = load_and_check_pcd(path)
        if pcd is None:
            continue

        human = tracker.filter_human_points(pcd)
        env.append(pcd)
        clusters = tracker.cluster_humans(human)

        if clusters:
            tracker.update(clusters, ts)
            dets = []
            for cl in clusters:
                dets.append({
                    "position": tracker.centroid(cl).tolist(),
                    "status": "MOVING"
                })
            tracking_data.append({
                "frame_id": idx,
                "timestamp": ts,
                "detections": dets
            })

    tracker.interpolate_missing()

    with open("tracking_results.json", "w") as f:
        json.dump(tracking_data, f, indent=4)

    visualize_tracks(tracker.tracks, env)
    plot_velocity(tracker.tracks)

if __name__ == "__main__":
    main()
