#!/usr/bin/env python3
# ============================================================
# GNN-based OC-SORT–style 3D Human Tracker (FIXED)
# Input/Output behavior preserved
# ============================================================

import json
import os
import glob
import bisect
import numpy as np
import open3d as o3d
from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------- Constants --------------------
MIN_POINTS_PER_FILE = 2
MIN_POINTS_CLUSTER = 5
EPS_CLUSTER = 1.0
MIN_HEIGHT = -0.5
MAX_HEIGHT = 1.5
MAX_SPEED = 5.0

MAX_MISSED_FRAMES = 8
CHI2_GATE = 9.21  # 95% chi-square gate (3 DoF)

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

# -------------------- Track --------------------
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
        return self.kf.x[:3], self.kf.P[:3, :3]

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

# -------------------- GNN Association --------------------
class AssociationGNN(nn.Module):
    def __init__(self, in_dim=7, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# -------------------- Tracker --------------------
class OCSORT3D_GNN:
    def __init__(self):
        self.tracks = {}
        self.next_id = 0
        self.all_timestamps = []
        self.gnn = AssociationGNN().eval()

    # ---------- Point Cloud ----------
    def filter_human_points(self, pcd):
        pts = np.asarray(pcd.points)
        idx = np.where((pts[:,2] > MIN_HEIGHT) & (pts[:,2] < MAX_HEIGHT))[0]
        return pcd.select_by_index(idx)

    def cluster_humans(self, pcd):
        if len(pcd.points) < MIN_POINTS_CLUSTER:
            return []
        labels = np.array(pcd.cluster_dbscan(EPS_CLUSTER, MIN_POINTS_CLUSTER))
        return [pcd.select_by_index(np.where(labels == l)[0])
                for l in np.unique(labels) if l != -1]

    def centroid(self, cluster):
        return np.mean(np.asarray(cluster.points), axis=0)

    # ---------- GNN Association ----------
    def associate(self, detections, timestamp):
        if not self.tracks:
            return [], list(range(len(detections))), []

        track_ids = list(self.tracks.keys())
        preds, covs = [], []

        for tid in track_ids:
            p, c = self.tracks[tid].predict(timestamp)
            preds.append(p)
            covs.append(c)

        preds = np.asarray(preds)
        dets = np.asarray(detections)

        pair_features = []
        pairs = []

        for i, tid in enumerate(track_ids):
            for j, det in enumerate(dets):
                diff = det - preds[i]
                maha = diff.T @ np.linalg.inv(covs[i]) @ diff
                if maha > CHI2_GATE:
                    continue

                pair_features.append(
                    np.hstack([diff, preds[i], maha])
                )
                pairs.append((tid, j))

        if not pair_features:
            return [], list(range(len(detections))), track_ids

        # ---- FIX: safe tensor + safe shape ----
        feats = torch.from_numpy(np.asarray(pair_features)).float()
        with torch.no_grad():
            scores = self.gnn(feats).view(-1).cpu().numpy()

        order = np.argsort(scores)[::-1]
        matched_tracks, matched_dets = set(), set()
        matches = []

        for idx in order:
            tid, did = pairs[idx]
            if tid in matched_tracks or did in matched_dets:
                continue
            if scores[idx] >= 0.5:
                matches.append((tid, did))
                matched_tracks.add(tid)
                matched_dets.add(did)

        unmatched_tracks = [t for t in track_ids if t not in matched_tracks]
        unmatched_dets = [i for i in range(len(detections)) if i not in matched_dets]

        return matches, unmatched_dets, unmatched_tracks

    # ---------- Update ----------
    def update(self, clusters, timestamp):
        self.all_timestamps.append(timestamp)
        detections = [self.centroid(c) for c in clusters]

        matches, unmatched_dets, unmatched_tracks = self.associate(detections, timestamp)

        for tid, did in matches:
            self.tracks[tid].update(detections[did], timestamp)

        for did in unmatched_dets:
            self.tracks[self.next_id] = Track(self.next_id, detections[did], timestamp)
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
            ts, cs = tr.timestamps, tr.centroids
            new_c, new_t = [], []
            for t in all_ts:
                if t in ts:
                    i = ts.index(t)
                    new_c.append(cs[i])
                    new_t.append(t)
                else:
                    i = bisect.bisect_left(ts, t)
                    if i == 0 or i == len(ts):
                        continue
                    t0, t1 = ts[i-1], ts[i]
                    c0, c1 = cs[i-1], cs[i]
                    a = (t - t0) / (t1 - t0)
                    new_c.append((1-a)*c0 + a*c1)
                    new_t.append(t)
            tr.centroids, tr.timestamps = new_c, new_t

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
            ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=1, c="gray", alpha=0.3)
    for tid, tr in tracks.items():
        c = np.asarray(tr.centroids)
        if len(c) > 1:
            ax.plot(c[:,0], c[:,1], c[:,2], linewidth=2, label=f"ID {tid}")
    ax.legend()
    plt.show()

def plot_velocity(tracks, smoothing_window=5):
    plt.figure(figsize=(10,5))
    for tid, tr in tracks.items():
        c = np.asarray(tr.centroids)
        t = np.asarray(tr.timestamps)
        if len(c) < 2:
            continue
        order = np.argsort(t)
        c, t = c[order], t[order]
        speeds = []
        for i in range(len(t)-1):
            dt = max((t[i+1]-t[i])/1000.0, 1e-3)
            speeds.append(min(np.linalg.norm(c[i+1]-c[i])/dt, MAX_SPEED))
        if len(speeds) >= smoothing_window:
            kernel = np.ones(smoothing_window)/smoothing_window
            speeds = np.convolve(speeds, kernel, mode="same")
        plt.plot(t[1:], speeds, label=f"ID {tid}")
    plt.ylim(0, MAX_SPEED+1)
    plt.legend()
    plt.show()

# -------------------- Main --------------------
def main():
    tracker = OCSORT3D_GNN()
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
            dets = [{"position": tracker.centroid(c).tolist(), "status": "MOVING"}
                    for c in clusters]
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
