#!/usr/bin/env python3
"""
Multi-object point cloud tracker - DeepSORT-inspired (3D adapted)
FINAL version with JSON output in the required format

Output saved to: tracking_results.json
"""

import re
import json
from pathlib import Path
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cdist
import lap
from filterpy.kalman import KalmanFilter
from typing import List, Optional, Dict, Any


# ================= CONFIG =================
DATA_FOLDER = Path("mapHumanOnly")
OUTPUT_JSON = Path("tracking_results.json")

VOXEL_SIZE = 0.025
DBSCAN_EPS = 0.15
DBSCAN_MINPTS = 15

MAX_AGE = 20
MIN_CONFIRMED_HITS = 3
ASSOCIATION_THRESHOLD = 0.92
MAX_ASSOCIATION_DISTANCE = 1.8

MOTION_WEIGHT = 0.6
APPEARANCE_WEIGHT = 0.4

FEATURE_DIM = 128
MIN_CLUSTER_POINTS = 35
# ==========================================


def parse_timestamp_ms(p: Path) -> int:
    m = re.search(r'_(\d+)ms$', p.stem)
    return int(m.group(1)) if m else -1


def load_and_preprocess(path: Path) -> Optional[o3d.geometry.PointCloud]:
    try:
        pcd = o3d.io.read_point_cloud(str(path))
        if len(pcd.points) < 80:
            return None
        pcd = pcd.voxel_down_sample(VOXEL_SIZE)
        try:
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.8)
        except:
            pass
        if len(pcd.points) < MIN_CLUSTER_POINTS:
            return None
        return pcd
    except:
        return None


# ── PointNet++ Appearance Extractor ────────────────────────────────────────

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        last_channel = in_channel + 3
        self.mlp_convs = nn.ModuleList()
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            last_channel = out_channel

    def forward(self, xyz, points):
        B, N, _ = xyz.shape
        
        if self.npoint is None:  # Global pooling
            new_xyz = xyz.mean(dim=1, keepdim=True)
            grouped_xyz = xyz.unsqueeze(1) - new_xyz.unsqueeze(2)
            if points is not None:
                grouped_points = points.unsqueeze(1)
                new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
            else:
                new_points = grouped_xyz
        else:
            fps_idx = farthest_point_sample(xyz, self.npoint)
            new_xyz = index_points(xyz, fps_idx)
            idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
            grouped_xyz = index_points(xyz, idx)
            grouped_xyz -= new_xyz.view(B, self.npoint, 1, 3)
            if points is not None:
                grouped_points = index_points(points, idx)
                new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
            else:
                new_points = grouped_xyz

        new_points = new_points.permute(0, 3, 2, 1)
        for conv in self.mlp_convs:
            new_points = F.relu(conv(new_points))
        new_points = torch.max(new_points, 2)[0].permute(0, 2, 1)
        return new_xyz, new_points


class SimplePointNetPlusPlus(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.sa1 = PointNetSetAbstraction(512, 0.2, 32, 0, [64, 64, 128])
        self.sa2 = PointNetSetAbstraction(128, 0.4, 64, 128, [128, 128, 256])
        self.sa3 = PointNetSetAbstraction(None, None, None, 256, [256, 512, 1024])
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, out_dim)
        )

    def forward(self, xyz):
        B = xyz.shape[0]
        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        _, global_feat = self.sa3(l2_xyz, l2_points)
        x = global_feat.view(B, 1024)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)


def farthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[torch.arange(B), farthest].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    dist2 = torch.sum((xyz[:, None] - new_xyz[:, :, None]) ** 2, -1)
    return dist2.topk(nsample, -1, largest=False)[1]


def index_points(points, idx):
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, device=points.device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx]


class AppearanceExtractor:
    def __init__(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.model = SimplePointNetPlusPlus(FEATURE_DIM).to(device)
        self.model.eval()

    def __call__(self, points: np.ndarray) -> np.ndarray:
        if len(points) < 64:
            return np.zeros(FEATURE_DIM, dtype=np.float32)
        centroid = points.mean(axis=0)
        points = points - centroid
        max_dist = np.max(np.linalg.norm(points, axis=1)) + 1e-8
        points /= max_dist
        xyz = torch.from_numpy(points).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(xyz).cpu().numpy().flatten()
        return feat


# ── Track ──────────────────────────────────────────────────────────────────

class Track:
    next_id = 1

    def __init__(self, init_pos: np.ndarray):
        self.id = Track.next_id
        Track.next_id += 1

        self.kf = KalmanFilter(dim_x=6, dim_z=3)
        self.kf.x[:3] = init_pos.reshape(3, 1)

        self.kf.F = np.eye(6)
        self.kf.H = np.eye(3, 6)

        self.kf.P *= 15.0
        self.kf.R = np.eye(3) * 0.15**2
        self.kf.Q = np.eye(6) * 1.2
        self.kf.Q[3:, 3:] *= 5.0

        self.age = 1
        self.hits = 1
        self.time_since_update = 0
        self.feature = None

    def predict(self):
        dt = 0.033
        self.kf.F[0, 3] = dt
        self.kf.F[1, 4] = dt
        self.kf.F[2, 5] = dt
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return self.kf.x[:3].flatten()

    def update(self, meas: np.ndarray, feat: Optional[np.ndarray] = None):
        self.kf.update(meas.reshape(3, 1))
        if feat is not None:
            self.feature = feat.copy()
        self.time_since_update = 0
        self.hits += 1

    def get_position(self) -> np.ndarray:
        return self.kf.x[:3].flatten()

    def get_velocity(self) -> np.ndarray:
        return self.kf.x[3:6].flatten()

    def get_speed(self) -> float:
        return float(np.linalg.norm(self.get_velocity()))

    def get_confidence(self) -> float:
        # Simple heuristic: higher hits → higher confidence
        return min(1.0, 0.3 + 0.1 * self.hits)


# ── Multi-Person Tracker ───────────────────────────────────────────────────

class MultiPersonTracker:
    def __init__(self):
        self.tracks: List[Track] = []
        self.appearance = AppearanceExtractor()
        self.frame_idx = 0
        self.results: List[Dict[str, Any]] = []  # to save in JSON

    def update(self, centroids: List[np.ndarray], features: List[np.ndarray], frame_id: int, timestamp_ms: int):
        self.frame_idx += 1

        for t in self.tracks:
            t.predict()

        matched = []
        unmatched_dets = list(range(len(centroids)))
        unmatched_trks = list(range(len(self.tracks)))

        n_det = len(centroids)
        n_trk = len(self.tracks)

        if n_det > 0 and n_trk > 0:
            cost = self._compute_cost_matrix(centroids, features)

            if cost.shape == (n_det, n_trk) and cost.size > 0 and not np.isnan(cost).any():
                try:
                    row_ind, col_ind, _ = lap.lapjv(cost, extend_cost=True)
                    row_ind = np.atleast_1d(row_ind)
                    col_ind = np.atleast_1d(col_ind)

                    matched = []
                    for k in range(min(len(row_ind), len(col_ind))):
                        i = int(row_ind[k])
                        j = int(col_ind[k])
                        if 0 <= i < n_det and 0 <= j < n_trk and cost[i, j] < ASSOCIATION_THRESHOLD:
                            matched.append((i, j))

                    matched_d = {i for i, _ in matched}
                    matched_t = {j for _, j in matched}

                    unmatched_dets = [i for i in range(n_det) if i not in matched_d]
                    unmatched_trks = [j for j in range(n_trk) if j not in matched_t]
                except Exception as e:
                    print(f"[WARN] Association failed: {e} (cost shape: {cost.shape})")

        for d_idx, t_idx in matched:
            self.tracks[t_idx].update(centroids[d_idx], features[d_idx])

        for i in unmatched_dets:
            if len(centroids[i]) >= 3:
                t = Track(centroids[i])
                t.feature = features[i].copy()
                self.tracks.append(t)

        self.tracks = [t for t in self.tracks if t.time_since_update < MAX_AGE]

        # Build detection list for JSON
        detections = []
        for t in self.tracks:
            if t.hits >= MIN_CONFIRMED_HITS:
                detections.append({
                    "human_id": t.id,
                    "centroid": t.get_position().tolist(),
                    "velocity": t.get_velocity().tolist(),
                    "speed": t.get_speed(),
                    "confidence": t.get_confidence()
                })

        # Save frame result
        self.results.append({
            "frame_id": frame_id,
            "timestamp_ms": timestamp_ms,
            "detections": detections
        })

        return len(detections)


    def _compute_cost_matrix(self, det_centroids: List[np.ndarray], det_features: List[np.ndarray]):
        n_det = len(det_centroids)
        n_trk = len(self.tracks)

        if n_det == 0 or n_trk == 0:
            return np.zeros((n_det, n_trk), dtype=float)

        det_c = np.array(det_centroids)
        trk_c = np.array([t.get_position() for t in self.tracks])

        motion_cost = cdist(det_c, trk_c, 'euclidean')
        motion_cost = np.clip(motion_cost / MAX_ASSOCIATION_DISTANCE, 0, 1.5)

        det_f = np.stack(det_features) if det_features else np.empty((n_det, FEATURE_DIM))
        trk_f = np.array([t.feature for t in self.tracks if t.feature is not None])

        if len(trk_f) > 0 and det_f.shape[1] == trk_f.shape[1]:
            app_cost = 1 - np.dot(det_f, trk_f.T)
            app_cost = np.clip(app_cost, 0, 1.0)
        else:
            app_cost = np.full((n_det, n_trk), 0.7)

        return MOTION_WEIGHT * motion_cost + APPEARANCE_WEIGHT * app_cost


    def save_results(self):
        """Save all frames to JSON"""
        with open(OUTPUT_JSON, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to: {OUTPUT_JSON}")


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    files = sorted(DATA_FOLDER.glob("occupied_*ms.pcd"), key=parse_timestamp_ms)
    tracker = MultiPersonTracker()

    print(f"Found {len(files)} frames. Starting...\n")

    for i, filepath in enumerate(files, 1):
        ts_ms = parse_timestamp_ms(filepath)
        print(f"Frame {i:4d} | {ts_ms//1000:4d}s ", end="")

        pcd = load_and_preprocess(filepath)
        if pcd is None:
            print("→ skipped")
            continue

        labels = np.array(pcd.cluster_dbscan(DBSCAN_EPS, DBSCAN_MINPTS, print_progress=False))

        centroids = []
        features = []

        for lbl in np.unique(labels):
            if lbl == -1:
                continue
            pts = np.asarray(pcd.points)[labels == lbl]
            if len(pts) < MIN_CLUSTER_POINTS:
                continue
            cent = pts.mean(axis=0)
            feat = tracker.appearance(pts)
            centroids.append(cent)
            features.append(feat)

        num_detections = tracker.update(centroids, features, frame_id=i, timestamp_ms=ts_ms)
        print(f"→ {num_detections} active tracks")

    # Save final results
    tracker.save_results()