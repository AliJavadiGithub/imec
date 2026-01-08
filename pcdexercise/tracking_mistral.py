#!/usr/bin/env python3
"""
Multi-object point cloud tracker - DeepSORT-inspired (3D adapted)
Fixed version with robust association handling and improved tracking
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
from scipy.spatial.distance import cdist, euclidean
import lap
from filterpy.kalman import KalmanFilter
from typing import List, Optional, Dict, Any
from collections import defaultdict

# ================= CONFIG =================
DATA_FOLDER = Path("mapHumanOnly")
OUTPUT_JSON = Path("tracking_results.json")

# Clustering parameters
VOXEL_SIZE = 0.02       # Finer voxelization
DBSCAN_EPS = 0.25       # Larger clusters
DBSCAN_MINPTS = 10      # Fewer points per cluster

# Tracking parameters
MAX_AGE = 30            # Keep tracks longer
MIN_CONFIRMED_HITS = 2  # Confirm tracks faster
ASSOCIATION_THRESHOLD = 0.8  # Easier association
MAX_ASSOCIATION_DISTANCE = 2.0  # Larger search radius

# Cost matrix weights
MOTION_WEIGHT = 0.5     # Reduced motion weight
APPEARANCE_WEIGHT = 0.5 # Increased appearance weight

FEATURE_DIM = 128
MIN_CLUSTER_POINTS = 20  # Reduced minimum points
# ==========================================

def parse_timestamp_ms(p: Path) -> int:
    m = re.search(r'_(\d+)ms$', p.stem)
    return int(m.group(1)) if m else -1

def load_and_preprocess(path: Path) -> Optional[o3d.geometry.PointCloud]:
    try:
        pcd = o3d.io.read_point_cloud(str(path))
        if len(pcd.points) < 50:  # Reduced minimum points
            return None
        pcd = pcd.voxel_down_sample(VOXEL_SIZE)
        try:
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        except:
            pass
        if len(pcd.points) < MIN_CLUSTER_POINTS:
            return None
        return pcd
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def merge_close_clusters(clusters, threshold=0.3):
    """Merge clusters that are too close to each other"""
    if not clusters:
        return clusters

    # Convert to numpy array for distance calculation
    centroids = np.array([np.mean(cluster, axis=0) for cluster in clusters])
    n_clusters = len(centroids)

    # Compute distance matrix
    dist_matrix = cdist(centroids, centroids)

    # Find clusters to merge
    to_merge = set()
    for i in range(n_clusters):
        for j in range(i+1, n_clusters):
            if dist_matrix[i,j] < threshold:
                to_merge.add((i, j))

    # If no merges needed, return original
    if not to_merge:
        return clusters

    # Create union-find structure to group clusters
    parent = list(range(n_clusters))
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    def union(u, v):
        root_u = find(u)
        root_v = find(v)
        if root_u != root_v:
            parent[root_v] = root_u

    for i, j in to_merge:
        union(i, j)

    # Group clusters by their root parent
    groups = defaultdict(list)
    for i in range(n_clusters):
        groups[find(i)].append(i)

    # Merge clusters in each group
    merged_clusters = []
    for group in groups.values():
        if len(group) == 1:
            merged_clusters.append(clusters[group[0]])
        else:
            # Combine all points in the group
            merged_points = np.vstack([clusters[i] for i in group])
            merged_clusters.append(merged_points)

    return merged_clusters

# ── PointNet++ Appearance Extractor with fixes ────────────────────────────
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
            # Fixed farthest point sampling with size checks
            if N < self.npoint:
                # If we have fewer points than requested, just use all points
                fps_idx = torch.arange(N, device=xyz.device).unsqueeze(0).repeat(B, 1)
            else:
                fps_idx = farthest_point_sample(xyz, self.npoint)
            new_xyz = index_points(xyz, fps_idx)

            # Fixed query ball point with size checks
            idx = query_ball_point(self.radius, min(self.nsample, N), xyz, new_xyz)
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

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Fixed version with proper size handling
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape

    # Compute squared distances
    dist2 = torch.sum((xyz.unsqueeze(1) - new_xyz.unsqueeze(2)) ** 2, dim=-1)

    # For each center, find the nsample closest points
    idx = torch.topk(dist2, k=min(nsample, N), dim=2, largest=False)[1]
    return idx

def farthest_point_sample(xyz, npoint):
    """
    Fixed version with proper size handling
    """
    device = xyz.device
    B, N, C = xyz.shape

    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10

    # Initialize with random point
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[torch.arange(B), farthest].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return centroids

def index_points(points, idx):
    """
    Fixed version with proper size handling
    """
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, device=points.device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx]

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
        try:
            l1_xyz, l1_points = self.sa1(xyz, None)
            l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
            _, global_feat = self.sa3(l2_xyz, l2_points)
            x = global_feat.view(B, 1024)
            x = self.fc(x)
            return F.normalize(x, p=2, dim=1)
        except Exception as e:
            print(f"PointNet forward failed: {e}")
            # Return zero features if processing fails
            return torch.zeros(B, self.fc[-1].out_features, device=xyz.device)

class AppearanceExtractor:
    def __init__(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.model = SimplePointNetPlusPlus(FEATURE_DIM).to(device)
        self.model.eval()

    def __call__(self, points: np.ndarray) -> np.ndarray:
        try:
            if len(points) < 30:
                return np.zeros(FEATURE_DIM, dtype=np.float32)
            centroid = points.mean(axis=0)
            points = points - centroid
            max_dist = np.max(np.linalg.norm(points, axis=1)) + 1e-8
            if max_dist > 0:
                points /= max_dist
            xyz = torch.from_numpy(points).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.model(xyz).cpu().numpy().flatten()
            return feat
        except Exception as e:
            print(f"Feature extraction failed: {e}")
            return np.zeros(FEATURE_DIM, dtype=np.float32)

# ── Track with Improved Kalman Filter ────────────────────────────────────────
class Track:
    next_id = 1

    def __init__(self, init_pos: np.ndarray):
        self.id = Track.next_id
        Track.next_id += 1

        self.kf = KalmanFilter(dim_x=6, dim_z=3)
        self.kf.x[:3] = init_pos.reshape(3, 1)

        self.kf.F = np.eye(6)
        self.kf.H = np.eye(3, 6)

        self.kf.P *= 10.0  # Reduced initial uncertainty
        self.kf.R = np.eye(3) * 0.05**2  # More trust in measurements
        self.kf.Q = np.eye(6) * 0.1  # Increased process noise
        self.kf.Q[3:, 3:] *= 10.0  # Allow faster velocity changes

        self.age = 1
        self.hits = 1
        self.time_since_update = 0
        self.feature = None
        self.history = []  # Store position history for smoothing

    def predict(self):
        dt = 0.033  # ~30fps
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
        self.history.append(meas.copy())  # Store for smoothing

    def get_position(self) -> np.ndarray:
        return self.kf.x[:3].flatten()

    def get_smoothed_position(self) -> np.ndarray:
        """Return smoothed position using history"""
        if len(self.history) < 2:
            return self.get_position()
        # Simple moving average of last 5 positions
        return np.mean(self.history[-min(5, len(self.history)):], axis=0)

    def get_velocity(self) -> np.ndarray:
        return self.kf.x[3:6].flatten()

    def get_speed(self) -> float:
        return float(np.linalg.norm(self.get_velocity()))

    def get_confidence(self) -> float:
        # Higher confidence for tracks with more hits and recent updates
        return min(1.0, 0.2 + 0.2 * self.hits - 0.05 * self.time_since_update)

# ── Multi-Person Tracker with Improvements ──────────────────────────────────
class MultiPersonTracker:
    def __init__(self):
        self.tracks: List[Track] = []
        self.appearance = AppearanceExtractor()
        self.frame_idx = 0
        self.results: List[Dict[str, Any]] = []
        self.last_timestamp_ms = -1
        self.debug_info = []

    def update(self, centroids: List[np.ndarray], features: List[np.ndarray], frame_id: int, timestamp_ms: int):
        self.frame_idx += 1

        # Handle missing frames
        if self.last_timestamp_ms > 0:
            dt_ms = timestamp_ms - self.last_timestamp_ms
            expected_dt_ms = 33  # ~30fps
            if dt_ms > 1.5 * expected_dt_ms:
                num_missing = int((dt_ms - expected_dt_ms) / expected_dt_ms)
                print(f"[WARN] Detected {num_missing} missing frames between {self.last_timestamp_ms}ms and {timestamp_ms}ms")
                self._handle_missing_frames(num_missing, timestamp_ms)

        # Predict existing tracks
        for t in self.tracks:
            t.predict()

        # Associate detections with tracks
        matched, unmatched_dets, unmatched_trks = self._associate_detections(centroids, features)

        # Update matched tracks
        for d_idx, t_idx in matched:
            self.tracks[t_idx].update(centroids[d_idx], features[d_idx])

        # Create new tracks for unmatched detections
        for i in unmatched_dets:
            if len(centroids[i]) >= 3:
                t = Track(centroids[i])
                t.feature = features[i].copy()
                self.tracks.append(t)

        # Remove lost tracks
        self.tracks = [t for t in self.tracks if t.time_since_update < MAX_AGE]

        # Build detection list for JSON
        detections = []
        for t in self.tracks:
            if t.hits >= MIN_CONFIRMED_HITS:
                detections.append({
                    "human_id": t.id,
                    "centroid": t.get_smoothed_position().tolist(),
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
        self.last_timestamp_ms = timestamp_ms

        # Debug info
        self.debug_info.append({
            "frame_id": frame_id,
            "num_detections": len(centroids),
            "num_tracks": len(self.tracks),
            "num_matched": len(matched)
        })

        return len(detections)

    def _associate_detections(self, centroids, features):
        """Robust association with proper error handling"""
        n_det = len(centroids)
        n_trk = len(self.tracks)

        # If no detections or tracks, return empty associations
        if n_det == 0 or n_trk == 0:
            return [], list(range(n_det)), list(range(n_trk))

        try:
            # Compute cost matrix
            cost = self._compute_cost_matrix(centroids, features)

            # Validate cost matrix
            if cost.size == 0 or np.isnan(cost).any():
                print("[WARN] Invalid cost matrix shape or content")
                return [], list(range(n_det)), list(range(n_trk))

            # Use Hungarian algorithm without extend_limit
            try:
                # Get assignment without extend_limit
                row_ind, col_ind, _ = lap.lapjv(cost)

                # Convert to proper numpy arrays
                row_ind = np.atleast_1d(row_ind)
                col_ind = np.atleast_1d(col_ind)

                # Find valid matches
                matched = []
                for k in range(min(len(row_ind), len(col_ind))):
                    i = int(row_ind[k])
                    j = int(col_ind[k])
                    if i >= 0 and j >= 0 and i < n_det and j < n_trk and cost[i, j] < ASSOCIATION_THRESHOLD:
                        matched.append((i, j))

                # Determine unmatched detections and tracks
                matched_d = {i for i, _ in matched}
                matched_t = {j for _, j in matched}

                unmatched_dets = [i for i in range(n_det) if i not in matched_d]
                unmatched_trks = [j for j in range(n_trk) if j not in matched_t]

                return matched, unmatched_dets, unmatched_trks

            except Exception as e:
                print(f"[WARN] Hungarian algorithm failed: {str(e)}")
                return [], list(range(n_det)), list(range(n_trk))

        except Exception as e:
            print(f"[WARN] Association computation failed: {str(e)}")
            return [], list(range(n_det)), list(range(n_trk))

    def _handle_missing_frames(self, num_missing: int, next_timestamp_ms: int):
        """Improved handling of missing frames with interpolation"""
        dt_ms = 33  # ~30fps
        for i in range(1, num_missing + 1):
            current_ts = self.last_timestamp_ms + i * dt_ms

            # Predict all tracks forward
            for t in self.tracks:
                t.predict()

            # For missing frames, save predicted positions
            if i == num_missing:  # Only save the last predicted frame
                detections = []
                for t in self.tracks:
                    if t.hits >= MIN_CONFIRMED_HITS:
                        detections.append({
                            "human_id": t.id,
                            "centroid": t.get_smoothed_position().tolist(),
                            "velocity": t.get_velocity().tolist(),
                            "speed": t.get_speed(),
                            "confidence": t.get_confidence() * 0.7,  # Lower confidence
                            "is_predicted": True
                        })

                self.results.append({
                    "frame_id": self.frame_idx,
                    "timestamp_ms": current_ts,
                    "detections": detections,
                    "is_predicted": True
                })

    def _compute_cost_matrix(self, det_centroids, det_features):
        """Improved cost matrix computation with validation"""
        n_det = len(det_centroids)
        n_trk = len(self.tracks)

        if n_det == 0 or n_trk == 0:
            return np.zeros((n_det, n_trk))

        try:
            # Position cost
            det_c = np.array(det_centroids)
            trk_c = np.array([t.get_smoothed_position() for t in self.tracks])
            motion_cost = cdist(det_c, trk_c, 'euclidean')
            motion_cost = np.clip(motion_cost / MAX_ASSOCIATION_DISTANCE, 0, 1.0)

            # Feature cost
            if det_features and len(det_features) > 0:
                det_f = np.stack(det_features)
            else:
                det_f = np.zeros((n_det, FEATURE_DIM))

            trk_f = np.array([t.feature for t in self.tracks if t.feature is not None] or [np.zeros(FEATURE_DIM)])

            # Handle feature dimension mismatches
            if det_f.shape[1] != trk_f.shape[1]:
                min_dim = min(det_f.shape[1], trk_f.shape[1])
                det_f = det_f[:, :min_dim]
                trk_f = trk_f[:, :min_dim]

            if len(trk_f) > 0 and det_f.shape[1] == trk_f.shape[1]:
                app_cost = 1 - np.dot(det_f, trk_f.T)
                app_cost = np.clip(app_cost, 0, 1.0)
            else:
                app_cost = np.full((n_det, n_trk), 0.5)

            # Combined cost
            return MOTION_WEIGHT * motion_cost + APPEARANCE_WEIGHT * app_cost

        except Exception as e:
            print(f"[WARN] Cost matrix computation failed: {str(e)}")
            return np.ones((n_det, n_trk))  # Return high cost matrix as fallback

    def save_results(self):
        """Save all frames to JSON and debug info"""
        with open(OUTPUT_JSON, 'w') as f:
            json.dump(self.results, f, indent=2)

        # Save debug info to separate file
        with open("tracking_debug.json", 'w') as f:
            json.dump(self.debug_info, f, indent=2)

        print(f"\nResults saved to: {OUTPUT_JSON}")
        print(f"Debug info saved to: tracking_debug.json")

# ── Main with Debug Visualization ────────────────────────────────────────────
if __name__ == "__main__":
    files = sorted(DATA_FOLDER.glob("occupied_*ms.pcd"), key=parse_timestamp_ms)
    tracker = MultiPersonTracker()

    print(f"Found {len(files)} frames. Starting with improved parameters...\n")

    for i, filepath in enumerate(files, 1):
        ts_ms = parse_timestamp_ms(filepath)
        print(f"Frame {i:4d} | {ts_ms//1000:4d}s ", end="")

        pcd = load_and_preprocess(filepath)
        if pcd is None:
            print("→ skipped (no points)")
            continue

        # Get clusters with improved parameters
        labels = np.array(pcd.cluster_dbscan(DBSCAN_EPS, DBSCAN_MINPTS, print_progress=False))

        # Extract clusters
        clusters = []
        for lbl in np.unique(labels):
            if lbl == -1:
                continue
            pts = np.asarray(pcd.points)[labels == lbl]
            if len(pts) >= MIN_CLUSTER_POINTS:
                clusters.append(pts)

        # Merge close clusters
        if clusters:
            clusters = merge_close_clusters(clusters, threshold=0.4)

        centroids = []
        features = []

        for cluster in clusters:
            try:
                cent = np.mean(cluster, axis=0)
                feat = tracker.appearance(cluster)
                centroids.append(cent)
                features.append(feat)
            except Exception as e:
                print(f"[WARN] Failed to process cluster: {e}")
                continue

        num_detections = tracker.update(centroids, features, frame_id=i, timestamp_ms=ts_ms)
        print(f"→ {num_detections} active tracks | {len(centroids)} detections")

    # Save final results
    tracker.save_results()
