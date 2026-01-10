import os
import re
import json
import numpy as np
import open3d as o3d
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

# --- System Settings ---
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

class Track:
    def __init__(self, id, centroid, features):
        self.id = id
        # State: [x, y, z, vx, vy, vz]
        self.kf = self.init_kalman(centroid)
        self.feature_history = [features]
        self.hits = 1           # Consecutive detections
        self.age = 1            # Total frames active
        self.skipped_frames = 0 
        self.is_static = False
        self.last_seen_pos = centroid

    def init_kalman(self, pos):
        kf = KalmanFilter(dim_x=6, dim_z=3)
        kf.F = np.eye(6) # Constant velocity transition
        kf.H = np.array([[1, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0]])
        kf.x[:3] = pos.reshape(3, 1)
        kf.P *= 5.0      # Initial uncertainty
        kf.R *= 15.0     # High measurement noise to trust the model (Smoothes Speed)
        kf.Q = np.eye(6) * 0.01 # Low process noise for steady motion
        return kf

    def get_avg_features(self):
        return np.mean(self.feature_history[-10:], axis=0)

class HumanTrackerMOT:
    def __init__(self):
        self.tracks = []
        self.next_id = 1
        self.lost_tracks = [] 
        self.history = []
        
        # Stability Hyperparameters
        self.min_hits = 5           # Frames to confirm ID (Filters ghosts)
        self.max_skip_dynamic = 25  # Grace period for moving targets
        self.max_skip_static = 60   # Grace period for stationary targets
        self.gating_threshold = 1.0 # Max meters per frame (Prevents ID swaps)
        
        # Detection Hyperparameters
        self.eps = 0.55             # Bridging gaps in sparse point clouds
        self.min_points = 8

    def is_valid_detection(self, features):
        w, l, h, pts = features
        # Filter for typical human bounding box dimensions and point density
        return (0.4 < h < 2.0) and (0.2 < w < 1.0) and (pts > 10)

    def extract_features(self, cluster):
        bbox = cluster.get_axis_aligned_bounding_box()
        extent = bbox.get_extent() 
        return np.array([extent[0], extent[1], extent[2], len(cluster.points)])

    def update(self, frame_id, timestamp, pcd_file_path):
        detections = []
        if os.path.exists(pcd_file_path):
            pcd = o3d.io.read_point_cloud(pcd_file_path)
            if not pcd.is_empty():
                # Pre-processing: Density normalization and noise removal
                pcd = pcd.voxel_down_sample(voxel_size=0.04)
                pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
                
                # Clustering
                labels = np.array(pcd.cluster_dbscan(eps=self.eps, min_points=self.min_points))
                
                for label in np.unique(labels[labels >= 0]):
                    indices = np.where(labels == label)[0]
                    cluster = pcd.select_by_index(indices)
                    feat = self.extract_features(cluster)
                    if self.is_valid_detection(feat):
                        detections.append({'centroid': cluster.get_center(), 'features': feat})

        # Predict Kalman states based on actual dt
        prev_ts = self.history[-1]['timestamp_ms'] if self.history else timestamp - 33
        dt = max((timestamp - prev_ts) / 1000.0, 0.001)
        for t in self.tracks:
            for i in range(3): t.kf.F[i, i+3] = dt
            t.kf.predict()
            t.age += 1

        # Association using the Hungarian Algorithm
        assigned_tracks, assigned_dets = set(), set()
        if self.tracks and detections:
            cost_matrix = np.zeros((len(self.tracks), len(detections)))
            for i, track in enumerate(self.tracks):
                for j, det in enumerate(detections):
                    dist = np.linalg.norm(track.kf.x[:3].flatten() - det['centroid'])
                    # Apply spatial gating to prohibit distant associations
                    cost_matrix[i, j] = dist if dist < self.gating_threshold else 999.0

            rows, cols = linear_sum_assignment(cost_matrix)
            for r, c in zip(rows, cols):
                if cost_matrix[r, c] < self.gating_threshold:
                    self.tracks[r].kf.update(detections[c]['centroid'])
                    self.tracks[r].last_seen_pos = detections[c]['centroid']
                    self.tracks[r].skipped_frames = 0
                    self.tracks[r].hits += 1
                    assigned_tracks.add(r)
                    assigned_dets.add(c)

        # Track management and recovery
        active = []
        for i, t in enumerate(self.tracks):
            if i not in assigned_tracks: t.skipped_frames += 1
            speed = np.linalg.norm(t.kf.x[3:])
            t.is_static = speed < 0.12 # Threshold for static vs moving status
            limit = self.max_skip_static if t.is_static else self.max_skip_dynamic
            
            if t.skipped_frames <= limit: active.append(t)
            else: self.lost_tracks.append(t)
        self.tracks = active

        # Re-Identification and Initialization
        for j, det in enumerate(detections):
            if j not in assigned_dets:
                found_reid = False
                # Try to resume a recently lost track near this detection
                for lt in self.lost_tracks:
                    if np.linalg.norm(lt.last_seen_pos - det['centroid']) < 1.2:
                        lt.kf.x[:3] = det['centroid'].reshape(3, 1)
                        lt.skipped_frames = 0
                        self.tracks.append(lt)
                        self.lost_tracks.remove(lt)
                        found_reid = True
                        break
                if not found_reid:
                    self.tracks.append(Track(self.next_id, det['centroid'], det['features']))
                    self.next_id += 1

        # Store confirmed detections
        curr_res = []
        for t in self.tracks:
            if t.skipped_frames == 0 and t.hits >= self.min_hits:
                pos = t.kf.x[:3].flatten().tolist()
                speed = float(np.linalg.norm(t.kf.x[3:]))
                curr_res.append({
                    "id": int(t.id), 
                    "position": [round(p, 3) for p in pos],
                    "speed": round(speed, 3),
                    "status": "STATIC" if t.is_static else "MOVING"
                })
        self.history.append({"frame_id": frame_id, "timestamp_ms": timestamp, "detections": curr_res})

    def finalize_results(self, json_name="tracking_results.json"):
        # Post-process: Filter short tracks to eliminate noise and fragmentation
        all_ids = {}
        for frame in self.history:
            for det in frame['detections']:
                all_ids[det['id']] = all_ids.get(det['id'], 0) + 1
        
        valid_ids = [idx for idx, count in all_ids.items() if count > 30]
        final_history = []
        for frame in self.history:
            clean_dets = [d for d in frame['detections'] if d['id'] in valid_ids]
            if clean_dets:
                frame['detections'] = clean_dets
                final_history.append(frame)
        
        with open(json_name, 'w') as f:
            json.dump(final_history, f, indent=4)
        print(f"✅ Final Refined results saved to {json_name}")

def main():
    tracker = HumanTrackerMOT()
    data_path = "mapAll/" 
    files = sorted([f for f in os.listdir(data_path) if f.endswith('.pcd')],
                   key=lambda x: int(re.search(r'(\d+)ms', x).group(1)))

    print(f"Processing {len(files)} frames...")
    for i, filename in enumerate(files):
        ts = int(re.search(r'(\d+)ms', filename).group(1))
        tracker.update(i, ts, os.path.join(data_path, filename))
        if i % 100 == 0: print(f"Progress: {i}/{len(files)}")

    tracker.finalize_results()

if __name__ == "__main__":
    main()