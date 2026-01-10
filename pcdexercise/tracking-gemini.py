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
        # x = [x, y, z, vx, vy, vz]
        self.kf = self.init_kalman(centroid)
        self.feature_history = [features]
        self.hits = 1           # Number of consecutive detections
        self.age = 1            # Total frames active
        self.skipped_frames = 0 
        self.is_static = False
        self.last_seen_pos = centroid

    def init_kalman(self, pos):
        kf = KalmanFilter(dim_x=6, dim_z=3)
        # Constant velocity model
        kf.F = np.eye(6)
        kf.H = np.array([[1, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0]])
        
        kf.x[:3] = pos.reshape(3, 1)
        # Increase P for initial state uncertainty
        kf.P *= 5.0 
        # Increase R to trust the model more than noisy DBSCAN centroids (Reduces Jitter)
        kf.R *= 10.0    
        # Low Q assumes smooth, constant motion
        kf.Q = np.eye(6) * 0.01 
        return kf

    def get_avg_features(self):
        return np.mean(self.feature_history[-10:], axis=0)

class HumanTrackerMOT:
    def __init__(self):
        self.tracks = []
        self.next_id = 1
        self.lost_tracks = [] 
        self.history = []
        
        # Hyperparameters for stability
        self.min_hits = 3           # Frames required to "confirm" a track
        self.max_skip_dynamic = 15  # Recovery window for moving targets
        self.max_skip_static = 45   # Recovery window for targets that stopped
        self.dist_weight = 0.9      # Focus primarily on spatial proximity
        self.feat_weight = 0.1
        self.gating_threshold = 1.2 # Max meters a person can move between frames

    def is_valid_detection(self, features):
        # [Width, Length, Height, Point Count]
        w, l, h, pts = features
        # Filter for human-sized clusters
        return (0.35 < h < 2.0) and (0.2 < w < 1.2) and (pts > 12)

    def extract_features(self, cluster):
        bbox = cluster.get_axis_aligned_bounding_box()
        extent = bbox.get_extent() 
        return np.array([extent[0], extent[1], extent[2], len(cluster.points)])

    def update(self, frame_id, timestamp, pcd_file_path):
        detections = []
        
        # 1. Detection Phase
        if os.path.exists(pcd_file_path):
            pcd = o3d.io.read_point_cloud(pcd_file_path)
            if not pcd.is_empty():
                pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
                labels = np.array(pcd.cluster_dbscan(eps=0.45, min_points=10))
                
                for label in np.unique(labels[labels >= 0]):
                    indices = np.where(labels == label)[0]
                    cluster = pcd.select_by_index(indices)
                    feat = self.extract_features(cluster)
                    if self.is_valid_detection(feat):
                        detections.append({'centroid': cluster.get_center(), 'features': feat})

        # 2. Prediction Phase
        # Calculate dt from filename timestamps
        prev_ts = self.history[-1]['timestamp_ms'] if self.history else timestamp - 33
        dt = max((timestamp - prev_ts) / 1000.0, 0.001)
        
        for t in self.tracks:
            for i in range(3): t.kf.F[i, i+3] = dt
            t.kf.predict()
            t.age += 1

        # 3. Association Phase (Hungarian Algorithm)
        assigned_tracks, assigned_dets = set(), set()
        if self.tracks and detections:
            cost_matrix = np.zeros((len(self.tracks), len(detections)))
            for i, track in enumerate(self.tracks):
                for j, det in enumerate(detections):
                    dist = np.linalg.norm(track.kf.x[:3].flatten() - det['centroid'])
                    # Gating: if too far, make association impossible
                    if dist > self.gating_threshold:
                        cost_matrix[i, j] = 999.0
                    else:
                        feat_dist = np.linalg.norm(track.get_avg_features()[:2] - det['features'][:2])
                        cost_matrix[i, j] = (self.dist_weight * dist) + (self.feat_weight * feat_dist)

            rows, cols = linear_sum_assignment(cost_matrix)
            for r, c in zip(rows, cols):
                if cost_matrix[r, c] < self.gating_threshold:
                    self.tracks[r].kf.update(detections[c]['centroid'])
                    self.tracks[r].feature_history.append(detections[c]['features'])
                    self.tracks[r].last_seen_pos = detections[c]['centroid']
                    self.tracks[r].skipped_frames = 0
                    self.tracks[r].hits += 1
                    assigned_tracks.add(r)
                    assigned_dets.add(c)

        # 4. Track Management
        new_active_tracks = []
        for i, t in enumerate(self.tracks):
            if i not in assigned_tracks:
                t.skipped_frames += 1
                t.hits = 0 # Reset consecutive hits
            
            speed = np.linalg.norm(t.kf.x[3:])
            t.is_static = speed < 0.15
            limit = self.max_skip_static if t.is_static else self.max_skip_dynamic
            
            if t.skipped_frames <= limit:
                new_active_tracks.append(t)
            else:
                self.lost_tracks.append(t)
        self.tracks = new_active_tracks

        # 5. Initialization of New Tracks
        for j, det in enumerate(detections):
            if j not in assigned_dets:
                # Re-ID logic: Check lost tracks first
                found_reid = False
                for lt in self.lost_tracks:
                    if np.linalg.norm(lt.last_seen_pos - det['centroid']) < 1.5:
                        lt.kf.x[:3] = det['centroid'].reshape(3, 1)
                        lt.skipped_frames = 0
                        self.tracks.append(lt)
                        self.lost_tracks.remove(lt)
                        found_reid = True
                        break
                if not found_reid:
                    self.tracks.append(Track(self.next_id, det['centroid'], det['features']))
                    self.next_id += 1

        # 6. Record Results (Only for "Confirmed" tracks to reduce noise)
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
        # Final pass: Remove very short-lived tracks
        all_ids = {}
        for frame in self.history:
            for det in frame['detections']:
                all_ids[det['id']] = all_ids.get(det['id'], 0) + 1
        
        valid_ids = [idx for idx, count in all_ids.items() if count > 20]
        
        final_history = []
        for frame in self.history:
            clean_dets = [d for d in frame['detections'] if d['id'] in valid_ids]
            if clean_dets:
                frame['detections'] = clean_dets
                final_history.append(frame)
        
        with open(json_name, 'w') as f:
            json.dump(final_history, f, indent=4)
        print(f"✅ Refined results saved to {json_name}")

def main():
    tracker = HumanTrackerMOT()
    data_path = "mapAll/" 
    files = sorted([f for f in os.listdir(data_path) if f.endswith('.pcd')],
                   key=lambda x: int(re.search(r'(\d+)ms', x).group(1)))

    for i, filename in enumerate(files):
        ts = int(re.search(r'(\d+)ms', filename).group(1))
        tracker.update(i, ts, os.path.join(data_path, filename))
        if i % 100 == 0: print(f"Processing frame {i}/{len(files)}...")

    tracker.finalize_results()

if __name__ == "__main__":
    main()