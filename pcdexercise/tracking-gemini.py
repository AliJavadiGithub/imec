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
        self.kf = self.init_kalman(centroid)
        # Store initial features for Re-ID and matching
        self.feature_history = [features] 
        self.age = 0            
        self.skipped_frames = 0 
        self.is_static = False
        self.last_seen_pos = centroid

    def init_kalman(self, pos):
        kf = KalmanFilter(dim_x=6, dim_z=3)
        kf.F = np.eye(6) 
        kf.H = np.array([[1, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0]])
        kf.x[:3] = pos.reshape(3, 1)
        kf.P *= 5.0
        kf.R *= 0.5    
        kf.Q = np.eye(6) * 0.05 
        return kf

    def get_avg_features(self):
        # Returns the moving average of the person's shape features
        return np.mean(self.feature_history[-10:], axis=0)

class HumanTrackerMOT:
    def __init__(self):
        self.tracks = []
        self.next_id = 1
        self.lost_tracks = [] 
        self.max_skip_dynamic = 15 
        self.max_skip_static = 50 
        self.history = []
        self.static_speed_threshold = 0.15
        
        # Weights for the matching cost (Distance vs Shape)
        self.dist_weight = 0.7
        self.feat_weight = 0.3

    def is_human_shape(self, features):
        w, l, h, pts = features
        # Basic human dimensions: 0.5m < height < 2.0m
        return 0.5 < h < 2.0 and 0.3 < w < 1.2

    def extract_features(self, cluster):
        """Extracts shape-based fingerprints for the human cluster."""
        bbox = cluster.get_axis_aligned_bounding_box()
        extent = bbox.get_extent() # [width, length, height]
        num_points = len(cluster.points)
        # Feature vector: [Width, Length, Height, Point Density]
        return np.array([extent[0], extent[1], extent[2], num_points])

    def update(self, frame_id, timestamp, pcd_file_path):
        detections = []
        valid_data = False
        
        try:
            if os.path.exists(pcd_file_path) and os.path.getsize(pcd_file_path) > 100:
                pcd = o3d.io.read_point_cloud(pcd_file_path)
                if not pcd.is_empty():
                    # Denoising
                    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
                    # Segmentation
                    labels = np.array(pcd.cluster_dbscan(eps=0.65, min_points=15))
                    
                    if labels.size > 0:
                        for label in np.unique(labels[labels >= 0]):
                            indices = np.where(labels == label)[0]
                            cluster = pcd.select_by_index(indices)
                            features = self.extract_features(cluster)
                            
                            if self.is_human_shape(features):
                                detections.append({
                                    'centroid': cluster.get_center(), 
                                    'features': features
                                })
                        valid_data = True
        except Exception as e:
            print(f"Error processing frame {frame_id}: {e}")

        # Prediction Phase
        dt = max((timestamp - (self.history[-1]['timestamp_ms'] if self.history else timestamp-33)) / 1000.0, 0.001)
        for t in self.tracks:
            for i in range(3): t.kf.F[i, i+3] = dt
            t.kf.predict()
            t.age += 1

        assigned_tracks = set()
        assigned_dets = set()

        # Data Association Phase (Fusing Distance + Shape Features)
        if valid_data and self.tracks and detections:
            cost_matrix = np.zeros((len(self.tracks), len(detections)))
            for i, track in enumerate(self.tracks):
                for j, det in enumerate(detections):
                    # 1. Spatial distance cost
                    dist = np.linalg.norm(track.kf.x[:3].flatten() - det['centroid'])
                    
                    # 2. Feature similarity cost (Euclidean distance in feature space)
                    avg_feat = track.get_avg_features()
                    feat_dist = np.linalg.norm(avg_feat - det['features'])
                    
                    # 3. Combined Cost
                    cost_matrix[i, j] = (self.dist_weight * dist) + (self.feat_weight * feat_dist)

            rows, cols = linear_sum_assignment(cost_matrix)
            for r, c in zip(rows, cols):
                # Threshold for a valid match (Adjusted for fused cost)
                if cost_matrix[r, c] < 1.2: 
                    self.tracks[r].kf.update(detections[c]['centroid'])
                    self.tracks[r].feature_history.append(detections[c]['features'])
                    self.tracks[r].last_seen_pos = detections[c]['centroid']
                    self.tracks[r].skipped_frames = 0
                    assigned_tracks.add(r)
                    assigned_dets.add(c)

        # Management Phase (Cleanup and Re-ID)
        active_tracks = []
        for i, t in enumerate(self.tracks):
            if i not in assigned_tracks: t.skipped_frames += 1
            
            vel = np.linalg.norm(t.kf.x[3:])
            t.is_static = vel < self.static_speed_threshold
            
            limit = self.max_skip_static if t.is_static else self.max_skip_dynamic
            if t.skipped_frames < limit:
                active_tracks.append(t)
            else:
                self.lost_tracks.append(t)
                if len(self.lost_tracks) > 20: self.lost_tracks.pop(0)

        self.tracks = active_tracks
        
        # New Track / Re-Identification
        for j, det in enumerate(detections):
            if j not in assigned_dets:
                found_reid = False
                for lt in self.lost_tracks:
                    # Check Re-ID using distance AND feature similarity
                    dist = np.linalg.norm(lt.last_seen_pos - det['centroid'])
                    feat_sim = np.linalg.norm(lt.get_avg_features() - det['features'])
                    
                    if dist < 1.5 and feat_sim < 0.5:
                        lt.kf.x[:3] = det['centroid'].reshape(3, 1)
                        lt.skipped_frames = 0
                        self.tracks.append(lt)
                        self.lost_tracks.remove(lt)
                        found_reid = True
                        break
                
                if not found_reid:
                    self.tracks.append(Track(self.next_id, det['centroid'], det['features']))
                    self.next_id += 1

        # Recording Results
        curr_res = []
        for t in self.tracks:
            if t.skipped_frames == 0:
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
        # Logic remains same but ensures only quality tracks are kept
        valid_ids = []
        all_tracks_data = {}
        for frame in self.history:
            for det in frame['detections']:
                idx = det['id']
                if idx not in all_tracks_data: all_tracks_data[idx] = []
                all_tracks_data[idx].append(det['position'])

        for idx, positions in all_tracks_data.items():
            if len(positions) > 30: # Minimum 30 frames for a valid person
                valid_ids.append(idx)

        final_history = []
        for frame in self.history:
            clean_dets = [d for d in frame['detections'] if d['id'] in valid_ids]
            if clean_dets:
                frame['detections'] = clean_dets
                final_history.append(frame)
        
        with open(json_name, 'w') as f:
            json.dump(final_history, f, indent=4)
        print(f"✅ Enhanced tracking complete. Saved to {json_name}")

def main():
    tracker = HumanTrackerMOT()
    data_path = "mapAll/" 
    if not os.path.exists(data_path):
        print(f"Error: Directory {data_path} not found.")
        return

    files = sorted([f for f in os.listdir(data_path) if f.endswith('.pcd')],
                   key=lambda x: int(re.search(r'(\d+)ms', x).group(1)))

    print(f"Processing {len(files)} frames with Feature Fusion...")
    for i, filename in enumerate(files):
        ts = int(re.search(r'(\d+)ms', filename).group(1))
        tracker.update(i, ts, os.path.join(data_path, filename))
        if i % 50 == 0: print(f"Progress: {i}/{len(files)}")

    tracker.finalize_results()

if __name__ == "__main__":
    main()