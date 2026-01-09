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
        kf.P *= 10.0 # Increased uncertainty for better adaptation
        kf.R *= 0.5    
        kf.Q = np.eye(6) * 0.1 
        return kf

    def get_avg_features(self):
        return np.mean(self.feature_history[-15:], axis=0)

class HumanTrackerMOT:
    def __init__(self):
        self.tracks = []
        self.next_id = 1
        self.lost_tracks = [] 
        self.max_skip_dynamic = 20 # Increased to handle temporary occlusion
        self.max_skip_static = 60 
        self.history = []
        self.static_speed_threshold = 0.1
        
        # Matching weights
        self.dist_weight = 0.8
        self.feat_weight = 0.2

    def is_valid_detection(self, features):
        """
        Modified to allow 'Partial Bodies'.
        Features: [Width, Length, Height, Point Count]
        """
        w, l, h, pts = features
        # Lowered height from 0.5m to 0.3m to catch upper bodies/heads
        # Lowered min_points to catch sparse data
        is_human = (0.3 < h < 2.2) and (0.2 < w < 1.5) and (pts > 8)
        return is_human

    def extract_features(self, cluster):
        bbox = cluster.get_axis_aligned_bounding_box()
        extent = bbox.get_extent() 
        return np.array([extent[0], extent[1], extent[2], len(cluster.points)])

    def update(self, frame_id, timestamp, pcd_file_path):
        detections = []
        valid_data = False
        
        try:
            if os.path.exists(pcd_file_path):
                pcd = o3d.io.read_point_cloud(pcd_file_path)
                if not pcd.is_empty():
                    # Optimized Denoising
                    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=15, std_ratio=2.5)
                    
                    # Adaptive DBSCAN: eps 0.5 is usually good for indoors/mid-range
                    labels = np.array(pcd.cluster_dbscan(eps=0.5, min_points=10))
                    
                    if labels.size > 0:
                        for label in np.unique(labels[labels >= 0]):
                            indices = np.where(labels == label)[0]
                            cluster = pcd.select_by_index(indices)
                            features = self.extract_features(cluster)
                            
                            if self.is_valid_detection(features):
                                detections.append({
                                    'centroid': cluster.get_center(), 
                                    'features': features
                                })
                        valid_data = True
        except Exception as e:
            print(f"Error frame {frame_id}: {e}")

        # Predict
        dt = max((timestamp - (self.history[-1]['timestamp_ms'] if self.history else timestamp-33)) / 1000.0, 0.001)
        for t in self.tracks:
            for i in range(3): t.kf.F[i, i+3] = dt
            t.kf.predict()
            t.age += 1

        assigned_tracks = set()
        assigned_dets = set()

        # Association
        if valid_data and self.tracks and detections:
            cost_matrix = np.zeros((len(self.tracks), len(detections)))
            for i, track in enumerate(self.tracks):
                for j, det in enumerate(detections):
                    dist = np.linalg.norm(track.kf.x[:3].flatten() - det['centroid'])
                    # Compare shape, but emphasize height difference less (since it changes)
                    avg_f = track.get_avg_features()
                    feat_dist = np.linalg.norm(avg_f[:2] - det['features'][:2]) # Compare W/L mainly
                    
                    cost_matrix[i, j] = (self.dist_weight * dist) + (self.feat_weight * feat_dist)

            rows, cols = linear_sum_assignment(cost_matrix)
            for r, c in zip(rows, cols):
                if cost_matrix[r, c] < 1.5: # Relaxed threshold
                    self.tracks[r].kf.update(detections[c]['centroid'])
                    self.tracks[r].feature_history.append(detections[c]['features'])
                    self.tracks[r].last_seen_pos = detections[c]['centroid']
                    self.tracks[r].skipped_frames = 0
                    assigned_tracks.add(r)
                    assigned_dets.add(c)

        # Update lists
        active_tracks = []
        for i, t in enumerate(self.tracks):
            if i not in assigned_tracks: t.skipped_frames += 1
            vel = np.linalg.norm(t.kf.x[3:])
            t.is_static = vel < self.static_speed_threshold
            limit = self.max_skip_static if t.is_static else self.max_skip_dynamic
            if t.skipped_frames < limit: active_tracks.append(t)
            else: self.lost_tracks.append(t)
        self.tracks = active_tracks
        
        for j, det in enumerate(detections):
            if j not in assigned_dets:
                found_reid = False
                for lt in self.lost_tracks:
                    if np.linalg.norm(lt.last_seen_pos - det['centroid']) < 2.0:
                        lt.kf.x[:3] = det['centroid'].reshape(3, 1)
                        lt.skipped_frames = 0
                        self.tracks.append(lt)
                        self.lost_tracks.remove(lt)
                        found_reid = True
                        break
                if not found_reid:
                    self.tracks.append(Track(self.next_id, det['centroid'], det['features']))
                    self.next_id += 1

        # Save data
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
        # We only filter out very short-lived tracks (ghosts)
        valid_ids = []
        all_tracks_data = {}
        for frame in self.history:
            for det in frame['detections']:
                idx = det['id']
                if idx not in all_tracks_data: all_tracks_data[idx] = []
                all_tracks_data[idx].append(det['position'])

        for idx, positions in all_tracks_data.items():
            if len(positions) > 15: # Reduced from 50 to 15 to keep partial detections
                valid_ids.append(idx)

        final_history = []
        for frame in self.history:
            clean_dets = [d for d in frame['detections'] if d['id'] in valid_ids]
            if clean_dets:
                frame['detections'] = clean_dets
                final_history.append(frame)
        
        with open(json_name, 'w') as f:
            json.dump(final_history, f, indent=4)
        print(f"✅ Finalized. Saved to {json_name}")

def main():
    tracker = HumanTrackerMOT()
    data_path = "mapAll/" 
    files = sorted([f for f in os.listdir(data_path) if f.endswith('.pcd')],
                   key=lambda x: int(re.search(r'(\d+)ms', x).group(1)))

    print(f"Processing {len(files)} frames (Partial Body Support)...")
    for i, filename in enumerate(files):
        ts = int(re.search(r'(\d+)ms', filename).group(1))
        tracker.update(i, ts, os.path.join(data_path, filename))
        if i % 100 == 0: print(f"Frame: {i}/{len(files)}")

    tracker.finalize_results()

if __name__ == "__main__":
    main()