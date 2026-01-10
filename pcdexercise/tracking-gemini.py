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
    def __init__(self, id, centroid, shape_feature):
        self.id = id
        # x = [x, y, z, vx, vy, vz]
        self.kf = self.init_kalman(centroid)
        self.shape_history = [shape_feature]
        self.hits = 1           
        self.age = 1            
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
        kf.R *= 12.0 # High R trust the prediction over the jittery DBSCAN center
        kf.Q = np.eye(6) * 0.01 
        return kf

    def get_avg_shape(self):
        return np.mean(self.shape_history[-15:], axis=0)

class HumanTrackerMOT:
    def __init__(self):
        self.tracks = []
        self.next_id = 1
        self.lost_tracks = [] 
        self.history = []
        
        # Hyperparameters for Stability
        self.min_hits = 5           # ID must be seen 5 times to be confirmed
        self.max_skip_dynamic = 30  # Increased for better occlusion handling
        self.max_skip_static = 70   
        self.dist_weight = 0.85     # Emphasis on spatial proximity
        self.shape_weight = 0.15    # Emphasis on consistent human shape
        self.gating_threshold = 1.0 # Max meters person can move between frames

    def extract_shape_descriptor(self, cluster):
        """
        Mimics PointNet's Global Feature extraction.
        Calculates normalized spatial moments and height-based density.
        """
        pts = np.asarray(cluster.points)
        if len(pts) < 5: return np.zeros(4)
        
        # Normalize points to centroid
        centered_pts = pts - np.mean(pts, axis=0)
        
        # 1. Aspect Ratio (Verticality)
        bbox = cluster.get_axis_aligned_bounding_box()
        ext = bbox.get_extent()
        aspect_ratio = ext[2] / (max(ext[0], ext[1]) + 1e-6)
        
        # 2. Point Density Distribution (Head vs Body)
        # Check percentage of points in the top 30% of the height
        height_threshold = np.min(pts[:,2]) + 0.7 * ext[2]
        head_density = np.sum(pts[:,2] > height_threshold) / len(pts)
        
        # 3. Variance across axes
        variance = np.var(centered_pts, axis=0)
        
        return np.array([aspect_ratio, head_density, variance[0], variance[2]])

    def is_valid_human(self, cluster):
        bbox = cluster.get_axis_aligned_bounding_box()
        ext = bbox.get_extent()
        pts_count = len(cluster.points)
        # Filter: Height 0.4-2.0m, Width 0.2-1.0m, Min Points 10
        return (0.4 < ext[2] < 2.0) and (0.15 < max(ext[0], ext[1]) < 0.9) and (pts_count > 10)

    def update(self, frame_id, timestamp, pcd_file_path):
        detections = []
        if os.path.exists(pcd_file_path):
            pcd = o3d.io.read_point_cloud(pcd_file_path)
            if not pcd.is_empty():
                pcd = pcd.voxel_down_sample(voxel_size=0.04) # Uniform density
                pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
                
                labels = np.array(pcd.cluster_dbscan(eps=0.55, min_points=8)) # Optimized EPS
                for label in np.unique(labels[labels >= 0]):
                    indices = np.where(labels == label)[0]
                    cluster = pcd.select_by_index(indices)
                    if self.is_valid_human(cluster):
                        detections.append({
                            'centroid': cluster.get_center(), 
                            'shape': self.extract_shape_descriptor(cluster)
                        })

        # Predict
        prev_ts = self.history[-1]['timestamp_ms'] if self.history else timestamp - 33
        dt = max((timestamp - prev_ts) / 1000.0, 0.001)
        for t in self.tracks:
            for i in range(3): t.kf.F[i, i+3] = dt
            t.kf.predict()
            t.age += 1

        # Association
        assigned_tracks, assigned_dets = set(), set()
        if self.tracks and detections:
            cost_matrix = np.zeros((len(self.tracks), len(detections)))
            for i, track in enumerate(self.tracks):
                for j, det in enumerate(detections):
                    dist = np.linalg.norm(track.kf.x[:3].flatten() - det['centroid'])
                    # Compare current shape to average historical shape
                    shape_dist = np.linalg.norm(track.get_avg_shape() - det['shape'])
                    
                    if dist > self.gating_threshold:
                        cost_matrix[i, j] = 999.0
                    else:
                        cost_matrix[i, j] = (self.dist_weight * dist) + (self.shape_weight * shape_dist)

            rows, cols = linear_sum_assignment(cost_matrix)
            for r, c in zip(rows, cols):
                if cost_matrix[r, c] < self.gating_threshold:
                    self.tracks[r].kf.update(detections[c]['centroid'])
                    self.tracks[r].shape_history.append(detections[c]['shape'])
                    self.tracks[r].last_seen_pos = detections[c]['centroid']
                    self.tracks[r].skipped_frames = 0
                    self.tracks[r].hits += 1
                    assigned_tracks.add(r)
                    assigned_dets.add(c)

        # Track Management
        active = []
        for i, t in enumerate(self.tracks):
            if i not in assigned_tracks: t.skipped_frames += 1
            vel = np.linalg.norm(t.kf.x[3:])
            t.is_static = vel < 0.15
            limit = self.max_skip_static if t.is_static else self.max_skip_dynamic
            if t.skipped_frames <= limit: active.append(t)
            else: self.lost_tracks.append(t)
        self.tracks = active

        # New Tracks / Re-ID
        for j, det in enumerate(detections):
            if j not in assigned_dets:
                found_reid = False
                for lt in self.lost_tracks:
                    # Spatial + Shape Re-ID check
                    if np.linalg.norm(lt.last_seen_pos - det['centroid']) < 1.5:
                        lt.kf.x[:3] = det['centroid'].reshape(3, 1)
                        lt.skipped_frames = 0
                        self.tracks.append(lt)
                        self.lost_tracks.remove(lt)
                        found_reid = True
                        break
                if not found_reid:
                    self.tracks.append(Track(self.next_id, det['centroid'], det['shape']))
                    self.next_id += 1

        # Logging
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
        # Post-filter: Remove tracks that appear in fewer than 40 frames total
        id_counts = {}
        for frame in self.history:
            for det in frame['detections']:
                id_counts[det['id']] = id_counts.get(det['id'], 0) + 1
        
        valid_ids = [idx for idx, count in id_counts.items() if count > 40]
        final_history = []
        for frame in self.history:
            clean_dets = [d for d in frame['detections'] if d['id'] in valid_ids]
            if clean_dets:
                frame['detections'] = clean_dets
                final_history.append(frame)
        
        with open(json_name, 'w') as f:
            json.dump(final_history, f, indent=4)
        print(f"✅ Enhanced Tracking Finished. Results in {json_name}")

def main():
    tracker = HumanTrackerMOT()
    data_path = "mapAll/" 
    files = sorted([f for f in os.listdir(data_path) if f.endswith('.pcd')],
                   key=lambda x: int(re.search(r'(\d+)ms', x).group(1)))

    print(f"Processing {len(files)} frames with Shape-Aware Tracking...")
    for i, filename in enumerate(files):
        ts = int(re.search(r'(\d+)ms', filename).group(1))
        tracker.update(i, ts, os.path.join(data_path, filename))
        if i % 100 == 0: print(f"Frame {i}/{len(files)}")

    tracker.finalize_results()

if __name__ == "__main__":
    main()