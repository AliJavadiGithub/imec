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
        kf.R *= 10.0 # Trust prediction slightly more due to sparse jitter
        kf.Q = np.eye(6) * 0.01 
        return kf

    def get_avg_shape(self):
        return np.mean(self.shape_history[-15:], axis=0)
    
    def compute_confidence(self, min_hits, tau=10.0, v_max=2.0):
        # 1. Track maturity
        c_hits = min(1.0, self.hits / float(min_hits))

        # 2. Visibility / occlusion penalty
        c_visibility = np.exp(-self.skipped_frames / tau)

        # 3. Motion stability penalty
        speed = np.linalg.norm(self.kf.x[3:])
        c_motion = np.exp(-speed / v_max)

        # Weighted combination
        confidence = (0.5 * c_hits) + (0.3 * c_visibility) + (0.2 * c_motion)

        return float(np.clip(confidence, 0.0, 1.0))


class HumanTrackerMOT:
    def __init__(self):
        self.tracks = []
        self.next_id = 1
        self.lost_tracks = [] 
        self.history = []
        
        # Hyperparameters
        self.min_hits = 3           # Reduced slightly for sparse data
        self.max_skip_dynamic = 20  
        self.max_skip_static = 50   
        self.dist_weight = 0.80     
        self.shape_weight = 0.20    
        self.gating_threshold = 1.2 # Max meters between frames

    def extract_shape_descriptor(self, cluster):
        pts = np.asarray(cluster.points)
        if len(pts) < 5: return np.zeros(4)
        centered_pts = pts - np.mean(pts, axis=0)
        bbox = cluster.get_axis_aligned_bounding_box()
        ext = bbox.get_extent()
        aspect_ratio = ext[2] / (max(ext[0], ext[1]) + 1e-6)
        height_threshold = np.min(pts[:,2]) + 0.7 * ext[2]
        head_density = np.sum(pts[:,2] > height_threshold) / len(pts)
        variance = np.var(centered_pts, axis=0)
        return np.array([aspect_ratio, head_density, variance[0], variance[2]])

    def is_valid_human_geometry(self, cluster):
        """Refined heuristic based on your bounds [6.0, 6.5, 1.75]"""
        bbox = cluster.get_axis_aligned_bounding_box()
        ext = bbox.get_extent() # [width, length, height]
        pts_count = len(cluster.points)
        
        # Human dimensions in meters: 
        # Width/Length usually < 0.8m, Height between 0.8m and 2.1m
        is_human_sized = (0.7 < ext[2] < 2.1) and (max(ext[0], ext[1]) < 1.0)
        return is_human_sized and (pts_count >= 5)

    def update(self, frame_id, timestamp, pcd_file_path):
        detections = []
        if os.path.exists(pcd_file_path):
            pcd = o3d.io.read_point_cloud(pcd_file_path)
            if not pcd.is_empty():
                # --- GEOMETRIC IMPROVEMENT START ---
                # 1. Clean noise
                pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
                
                # 2. Ground Plane Subtraction
                # This prevents the tracker from 'tripping' on floor points
                # CHECK 2: Critical check for RANSAC requirement
                # ransac_n is 3, so we must have at least 3 points
                if len(pcd.points) < 3:
                    # If we don't have enough points for a plane, 
                    # we skip segmentation and treat the whole PCD as objects
                    objects_pcd = pcd
                else:
                    _, inliers = pcd.segment_plane(
                        distance_threshold=0.15,
                        ransac_n=3,
                        num_iterations=1000
                    )
                    objects_pcd = pcd.select_by_index(inliers, invert=True)
                

                # 3. Enhanced Clustering
                labels = np.array(objects_pcd.cluster_dbscan(eps=0.5, min_points=5))
                for label in np.unique(labels[labels >= 0]):
                    indices = np.where(labels == label)[0]
                    cluster = objects_pcd.select_by_index(indices)
                    if self.is_valid_human_geometry(cluster):
                        detections.append({
                            'centroid': cluster.get_center(), 
                            'shape': self.extract_shape_descriptor(cluster)
                        })
                # --- GEOMETRIC IMPROVEMENT END ---

        # Predict
        prev_ts = self.history[-1]['timestamp_ms'] if self.history else timestamp - 33
        dt = max((timestamp - prev_ts) / 1000.0, 0.001)
        for t in self.tracks:
            for i in range(3): t.kf.F[i, i+3] = dt
            t.kf.predict()
            t.age += 1

        # Association (Hungarian Algorithm)
        assigned_tracks, assigned_dets = set(), set()
        if self.tracks and detections:
            cost_matrix = np.zeros((len(self.tracks), len(detections)))
            for i, track in enumerate(self.tracks):
                for j, det in enumerate(detections):
                    dist = np.linalg.norm(track.kf.x[:3].flatten() - det['centroid'])
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

        # Logging Results
        curr_res = []
        for t in self.tracks:
            if t.skipped_frames == 0 and t.hits >= self.min_hits:
                pos = t.kf.x[:3].flatten().tolist()
                vel = t.kf.x[3:].flatten()
                speed = float(np.linalg.norm(vel))
                confidence = t.compute_confidence(self.min_hits)

                curr_res.append({
                    "id": int(t.id),
                    "position": [round(p, 3) for p in pos],
                    "velocity": [round(v, 3) for v in vel],  # 3D velocity [vx, vy, vz]
                    "speed": round(speed, 3),
                    "status": "STATIC" if t.is_static else "MOVING",
                    "confidence": round(confidence, 3)
                })

        self.history.append({"frame_id": frame_id, "timestamp_ms": timestamp, "detections": curr_res})

    def finalize_results(self, json_name="tracking_results.json"):
        # Reduced count threshold for shorter PCD sequences
        id_counts = {}
        for frame in self.history:
            for det in frame['detections']:
                id_counts[det['id']] = id_counts.get(det['id'], 0) + 1
        
        valid_ids = [idx for idx, count in id_counts.items() if count > 10]
        final_history = []
        for frame in self.history:
            clean_dets = [d for d in frame['detections'] if d['id'] in valid_ids]
            if clean_dets:
                frame['detections'] = clean_dets
                final_history.append(frame)
        
        with open(json_name, 'w') as f:
            json.dump(final_history, f, indent=4)
        print(f"✅ Tracking Finished. Results in {json_name}")

def get_user_choice():
    """Prompt user to choose between human-only or entire map"""
    print("\n" + "="*60)
    print("Point Cloud Dataset")
    print("="*60)
    print("\nChoose Dataset:")
    print("  [1] Human Only (mapHumanOnly)")
    print("  [2] Entire Occupancy Map (mapAll)")
    print("")
    
    while True:
        choice = input("Enter your choice (1 or 2): ").strip()
        if choice == "1":
            return "mapHumanOnly"
        elif choice == "2":
            return "mapAll"
        else:
            print("Invalid choice. Please enter 1 or 2.")

def main():
    tracker = HumanTrackerMOT()

    # Get user choice
    map_choice = get_user_choice()
    
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    data_dir = os.path.join(current_dir, map_choice)

    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pcd')],
                   key=lambda x: int(re.search(r'(\d+)ms', x).group(1)))

    for i, filename in enumerate(files):
        ts = int(re.search(r'(\d+)ms', filename).group(1))
        tracker.update(i, ts, os.path.join(data_dir, filename))
        if i % 50 == 0: print(f"Processing Frame {i}/{len(files)}")

    tracker.finalize_results()

if __name__ == "__main__":
    main()