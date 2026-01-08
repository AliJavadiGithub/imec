import os
import re
import json
import numpy as np
import open3d as o3d
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

# --- تنظیمات سیستمی ---
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

class Track:
    def __init__(self, id, centroid, features):
        self.id = id
        self.kf = self.init_kalman(centroid)
        self.feature_vector = features 
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
        kf.R *= 0.8    # نویز اندازه‌گیری (بیشتر شده برای نرمی مسیر)
        kf.Q = np.eye(6) * 0.01 
        return kf

class HumanTrackerMOT:
    def __init__(self):
        self.tracks = []
        self.next_id = 1
        self.lost_tracks = [] 
        self.max_skip_dynamic = 10 
        self.max_skip_static = 40 
        self.history = []
        self.static_speed_threshold = 0.2

    def is_human_shape(self, features):
        w, l, h = features[0], features[1], features[2]
        return 0.5 < h < 1.5 and 0.4 < w < 1.8

    def update(self, frame_id, timestamp, pcd_file_path):
        detections = []
        valid_data = False
        
        try:
            if os.path.getsize(pcd_file_path) > 100:
                pcd = o3d.io.read_point_cloud(pcd_file_path)
                if not pcd.is_empty():
                    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
                    labels = np.array(pcd.cluster_dbscan(eps=0.65, min_points=25, print_progress=False))
                    
                    if labels.size > 0:
                        for label in np.unique(labels[labels >= 0]):
                            indices = np.where(labels == label)[0]
                            cluster = pcd.select_by_index(indices)
                            bbox = cluster.get_axis_aligned_bounding_box()
                            extent = bbox.get_extent()
                            features = np.array([extent[0], extent[1], extent[2], len(cluster.points)])
                            
                            if self.is_human_shape(features):
                                detections.append({'centroid': cluster.get_center(), 'features': features})
                        valid_data = True
        except: pass

        dt = max((timestamp - (self.history[-1]['timestamp_ms'] if self.history else timestamp-33)) / 1000.0, 0.001)

        for t in self.tracks:
            for i in range(3): t.kf.F[i, i+3] = dt
            t.kf.predict()
            t.age += 1

        assigned_tracks = set()
        assigned_dets = set()
        if valid_data and self.tracks and detections:
            cost_matrix = np.zeros((len(self.tracks), len(detections)))
            for i, track in enumerate(self.tracks):
                for j, det in enumerate(detections):
                    dist = np.linalg.norm(track.kf.x[:3].flatten() - det['centroid'])
                    cost_matrix[i, j] = dist

            rows, cols = linear_sum_assignment(cost_matrix)
            for r, c in zip(rows, cols):
                if cost_matrix[r, c] < 0.8:
                    self.tracks[r].kf.update(detections[c]['centroid'])
                    self.tracks[r].last_seen_pos = detections[c]['centroid']
                    self.tracks[r].skipped_frames = 0
                    assigned_tracks.add(r)
                    assigned_dets.add(c)

        active_tracks = []
        for i, t in enumerate(self.tracks):
            if i not in assigned_tracks: t.skipped_frames += 1
            
            # آپدیت وضعیت ایستا/متحرک بر اساس سرعت تخمینی کالمن
            current_vel = np.linalg.norm(t.kf.x[3:])
            t.is_static = current_vel < self.static_speed_threshold
            
            limit = self.max_skip_static if t.is_static else self.max_skip_dynamic
            if t.skipped_frames < limit:
                active_tracks.append(t)
            else: self.lost_tracks.append(t)
        self.tracks = active_tracks
        
        for j, det in enumerate(detections):
            if j not in assigned_dets:
                found_reid = False
                for lt in self.lost_tracks:
                    if np.linalg.norm(lt.last_seen_pos - det['centroid']) < 1.0:
                        lt.kf.x[:3] = det['centroid'].reshape(3, 1)
                        lt.skipped_frames = 0
                        self.tracks.append(lt)
                        self.lost_tracks.remove(lt)
                        found_reid = True
                        break
                if not found_reid:
                    self.tracks.append(Track(self.next_id, det['centroid'], det['features']))
                    self.next_id += 1

        # ذخیره نتایج با فیلد Speed
        curr_res = []
        for t in self.tracks:
            if t.skipped_frames == 0:
                pos = t.kf.x[:3].flatten().tolist()
                vel = t.kf.x[3:].flatten().tolist()
                speed = float(np.linalg.norm(vel))
                curr_res.append({
                    "id": int(t.id), 
                    "position": [round(p, 3) for p in pos],
                    "speed": round(speed, 3), # فیلد مورد نیاز پلاتر
                    "status": "STATIC" if t.is_static else "MOVING"
                })
        
        self.history.append({"frame_id": frame_id, "timestamp_ms": timestamp, "detections": curr_res})

    def finalize_results(self, json_name="tracking_results.json"):
        print("\nFinalizing and filtering tracks...")
        all_tracks_data = {}
        for frame in self.history:
            for det in frame['detections']:
                idx = det['id']
                if idx not in all_tracks_data: all_tracks_data[idx] = []
                all_tracks_data[idx].append(det['position'])

        valid_ids = []
        for idx, positions in all_tracks_data.items():
            pos_arr = np.array(positions)
            total_dist = np.linalg.norm(pos_arr[-1] - pos_arr[0])
            # فیلتر: حداقل ۵۰ فریم عمر و ۱.۵ متر جابجایی کل
            if len(positions) > 50 and total_dist > 1.5:
                valid_ids.append(idx)

        final_history = []
        for frame in self.history:
            clean_dets = [d for d in frame['detections'] if d['id'] in valid_ids]
            if clean_dets:
                frame['detections'] = clean_dets
                final_history.append(frame)
        
        with open(json_name, 'w') as f:
            json.dump(final_history, f, indent=4)
        print(f"✅ Cleaned results with Speed data saved to {json_name}")

def main():
    tracker = HumanTrackerMOT()
    data_path = "mapAll/" 
    files = sorted([f for f in os.listdir(data_path) if f.endswith('.pcd')],
                   key=lambda x: int(re.search(r'(\d+)ms', x).group(1)))

    print(f"Processing {len(files)} frames...")
    for i, filename in enumerate(files):
        ts = int(re.search(r'(\d+)ms', filename).group(1))
        tracker.update(i, ts, os.path.join(data_path, filename))
        if i % 100 == 0: print(f"Frame: {i}/{len(files)}", end='\r')

    tracker.finalize_results()

if __name__ == "__main__":
    main()