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
        kf.R *= 10.0 
        kf.Q = np.eye(6) * 0.01 
        return kf

    def get_avg_shape(self):
        return np.mean(self.shape_history[-15:], axis=0)

class VisualTrackingSystem:
    def __init__(self):
        self.tracks = []
        self.next_id = 1
        self.lost_tracks = []
        self.gating_threshold = 1.2
        
        # Initialize Visualizer
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="Human Tracking MOT", width=1280, height=720)
        self.first_frame = True

    def update_and_render(self, pcd_path, frame_id):
        if not os.path.exists(pcd_path): return
        
        pcd = o3d.io.read_point_cloud(pcd_path)
        original_pcd = o3d.geometry.PointCloud(pcd) # Copy for background
        original_pcd.paint_uniform_color([0.2, 0.2, 0.2])
        
        # 1. Geometric Extraction (The "Robust Workflow")
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.15, ransac_n=3, num_iterations=1000)
        objects_pcd = pcd.select_by_index(inliers, invert=True)
        
        # 2. DBSCAN + Dimension Filtering
        labels = np.array(objects_pcd.cluster_dbscan(eps=0.5, min_points=5))
        detections = []
        for label in np.unique(labels[labels >= 0]):
            cluster = objects_pcd.select_by_index(np.where(labels == label)[0])
            bbox = cluster.get_axis_aligned_bounding_box()
            ext = bbox.get_extent()
            
            # Filter: Height 0.7-2.1m, Width < 1.0m
            if (0.7 < ext[2] < 2.1) and (max(ext[0], ext[1]) < 1.0):
                detections.append({'centroid': cluster.get_center(), 'cluster': cluster})

        # 3. Simple Association (Distance-based)
        assigned_tracks = set()
        geometries = [original_pcd]
        
        if self.tracks and detections:
            # For visualization, we'll use a simplified association
            for i, track in enumerate(self.tracks):
                best_dist = self.gating_threshold
                best_det_idx = -1
                for j, det in enumerate(detections):
                    dist = np.linalg.norm(track.kf.x[:3].flatten() - det['centroid'])
                    if dist < best_dist:
                        best_dist = dist
                        best_det_idx = j
                
                if best_det_idx != -1:
                    track.kf.update(detections[best_det_idx]['centroid'])
                    track.skipped_frames = 0
                    assigned_tracks.add(i)
                    
                    # Add Bounding Box + Color Points
                    cluster = detections[best_det_idx]['cluster']
                    cluster.paint_uniform_color([1, 0, 0])
                    bbox = cluster.get_axis_aligned_bounding_box()
                    bbox.color = (0, 1, 0)
                    geometries.append(cluster)
                    geometries.append(bbox)
                    print(f"Frame {frame_id}: Tracking ID {track.id}")

        # 4. Handle New Tracks
        for j, det in enumerate(detections):
            # If not close to any existing track, create new
            is_new = True
            for t in self.tracks:
                if np.linalg.norm(t.kf.x[:3].flatten() - det['centroid']) < self.gating_threshold:
                    is_new = False
                    break
            if is_new:
                self.tracks.append(Track(self.next_id, det['centroid'], np.zeros(4)))
                self.next_id += 1

        # Render Frame
        self.vis.clear_geometries()
        for g in geometries:
            self.vis.add_geometry(g, reset_bounding_box=self.first_frame)
        
        self.first_frame = False
        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self):
        self.vis.destroy_window()

def main():
    system = VisualTrackingSystem()
    data_path = "mapAll/" 
    files = sorted([f for f in os.listdir(data_path) if f.endswith('.pcd')],
                   key=lambda x: int(re.search(r'(\d+)ms', x).group(1)))

    for i, filename in enumerate(files):
        system.update_and_render(os.path.join(data_path, filename), i)
    
    system.close()

if __name__ == "__main__":
    main()