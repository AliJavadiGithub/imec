import os
import re
import json
import numpy as np
import open3d as o3d
import cv2
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

# --- System Settings ---
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

class Track:
    def __init__(self, id, centroid):
        self.id = id
        self.kf = self.init_kalman(centroid)
        self.skipped_frames = 0 
        self.hits = 1

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

class VisualTrackingSystem:
    def __init__(self, output_video="tracking_output.mp4", fps=20):
        self.tracks = []
        self.next_id = 1
        self.gating_threshold = 1.2
        
        # Open3D Visualizer Setup
        self.width, self.height = 1280, 720
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="Recording...", width=self.width, height=self.height, visible=True)
        
        # Video Writer Setup
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(output_video, fourcc, fps, (self.width, self.height))
        
        self.first_frame = True

    def update_and_record(self, pcd_path, frame_id):
        if not os.path.exists(pcd_path): return
        
        pcd = o3d.io.read_point_cloud(pcd_path)
        original_pcd = o3d.geometry.PointCloud(pcd)
        original_pcd.paint_uniform_color([0.15, 0.15, 0.15]) # Dark background
        
        # 1. Geometric Extraction
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.15, ransac_n=3, num_iterations=1000)
        objects_pcd = pcd.select_by_index(inliers, invert=True)
        
        # 2. Clustering & Detection
        labels = np.array(objects_pcd.cluster_dbscan(eps=0.5, min_points=5))
        detections = []
        for label in np.unique(labels[labels >= 0]):
            cluster = objects_pcd.select_by_index(np.where(labels == label)[0])
            bbox = cluster.get_axis_aligned_bounding_box()
            ext = bbox.get_extent()
            
            if (0.7 < ext[2] < 2.1) and (max(ext[0], ext[1]) < 1.0):
                detections.append({'centroid': cluster.get_center(), 'cluster': cluster})

        # 3. Association & Rendering Prep
        geometries = [original_pcd]
        assigned_dets = set()
        
        for track in self.tracks:
            best_dist = self.gating_threshold
            best_det_idx = -1
            for j, det in enumerate(detections):
                if j in assigned_dets: continue
                dist = np.linalg.norm(track.kf.x[:3].flatten() - det['centroid'])
                if dist < best_dist:
                    best_dist = dist
                    best_det_idx = j
            
            if best_det_idx != -1:
                track.kf.update(detections[best_det_idx]['centroid'])
                track.skipped_frames = 0
                track.hits += 1
                assigned_dets.add(best_det_idx)
                
                # Add visuals for active tracks
                cluster = detections[best_det_idx]['cluster']
                cluster.paint_uniform_color([1, 0, 0])
                bbox = cluster.get_axis_aligned_bounding_box()
                bbox.color = (0, 1, 0)
                geometries.extend([cluster, bbox])

        # 4. New Track Logic
        for j, det in enumerate(detections):
            if j not in assigned_dets:
                self.tracks.append(Track(self.next_id, det['centroid']))
                self.next_id += 1

        # 5. Render to Window and Capture Frame
        self.vis.clear_geometries()
        for g in geometries:
            self.vis.add_geometry(g, reset_bounding_box=self.first_frame)
        
        # Set a default camera view on first frame
        if self.first_frame:
            ctr = self.vis.get_view_control()
            ctr.set_zoom(0.8)
            self.first_frame = False

        self.vis.poll_events()
        self.vis.update_renderer()
        
        # Capture the image from the visualizer buffer
        image = self.vis.capture_screen_float_buffer(do_render=True)
        image = (np.asarray(image) * 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Add overlay text for Frame ID
        cv2.putText(image, f"Frame: {frame_id}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        self.video_writer.write(image)

    def close(self):
        self.video_writer.release()
        self.vis.destroy_window()
        print("Video saved as tracking_output.mp4")

def main():
    system = VisualTrackingSystem(output_video="human_tracking.mp4", fps=15)
    data_path = "mapAll/" 
    files = sorted([f for f in os.listdir(data_path) if f.endswith('.pcd')],
                   key=lambda x: int(re.search(r'(\d+)ms', x).group(1)))

    print(f"Recording {len(files)} frames...")
    for i, filename in enumerate(files):
        system.update_and_record(os.path.join(data_path, filename), i)
    
    system.close()

if __name__ == "__main__":
    main()