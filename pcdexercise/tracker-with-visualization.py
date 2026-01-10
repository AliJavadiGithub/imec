import os
import re
import cv2
import numpy as np
import open3d as o3d
from filterpy.kalman import KalmanFilter
from collections import deque

# --- System Settings ---
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

class Track:
    def __init__(self, id, centroid, max_path_len=30):
        self.id = id
        self.kf = self.init_kalman(centroid)
        self.skipped_frames = 0 
        self.hits = 1
        # Store last N positions for the trajectory tail
        self.path = deque(maxlen=max_path_len)

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
    def __init__(self, output_video="tracking_trajectories.mp4", fps=15):
        self.tracks = []
        self.next_id = 1
        self.gating_threshold = 1.2
        
        # Open3D Visualizer Setup
        self.width, self.height = 1280, 720
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="Processing Video...", width=self.width, height=self.height)
        
        # Video Writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(output_video, fourcc, fps, (self.width, self.height))
        
        # Minimap Config
        self.map_size = 250 
        self.map_range = 10.0 # meters
        self.first_frame = True

    def create_minimap(self, active_tracks):
        """Creates a BEV map with trajectory paths."""
        minimap = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        cv2.rectangle(minimap, (0,0), (self.map_size, self.map_size), (30, 30, 30), -1)
        
        scale = self.map_size / self.map_range
        center_off = self.map_range / 2

        # Draw 1m Grid
        for i in range(int(self.map_range) + 1):
            line_pos = int(i * scale)
            cv2.line(minimap, (line_pos, 0), (line_pos, self.map_size), (50, 50, 50), 1)
            cv2.line(minimap, (0, line_pos), (self.map_size, line_pos), (50, 50, 50), 1)

        for t in active_tracks:
            # 1. Draw Trajectory Path (Tail)
            points = list(t.path)
            for i in range(1, len(points)):
                pt1 = (int((points[i-1][0] + center_off) * scale), int((points[i-1][1] + center_off) * scale))
                pt2 = (int((points[i][0] + center_off) * scale), int((points[i][1] + center_off) * scale))
                
                # Fading effect: Alpha increases for newer points
                alpha = i / len(points)
                color = (int(0 * alpha), int(255 * alpha), int(255 * alpha)) # Cyan tail
                cv2.line(minimap, pt1, pt2, color, 2)

            # 2. Draw Current Position
            x, y = t.kf.x[0, 0], t.kf.x[1, 0]
            px = int((x + center_off) * scale)
            py = int((y + center_off) * scale)
            
            if 0 <= px < self.map_size and 0 <= py < self.map_size:
                cv2.circle(minimap, (px, py), 5, (0, 0, 255), -1) # Red dot for current head
                cv2.putText(minimap, f"ID:{t.id}", (px+7, py-7), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return minimap

    def update_and_record(self, pcd_path, frame_id):
        if not os.path.exists(pcd_path): return
        
        pcd = o3d.io.read_point_cloud(pcd_path)
        background = o3d.geometry.PointCloud(pcd)
        background.paint_uniform_color([0.1, 0.1, 0.1])
        
        # Robust Geometric Extraction
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=25, std_ratio=1.2)
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.15, ransac_n=3, num_iterations=1000)
        objects_pcd = pcd.select_by_index(inliers, invert=True)
        
        labels = np.array(objects_pcd.cluster_dbscan(eps=0.5, min_points=6))
        detections = []
        for label in np.unique(labels[labels >= 0]):
            cluster = objects_pcd.select_by_index(np.where(labels == label)[0])
            bbox = cluster.get_axis_aligned_bounding_box()
            ext = bbox.get_extent()
            if (0.8 < ext[2] < 2.1) and (max(ext[0], ext[1]) < 0.9):
                detections.append({'centroid': cluster.get_center(), 'cluster': cluster})

        # MOT Logic
        geometries = [background]
        assigned_dets = set()
        active_this_frame = []
        
        for track in self.tracks:
            # Prediction step for path smoothing
            track.kf.predict()
            
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
                assigned_dets.add(best_det_idx)
                
                # Update path history for the tail
                curr_pos = track.kf.x[:2].flatten()
                track.path.append(curr_pos)
                active_this_frame.append(track)
                
                # 3D visualization
                cluster = detections[best_det_idx]['cluster']
                cluster.paint_uniform_color([1, 0, 0])
                bbox = cluster.get_axis_aligned_bounding_box()
                bbox.color = (0, 1, 0)
                geometries.extend([cluster, bbox])
            else:
                track.skipped_frames += 1

        # Add new tracks for unassigned detections
        for j, det in enumerate(detections):
            if j not in assigned_dets:
                new_track = Track(self.next_id, det['centroid'])
                new_track.path.append(det['centroid'][:2])
                self.tracks.append(new_track)
                self.next_id += 1

        # Rendering
        self.vis.clear_geometries()
        for g in geometries:
            self.vis.add_geometry(g, reset_bounding_box=self.first_frame)
        
        if self.first_frame:
            self.vis.get_view_control().set_zoom(0.6)
            self.first_frame = False

        self.vis.poll_events()
        self.vis.update_renderer()
        
        # Image Processing
        img = (np.asarray(self.vis.capture_screen_float_buffer(True)) * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Overlay Minimap
        minimap = self.create_minimap(active_this_frame)
        y_off, x_off = 30, self.width - self.map_size - 30
        roi = img[y_off:y_off+self.map_size, x_off:x_off+self.map_size]
        img[y_off:y_off+self.map_size, x_off:x_off+self.map_size] = cv2.addWeighted(roi, 0.4, minimap, 0.6, 0)
        
        # Frame Text
        cv2.putText(img, f"Frame: {frame_id} | Tracks: {len(active_this_frame)}", 
                    (40, 40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)
        
        self.video_writer.write(img)

    def close(self):
        self.video_writer.release()
        self.vis.destroy_window()
        print("Export Complete.")

def main():
    system = VisualTrackingSystem()
    data_path = "mapAll/" 
    files = sorted([f for f in os.listdir(data_path) if f.endswith('.pcd')],
                   key=lambda x: int(re.search(r'(\d+)ms', x).group(1)))

    for i, filename in enumerate(files):
        system.update_and_record(os.path.join(data_path, filename), i)
    
    system.close()

if __name__ == "__main__":
    main()