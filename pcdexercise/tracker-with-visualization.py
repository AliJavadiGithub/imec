import os
import re
import cv2
import numpy as np
import open3d as o3d
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
    def __init__(self, output_video="tracking_with_map.mp4", fps=15):
        self.tracks = []
        self.next_id = 1
        self.gating_threshold = 1.2
        
        # Open3D Visualizer Setup
        self.width, self.height = 1280, 720
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="Processing...", width=self.width, height=self.height)
        
        # Video Writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(output_video, fourcc, fps, (self.width, self.height))
        
        # Minimap Config (based on your bounds: ~6m x ~7m)
        self.map_size = 200 # pixels
        self.map_range = 8.0 # meters covered by the map
        self.first_frame = True

    def create_minimap(self, active_tracks):
        """Creates a 2D Bird's Eye View overlay."""
        # Create a semi-transparent dark background for the map
        minimap = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        cv2.rectangle(minimap, (0,0), (self.map_size, self.map_size), (40, 40, 40), -1)
        
        # Draw grid lines (1 meter intervals)
        scale = self.map_size / self.map_range
        for i in range(int(self.map_range)):
            pos = int(i * scale)
            cv2.line(minimap, (pos, 0), (pos, self.map_size), (60, 60, 60), 1)
            cv2.line(minimap, (0, pos), (self.map_size, pos), (60, 60, 60), 1)

        for t in active_tracks:
            # Get X, Y from Kalman state
            x, y = t.kf.x[0, 0], t.kf.x[1, 0]
            
            # Map 3D coordinates to 2D pixel coordinates (centered)
            px = int((x + self.map_range/2) * scale)
            py = int((y + self.map_range/2) * scale)
            
            if 0 <= px < self.map_size and 0 <= py < self.map_size:
                # Draw person as a bright dot
                cv2.circle(minimap, (px, py), 4, (0, 0, 255), -1)
                cv2.putText(minimap, str(t.id), (px+5, py-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return minimap

    def update_and_record(self, pcd_path, frame_id):
        if not os.path.exists(pcd_path): return
        
        pcd = o3d.io.read_point_cloud(pcd_path)
        original_pcd = o3d.geometry.PointCloud(pcd)
        original_pcd.paint_uniform_color([0.15, 0.15, 0.15])
        
        # 1. Geometric Extraction
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.15, ransac_n=3, num_iterations=1000)
        objects_pcd = pcd.select_by_index(inliers, invert=True)
        
        # 2. Clustering
        labels = np.array(objects_pcd.cluster_dbscan(eps=0.5, min_points=5))
        detections = []
        for label in np.unique(labels[labels >= 0]):
            cluster = objects_pcd.select_by_index(np.where(labels == label)[0])
            bbox = cluster.get_axis_aligned_bounding_box()
            ext = bbox.get_extent()
            if (0.7 < ext[2] < 2.1) and (max(ext[0], ext[1]) < 1.0):
                detections.append({'centroid': cluster.get_center(), 'cluster': cluster})

        # 3. Tracking & 3D Visualization Setup
        geometries = [original_pcd]
        assigned_dets = set()
        active_this_frame = []
        
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
                assigned_dets.add(best_det_idx)
                active_this_frame.append(track)
                
                # 3D Visuals
                cluster = detections[best_det_idx]['cluster']
                cluster.paint_uniform_color([1, 0, 0])
                bbox = cluster.get_axis_aligned_bounding_box()
                bbox.color = (0, 1, 0)
                geometries.extend([cluster, bbox])

        # New Tracks
        for j, det in enumerate(detections):
            if j not in assigned_dets:
                self.tracks.append(Track(self.next_id, det['centroid']))
                self.next_id += 1

        # 4. Render 3D and Capture
        self.vis.clear_geometries()
        for g in geometries:
            self.vis.add_geometry(g, reset_bounding_box=self.first_frame)
        
        if self.first_frame:
            self.vis.get_view_control().set_zoom(0.7)
            self.first_frame = False

        self.vis.poll_events()
        self.vis.update_renderer()
        
        # Convert Open3D to OpenCV Image
        img = (np.asarray(self.vis.capture_screen_float_buffer(True)) * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # 5. Add 2D Minimap Overlay
        minimap = self.create_minimap(active_this_frame)
        # Place map in top-right corner with 20px padding
        y_off, x_off = 20, self.width - self.map_size - 20
        # Create a blended overlay (transparency)
        overlay = img[y_off:y_off+self.map_size, x_off:x_off+self.map_size]
        img[y_off:y_off+self.map_size, x_off:x_off+self.map_size] = cv2.addWeighted(overlay, 0.3, minimap, 0.7, 0)
        
        cv2.putText(img, f"Frame: {frame_id} | Humans: {len(active_this_frame)}", 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        self.video_writer.write(img)

    def close(self):
        self.video_writer.release()
        self.vis.destroy_window()

def main():
    system = VisualTrackingSystem()
    data_path = "mapAll/" 
    files = sorted([f for f in os.listdir(data_path) if f.endswith('.pcd')],
                   key=lambda x: int(re.search(r'(\d+)ms', x).group(1)))

    print(f"Generating video with 2D floor-plan for {len(files)} frames...")
    for i, filename in enumerate(files):
        system.update_and_record(os.path.join(data_path, filename), i)
    
    system.close()
    print("Done! Check 'tracking_with_map.mp4'")

if __name__ == "__main__":
    main()