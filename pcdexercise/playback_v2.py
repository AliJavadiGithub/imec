#!/usr/bin/env python3
"""
Point Cloud Playback Visualization with Tracking Overlay
Displays point cloud sequence + tracked humans (centroids, IDs, velocity, speed)
Uses tracking_results.json from tracking.py
"""

import open3d as o3d
import glob
import re
import time
import os
import sys
import json
import numpy as np


def extract_timestamp(filename):
    """Extract timestamp from filename like 'occupied_1234ms.pcd'"""
    match = re.search(r'occupied_(\d+)ms\.pcd', filename)
    return int(match.group(1)) if match else 0


def load_tracking_results(json_path="tracking_results.json"):
    """Load tracking data from JSON"""
    if not os.path.exists(json_path):
        print(f"Warning: {json_path} not found. No tracking overlay.")
        return {}
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        # Create lookup: frame_id -> list of detections
        tracking = {}
        for frame in data:
            fid = frame["frame_id"]
            tracking[fid] = frame["detections"]
        print(f"Loaded tracking data for {len(tracking)} frames")
        return tracking
    except Exception as e:
        print(f"Error loading tracking JSON: {e}")
        return {}


def get_user_choice():
    """Prompt user to choose visualization mode"""
    print("\n" + "="*60)
    print("Point Cloud Visualization with Tracking Overlay")
    print("="*60)
    print("\nChoose mode:")
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


def create_sphere(center, radius=0.15, color=[1.0, 0.0, 0.0]):
    """Create small red sphere at centroid"""
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.paint_uniform_color(color)
    sphere.translate(center)
    return sphere


def create_velocity_arrow(start, velocity, scale=0.5, color=[1.0, 0.0, 0.0]):
    """Create small arrow showing velocity direction"""
    end = start + np.array(velocity) * scale
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.02, cone_radius=0.05, cylinder_height=0.3, cone_height=0.1
    )
    arrow.paint_uniform_color(color)
    # Rotate and translate arrow
    direction = end - start
    if np.linalg.norm(direction) > 1e-6:
        direction /= np.linalg.norm(direction)
        arrow.rotate(o3d.geometry.get_rotation_matrix_from_xyz([0, 0, np.arctan2(direction[1], direction[0])]), center=(0,0,0))
        arrow.rotate(o3d.geometry.get_rotation_matrix_from_xyz([np.arccos(direction[2]), 0, 0]), center=(0,0,0))
    arrow.translate(start)
    return arrow


def main():
    # User choice
    map_choice = get_user_choice()
    
    # Directories
    current_dir = os.path.dirname(os.path.abspath(__file__))
    map_dir = os.path.join(current_dir, map_choice)
    
    # Load tracking results
    tracking_data = load_tracking_results("tracking_results.json")
    
    # Get PCD files
    all_files = glob.glob(os.path.join(map_dir, "*.pcd"))
    pcd_files = sorted([f for f in all_files if f.lower().endswith('.pcd')], key=extract_timestamp)
    
    if not pcd_files:
        print(f"\nError: No .pcd files found in {map_dir}/")
        sys.exit(1)
    
    print(f"\nFound {len(pcd_files)} point cloud files")
    
    # Load first point cloud
    pcd = o3d.io.read_point_cloud(pcd_files[0])
    if not pcd.has_points():
        print("Error: Failed to load first point cloud!")
        sys.exit(1)
    
    # Green color for points
    num_points = len(pcd.points)
    pcd.colors = o3d.utility.Vector3dVector([[0.0, 1.0, 0.0]] * num_points)
    
    # Visualizer setup
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=f"Point Cloud Playback + Tracking - {map_choice} (30Hz)",
        width=1280, height=720
    )
    vis.add_geometry(pcd)
    
    # Render options
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.0, 0.0, 0.0])
    opt.point_size = 5.0
    
    # View control
    view_ctrl = vis.get_view_control()
    view_ctrl.set_zoom(0.7)
    
    # Frame delay for 30 Hz
    frame_delay = 1.0 / 30.0
    
    print("\n" + "="*60)
    print("Starting playback with tracking overlay...")
    print("Close window to exit")
    print("="*60 + "\n")
    
    # Keep track of previous geometries to remove
    prev_geometries = []
    
    for i, pcd_file in enumerate(pcd_files):
        start_time = time.time()
        
        # Load new point cloud
        new_pcd = o3d.io.read_point_cloud(pcd_file)
        pcd.points = new_pcd.points
        pcd.colors = o3d.utility.Vector3dVector([[0.0, 1.0, 0.0]] * len(new_pcd.points))
        
        # Get frame ID (1-based)
        frame_id = i + 1
        
        # Clear previous overlay geometries
        for geom in prev_geometries:
            vis.remove_geometry(geom)
        prev_geometries = []
        
        # Add tracking overlay if available
        if frame_id in tracking_data:
            detections = tracking_data[frame_id]
            timestamp = extract_timestamp(pcd_file)
            
            for det in detections:
                cid = det["centroid"]
                vid = det["velocity"]
                speed = det["speed"]
                conf = det["confidence"]
                hid = det["human_id"]
                
                # Centroid sphere (red)
                sphere = create_sphere(cid, radius=0.15)
                vis.add_geometry(sphere)
                prev_geometries.append(sphere)
                
                # Velocity arrow
                if np.linalg.norm(vid) > 0.01:
                    arrow = create_velocity_arrow(cid, vid, scale=0.8)
                    vis.add_geometry(arrow)
                    prev_geometries.append(arrow)
                
                # Optional: add text label (Open3D text is limited, we use console for now)
                print(f"Frame {frame_id:4d} | ID {hid} | Speed {speed:.2f}m/s | Conf {conf:.2f}", end='\r')
        
        # Update geometry & render
        vis.update_geometry(pcd)
        if not vis.poll_events():
            print("\n\nWindow closed by user")
            break
        vis.update_renderer()
        
        # Maintain ~30 Hz
        elapsed = time.time() - start_time
        sleep_time = frame_delay - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
    
    print("\n\nPlayback complete!")
    vis.destroy_window()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPlayback interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError during visualization: {e}")
        sys.exit(1)