#!/usr/bin/env python3
"""
Point Cloud Playback Visualization
Displays point cloud sequences with green points on black background at 30Hz
"""

import open3d as o3d
import glob
import re
import time
import os
import sys


def extract_timestamp(filename):
    """Extract timestamp from filename like 'occupied_1234ms.pcd'"""
    match = re.search(r'occupied_(\d+)ms\.pcd', filename)
    if match:
        return int(match.group(1))
    return 0


def get_user_choice():
    """Prompt user to choose between human-only or entire map"""
    print("\n" + "="*60)
    print("Point Cloud Visualization")
    print("="*60)
    print("\nChoose visualization mode:")
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
    # Get user choice
    map_choice = get_user_choice()
    
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    map_dir = os.path.join(current_dir, map_choice)
    
    # Get all PCD files from chosen directory (only .pcd files to avoid venv, .DS_Store, etc)
    all_files = glob.glob(os.path.join(map_dir, "*.pcd"))
    # Filter to only include .pcd files
    pcd_files = sorted([f for f in all_files if f.lower().endswith('.pcd')], key=extract_timestamp)
    
    if not pcd_files:
        print(f"\nError: No point cloud files found in {map_dir}/")
        print("Please ensure the PCD files are in the correct directory.")
        sys.exit(1)
    
    print(f"\nFound {len(pcd_files)} point cloud files")
    print("Loading first point cloud...")
    
    # Load first point cloud to initialize
    pcd = o3d.io.read_point_cloud(pcd_files[0])
    
    if not pcd.has_points():
        print("Error: Failed to load point cloud!")
        sys.exit(1)
    
    print(f"Loaded {len(pcd.points)} points")
    
    # Set point color to green
    num_points = len(pcd.points)
    green_color = [0.0, 1.0, 0.0]  # Bright green color
    pcd.colors = o3d.utility.Vector3dVector([green_color] * num_points)
    
    # Create visualizer with black background
    print("Creating visualizer window...")
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Point Cloud Playback - {map_choice} (30Hz)", 
                      width=1280, height=720)
    vis.add_geometry(pcd)
    
    # Set render options for black background and larger points
    opt = vis.get_render_option()
    opt.background_color = [0.0, 0.0, 0.0]  # Black background
    opt.point_size = 5.0  # Larger point size for better visibility
    
    # Set view control
    view_ctrl = vis.get_view_control()
    view_ctrl.set_zoom(0.6)
    
    # Playback at 30Hz (33.33ms per frame)
    frame_delay = 1.0 / 30.0
    
    print("\n" + "="*60)
    print("Starting playback at 30Hz...")
    print("Close window to exit")
    print("="*60 + "\n")
    
    # Play through all point clouds
    for i, pcd_file in enumerate(pcd_files):
        start_time = time.time()
        
        # Load point cloud
        new_pcd = o3d.io.read_point_cloud(pcd_file)
        
        # Update points
        pcd.points = new_pcd.points
        
        # Set green color for all points
        num_points = len(new_pcd.points)
        pcd.colors = o3d.utility.Vector3dVector([green_color] * num_points)
        
        # Update visualization
        vis.update_geometry(pcd)
        
        # Display progress
        if i % 10 == 0:
            timestamp = extract_timestamp(pcd_file)
            progress = (i + 1) / len(pcd_files) * 100
            print(f"Frame {i+1}/{len(pcd_files)} ({progress:.1f}%) - "
                  f"Timestamp: {timestamp}ms - Points: {num_points}", end='\r')
        
        # Poll events and update
        if not vis.poll_events():
            print("\n\nWindow closed by user")
            break
        vis.update_renderer()
        
        # Maintain 30Hz timing
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
        print(f"\n\nError: {e}")
        sys.exit(1)
