import open3d as o3d
import json
import glob
import re
import time
import os
import sys
import numpy as np
import cv2

def extract_timestamp(filename):
    match = re.search(r'(\d+)ms', filename)
    return int(match.group(1)) if match else 0

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    map_dir = os.path.join(current_dir, "mapAll")
    json_path = os.path.join(current_dir, "tracking_output.json")
    video_output = "tracking_playback.mp4"

    # Load tracking data
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found! Run the tracking script first.")
        return

    with open(json_path, 'r') as f:
        tracking_data = json.load(f)

    # Convert to { frame_id: [detections...] }
    tracking_dict = {item['frame_id']: item['detections'] for item in tracking_data}

    # Load PCD frames
    all_files = glob.glob(os.path.join(map_dir, "*.pcd"))
    pcd_files = sorted(all_files, key=extract_timestamp)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Tracking Playback Recorder", width=1280, height=720, visible=True)

    pcd = o3d.io.read_point_cloud(pcd_files[0])
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 2.0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_output, fourcc, 30.0, (1280, 720))

    current_elements = []
    print(f"Recording tracking results to {video_output}...")

    for i, pcd_file in enumerate(pcd_files):
        new_pcd_data = o3d.io.read_point_cloud(pcd_file)
        pcd.points = new_pcd_data.points
        pcd.paint_uniform_color([0.4, 0.4, 0.4])
        vis.update_geometry(pcd)

        # remove previous frame geometry
        for e in current_elements:
            vis.remove_geometry(e, reset_bounding_box=False)
        current_elements.clear()

        # Draw detection overlays
        if i in tracking_dict:
            for det in tracking_dict[i]:
                pos = det['centroid']
                human_id = det['human_id']
                speed = det['speed']

                # simple ID-based color mapping
                np.random.seed(human_id)
                color = np.random.rand(3).tolist()

                bbox = o3d.geometry.AxisAlignedBoundingBox(
                    min_bound=(pos[0]-0.3, pos[1]-0.3, pos[2]-0.8),
                    max_bound=(pos[0]+0.3, pos[1]+0.3, pos[2]+0.8)
                )
                bbox.color = color
                vis.add_geometry(bbox, reset_bounding_box=False)
                current_elements.append(bbox)

                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
                sphere.paint_uniform_color(color)
                sphere.translate([pos[0], pos[1], pos[2] + 1.0])
                vis.add_geometry(sphere, reset_bounding_box=False)
                current_elements.append(sphere)

        vis.poll_events()
        vis.update_renderer()

        image = vis.capture_screen_float_buffer(False)
        image = (np.asarray(image) * 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        video_writer.write(image)

        if i % 50 == 0:
            print(f"Processing frame {i}/{len(pcd_files)}...", end='\r')

    video_writer.release()
    vis.destroy_window()
    print(f"\nSaved tracking video as {video_output}")

if __name__ == "__main__":
    main()
