import open3d as o3d
import json
import glob
import re
import os
import numpy as np
import cv2

def extract_timestamp(filename):
    match = re.search(r'(\d+)ms', filename)
    return int(match.group(1)) if match else 0

def get_color(obj_id):
    """Generates a consistent RGB color for a given ID."""
    np.random.seed(obj_id)
    return np.random.uniform(0.4, 1.0, size=3).tolist()

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    map_dir = os.path.join(current_dir, "mapAll")
    json_path = os.path.join(current_dir, "tracking_results.json")
    video_output = "tracking_playback.mp4"

    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found!")
        return
    
    with open(json_path, 'r') as f:
        tracking_data = json.load(f)
    
    tracking_dict = {item['frame_id']: item['detections'] for item in tracking_data}
    all_files = glob.glob(os.path.join(map_dir, "*.pcd"))
    pcd_files = sorted(all_files, key=extract_timestamp)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Tracking Overlay", width=1280, height=720, visible=True)
    
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
        pcd.paint_uniform_color([0.2, 0.2, 0.2])
        vis.update_geometry(pcd)

        for element in current_elements:
            vis.remove_geometry(element, reset_bounding_box=False)
        current_elements.clear()

        # Store labels to draw later on the 2D image
        labels_to_draw = []

        if i in tracking_dict:
            for det in tracking_dict[i]:
                obj_id = det['id']
                pos = det['position']
                speed = det.get('speed', 0.0)
                color = get_color(obj_id)
                
                # Create 3D Bounding Box
                bbox = o3d.geometry.AxisAlignedBoundingBox(
                    min_bound=(pos[0]-0.4, pos[1]-0.4, pos[2]-0.5),
                    max_bound=(pos[0]+0.4, pos[1]+0.4, pos[2]+1.2)
                )
                bbox.color = color
                vis.add_geometry(bbox, reset_bounding_box=False)
                current_elements.append(bbox)

                # Small sphere indicator
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
                sphere.paint_uniform_color(color)
                sphere.translate([pos[0], pos[1], pos[2] + 1.3])
                vis.add_geometry(sphere, reset_bounding_box=False)
                current_elements.append(sphere)

                # Project 3D point to 2D screen coordinates for labeling
                # We use the position slightly above the box
                labels_to_draw.append({
                    'pos': [pos[0], pos[1], pos[2] + 1.5],
                    'text': f"ID:{obj_id} | {speed:.1f}m/s",
                    'color': [int(c*255) for c in color[::-1]] # RGB to BGR
                })

        vis.poll_events()
        vis.update_renderer()
        
        # Capture screen
        image = vis.capture_screen_float_buffer(False)
        image = (np.asarray(image) * 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw labels using OpenCV
        view_ctl = vis.get_view_control()
        param = view_ctl.convert_to_pinhole_camera_parameters()
        intrinsic = param.intrinsic.intrinsic_matrix
        extrinsic = param.extrinsic

        for label in labels_to_draw:
            # Manual projection from 3D to 2D
            pt_world = np.array([label['pos'][0], label['pos'][1], label['pos'][2], 1])
            pt_cam = extrinsic @ pt_world
            if pt_cam[2] > 0: # Check if point is in front of camera
                coords = intrinsic @ pt_cam[:3]
                x, y = int(coords[0] / coords[2]), int(coords[1] / coords[2])
                
                if 0 <= x < 1280 and 0 <= y < 720:
                    cv2.putText(image, label['text'], (x, y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(image, label['text'], (x, y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, label['color'], 1)

        video_writer.write(image)
        if i % 20 == 0:
            print(f"Processing frame {i}/{len(pcd_files)}...", end='\r')

    video_writer.release()
    vis.destroy_window()
    print(f"\nVideo saved: {video_output}")

if __name__ == "__main__":
    main()