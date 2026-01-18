import open3d as o3d
import json
import glob
import re
import os
import numpy as np
import cv2

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


def extract_timestamp(filename):
    match = re.search(r'(\d+)ms', filename)
    return int(match.group(1)) if match else 0

def get_color(obj_id):
    """Generates a consistent RGB color for a given ID."""
    np.random.seed(obj_id)
    return np.random.uniform(0.4, 1.0, size=3).tolist()

def main():
    # Get user choice
    map_choice = get_user_choice()
    
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    map_dir = os.path.join(current_dir, map_choice)

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

    if not pcd_files:
        print(f"No PCD files found in {map_dir} directory.")
        return

    # Initialize Visualizer
    vis = o3d.visualization.Visualizer()
    # Note: The window might not be exactly 1280x720 depending on OS scaling
    vis.create_window(window_name="Tracking Overlay", width=1280, height=720, visible=True)
    
    pcd = o3d.io.read_point_cloud(pcd_files[0])
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 2.0
    
    video_writer = None
    # Use 'avc1' or 'mp4v' - avc1 is often more compatible with modern players
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 

    current_elements = []
    print(f"Recording tracking results to {video_output}...")

    # We store target dimensions once we capture the first frame
    target_width, target_height = None, None

    for i, pcd_file in enumerate(pcd_files):
        new_pcd_data = o3d.io.read_point_cloud(pcd_file)
        pcd.points = new_pcd_data.points
        pcd.paint_uniform_color([0.2, 0.2, 0.2])
        vis.update_geometry(pcd)

        for element in current_elements:
            vis.remove_geometry(element, reset_bounding_box=False)
        current_elements.clear()

        labels_to_draw = []

        if i in tracking_dict:
            for det in tracking_dict[i]:
                obj_id = det['id']
                pos = det['position']
                speed = det.get('speed', 0.0)
                color = get_color(obj_id)
                
                bbox = o3d.geometry.AxisAlignedBoundingBox(
                    min_bound=(pos[0]-0.4, pos[1]-0.4, pos[2]-0.5),
                    max_bound=(pos[0]+0.4, pos[1]+0.4, pos[2]+1.2)
                )
                bbox.color = color
                vis.add_geometry(bbox, reset_bounding_box=False)
                current_elements.append(bbox)

                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
                sphere.paint_uniform_color(color)
                sphere.translate([pos[0], pos[1], pos[2] + 1.3])
                vis.add_geometry(sphere, reset_bounding_box=False)
                current_elements.append(sphere)

                labels_to_draw.append({
                    'pos': [pos[0], pos[1], pos[2] + 1.5],
                    'text': f"ID:{obj_id} | {speed:.1f}m/s",
                    'color': [int(c*255) for c in color[::-1]] 
                })

        vis.poll_events()
        vis.update_renderer()
        
        # Capture frame
        image_raw = vis.capture_screen_float_buffer(True) # Set to True to wait for sync
        image = (np.asarray(image_raw) * 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # GET ACTUAL DIMENSIONS
        h, w, _ = image.shape

        if video_writer is None:
            target_width, target_height = w, h
            print(f"Detected Capture Resolution: {target_width}x{target_height}")
            video_writer = cv2.VideoWriter(video_output, fourcc, 30.0, (target_width, target_height))

        # Ensure consistency: if OS resizes window mid-stream, force it back to initial size
        if (w, h) != (target_width, target_height):
            image = cv2.resize(image, (target_width, target_height))

        # Camera projection logic for 2D labels
        view_ctl = vis.get_view_control()
        param = view_ctl.convert_to_pinhole_camera_parameters()
        intrinsic = param.intrinsic.intrinsic_matrix
        extrinsic = param.extrinsic

        for label in labels_to_draw:
            pt_world = np.array([label['pos'][0], label['pos'][1], label['pos'][2], 1])
            pt_cam = extrinsic @ pt_world
            if pt_cam[2] > 0: 
                coords = intrinsic @ pt_cam[:3]
                x, y = int(coords[0] / coords[2]), int(coords[1] / coords[2])
                
                # Check bounds using target dimensions
                if 0 <= x < target_width and 0 <= y < target_height:
                    cv2.putText(image, label['text'], (x, y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    cv2.putText(image, label['text'], (x, y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, label['color'], 1)

        video_writer.write(image)
        
        if i % 20 == 0:
            print(f"Processing frame {i}/{len(pcd_files)}...", end='\r')

    if video_writer:
        video_writer.release()
    vis.destroy_window()
    print(f"\nSuccess! Video saved: {video_output}")

if __name__ == "__main__":
    main()