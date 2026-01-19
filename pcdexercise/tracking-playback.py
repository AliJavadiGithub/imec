"""
tracking-playback.py
--------------------
Visualizes human tracking results over point cloud sequences and
records an annotated video.

Features:
- Open3D real-time playback
- Per-ID bounding boxes & markers
- 2D screen-projected labels (ID + speed)
- Stable video recording (auto resolution locking)

Input:
- tracking_results.json
- mapHumanOnly/ or mapAll/

Output:
- tracking_playback.mp4
"""

import os
import re
import glob
import json
import cv2
import numpy as np
import open3d as o3d


# =========================
# Utility
# =========================

def extract_timestamp_ms(filename: str) -> int:
    match = re.search(r"(\d+)ms", filename)
    return int(match.group(1)) if match else 0


def get_color_for_id(track_id: int):
    """Deterministic color generation per ID."""
    np.random.seed(track_id)
    return np.random.uniform(0.4, 1.0, size=3).tolist()


def get_user_choice():
    print("\n" + "=" * 60)
    print("Point Cloud Tracking Playback")
    print("=" * 60)
    print("\nChoose visualization mode:")
    print("  [1] Human Only (mapHumanOnly)")
    print("  [2] Entire Map (mapAll)")
    while True:
        choice = input("Enter your choice (1 or 2): ").strip()
        if choice == "1":
            return "mapHumanOnly"
        elif choice == "2":
            return "mapAll"
        print("Invalid input. Please enter 1 or 2.")


# =========================
# Visualization Helpers
# =========================

def create_bbox(center, color):
    bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=(center[0] - 0.4, center[1] - 0.4, center[2] - 0.5),
        max_bound=(center[0] + 0.4, center[1] + 0.4, center[2] + 1.2)
    )
    bbox.color = color
    return bbox


def create_marker(center, color):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
    sphere.paint_uniform_color(color)
    sphere.translate([center[0], center[1], center[2] + 1.3])
    return sphere


def project_to_screen(point_3d, intrinsic, extrinsic):
    """Project 3D world coordinate to 2D pixel coordinate."""
    pt = np.append(point_3d, 1.0)
    cam = extrinsic @ pt
    if cam[2] <= 0:
        return None

    pix = intrinsic @ cam[:3]
    return int(pix[0] / pix[2]), int(pix[1] / pix[2])


# =========================
# Main Playback
# =========================

def main():
    dataset = get_user_choice()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    map_dir = os.path.join(base_dir, dataset)

    json_path = os.path.join(base_dir, "tracking_results.json")
    video_path = "tracking_playback.mp4"

    if not os.path.exists(json_path):
        print(f"❌ tracking_results.json not found")
        return

    with open(json_path, "r") as f:
        tracking_data = json.load(f)

    tracking_by_frame = {
        frame["frame_id"]: frame["detections"]
        for frame in tracking_data
    }

    pcd_files = sorted(
        glob.glob(os.path.join(map_dir, "*.pcd")),
        key=extract_timestamp_ms
    )

    if not pcd_files:
        print("❌ No PCD files found")
        return

    # --- Open3D Visualizer ---
    vis = o3d.visualization.Visualizer()
    vis.create_window("Tracking Playback", width=1280, height=720)

    pcd = o3d.io.read_point_cloud(pcd_files[0])
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 2.0

    video_writer = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    target_w, target_h = None, None

    overlays = []
    print(f"🎥 Recording to {video_path}")

    # --- Frame Loop ---
    for frame_idx, pcd_path in enumerate(pcd_files):
        try:
            new_pcd = o3d.io.read_point_cloud(pcd_path)
        except Exception:
            continue

        pcd.points = new_pcd.points
        pcd.paint_uniform_color([0.2, 0.2, 0.2])
        vis.update_geometry(pcd)

        # Remove old overlays
        for obj in overlays:
            vis.remove_geometry(obj, reset_bounding_box=False)
        overlays.clear()

        labels = []

        # Draw tracked humans
        if frame_idx in tracking_by_frame:
            for det in tracking_by_frame[frame_idx]:
                pos = det["position"]
                track_id = det["id"]
                speed = det.get("speed", 0.0)

                color = get_color_for_id(track_id)

                bbox = create_bbox(pos, color)
                marker = create_marker(pos, color)

                vis.add_geometry(bbox, reset_bounding_box=False)
                vis.add_geometry(marker, reset_bounding_box=False)

                overlays.extend([bbox, marker])

                labels.append({
                    "pos": [pos[0], pos[1], pos[2] + 1.5],
                    "text": f"ID:{track_id} | {speed:.1f} m/s",
                    "color": [int(c * 255) for c in color[::-1]]
                })

        if not vis.poll_events():
            print("❌ Window closed")
            break
        vis.update_renderer()

        # --- Capture Frame ---
        image = np.asarray(vis.capture_screen_float_buffer(True))
        image = (image * 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        h, w = image.shape[:2]

        if video_writer is None:
            target_w, target_h = w, h
            video_writer = cv2.VideoWriter(
                video_path, fourcc, 30.0, (target_w, target_h)
            )
            print(f"📐 Video resolution locked: {target_w}x{target_h}")

        if (w, h) != (target_w, target_h):
            image = cv2.resize(image, (target_w, target_h))

        # --- Project labels ---
        vc = vis.get_view_control()
        cam = vc.convert_to_pinhole_camera_parameters()
        K = cam.intrinsic.intrinsic_matrix
        T = cam.extrinsic

        for label in labels:
            pt2d = project_to_screen(label["pos"], K, T)
            if pt2d:
                x, y = pt2d
                if 0 <= x < target_w and 0 <= y < target_h:
                    cv2.putText(image, label["text"], (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    cv2.putText(image, label["text"], (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, label["color"], 1)

        video_writer.write(image)

        if frame_idx % 20 == 0:
            print(f"Processing frame {frame_idx}/{len(pcd_files)}", end="\r")

    if video_writer:
        video_writer.release()

    vis.destroy_window()
    print(f"\n✅ Playback complete. Video saved as {video_path}")


if __name__ == "__main__":
    main()
