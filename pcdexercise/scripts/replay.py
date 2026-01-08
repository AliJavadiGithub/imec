import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import time
import numpy as np
import open3d as o3d

from perception.preprocess import preprocess
from perception.ground import estimate_ground
from perception.cluster import euclidean_clusters
from perception.human_filter import is_human
from perception.tracker import Tracker
from perception.motion import motion_state
from perception.visualize import draw_frame
from perception.export import export_csv, export_json

def load_pcd_sequence(path):
    files = sorted(glob.glob(path))
    for f in files:
        yield o3d.io.read_point_cloud(f)

def remove_ground(pcd):
    normal = estimate_ground(pcd)
    pts = np.asarray(pcd.points)
    # project to ground plane
    dist = pts @ normal
    mask = dist > 0.05
    pcd.points = o3d.utility.Vector3dVector(pts[mask])
    return pcd

def centroids_from_clusters(clusters):
    return [np.mean(c, axis=0) for c in clusters]

def replay(path, dt=0.1, export=True):
    tracker = Tracker()
    seq = load_pcd_sequence(path)

    for pcd in seq:
        pcd = preprocess(pcd)
        pcd = remove_ground(pcd)
        pts = np.asarray(pcd.points)

        if len(pts)==0:
            continue

        clusters = euclidean_clusters(pts)
        humans = [c for c in clusters if is_human(c)]
        centers = centroids_from_clusters(humans)

        tracker.update(centers, dt)

        for t in tracker.tracks:
            t.state = motion_state(t.vel)

        draw_frame(pcd, tracker.tracks)
        time.sleep(dt)

    if export:
        export_csv(tracker.tracks, "trajectories.csv")
        export_json(tracker.tracks, "trajectories.json")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to PCD files (e.g., mapAll/*.pcd)")
    parser.add_argument("--dt", type=float, default=0.1, help="Time between frames")
    parser.add_argument("--export", action="store_true", help="Export trajectories to CSV/JSON")
    args = parser.parse_args()

    replay(args.path, dt=args.dt, export=args.export)
