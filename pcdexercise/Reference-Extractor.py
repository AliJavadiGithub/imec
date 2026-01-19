"""
reference_extractor.py
----------------------
Extracts empirical human statistics from mapHumanOnly dataset.
"""

import os
import re
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


def extract_timestamp_ms(filename: str) -> int:
    match = re.search(r"(\d+)ms", filename)
    if not match:
        raise ValueError(f"Invalid timestamp: {filename}")
    return int(match.group(1))


def extract_metadata():
    path = "mapHumanOnly"
    files = sorted(
        [f for f in os.listdir(path) if f.endswith(".pcd")],
        key=extract_timestamp_ms
    )

    dimensions, point_counts, centroids, timestamps = [], [], [], []

    print(f"Analyzing {len(files)} frames...")

    for f in files:
        full_path = os.path.join(path, f)

        if os.path.getsize(full_path) < 200:
            continue

        try:
            pcd = o3d.io.read_point_cloud(full_path)
        except RuntimeError:
            continue

        if pcd.is_empty() or len(pcd.points) < 10:
            continue

        bbox = pcd.get_axis_aligned_bounding_box()

        dimensions.append(bbox.get_extent())
        point_counts.append(len(pcd.points))
        centroids.append(pcd.get_center())
        timestamps.append(extract_timestamp_ms(f) / 1000.0)

    if not centroids:
        print("No valid data found.")
        return

    dims = np.array(dimensions)
    pts = np.array(point_counts)
    pos = np.array(centroids)
    times = np.array(timestamps)

    d_pos = np.diff(pos, axis=0)
    d_time = np.diff(times).reshape(-1, 1)
    d_time[d_time <= 0] = 0.033

    speeds = np.linalg.norm(d_pos / d_time, axis=1)
    speeds = speeds[speeds < 5.0]

    print("\n" + "=" * 40)
    print("📊 CLEANED HUMAN PROFILE REPORT")
    print("=" * 40)
    print(f"Width/Length: {np.mean(dims[:,0]):.2f} x {np.mean(dims[:,1]):.2f} m")
    print(f"Height:       {np.mean(dims[:,2]):.2f} m")
    print(f"Avg Points:   {int(np.mean(pts))}")
    print(f"Min Reliable: {int(np.percentile(pts, 5))}")
    print(f"Normal Speed: {np.mean(speeds):.2f} m/s")
    print(f"Max Walking:  {np.max(speeds):.2f} m/s")
    print("=" * 40)

    plt.plot(pos[:, 0], pos[:, 1], "g-")
    plt.title("Cleaned Ground Truth Path")
    plt.savefig("ground_truth_path.png")


if __name__ == "__main__":
    extract_metadata()
