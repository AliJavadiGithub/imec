import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

TOP_K = 5

def visualize_tracking(json_path="tracking_results.json"):
    with open(json_path) as f:
        data = json.load(f)

    tracks = defaultdict(lambda: {"t": [], "x": [], "y": [], "speed": [], "len": 0})

    for frame in data:
        t = frame["timestamp_ms"] / 1000.0
        for d in frame["detections"]:
            tid = d["id"]
            tracks[tid]["t"].append(t)
            tracks[tid]["x"].append(d["position"][0])
            tracks[tid]["y"].append(d["position"][1])
            tracks[tid]["speed"].append(d.get("speed", 0.0))

    # pick top-K longest
    lengths = {tid: len(tr["t"]) for tid, tr in tracks.items()}
    top_ids = sorted(lengths.keys(), key=lambda k: lengths[k], reverse=True)[:TOP_K]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    for tid in top_ids:
        tr = tracks[tid]
        ax1.plot(tr["x"], tr["y"], label=f"ID {tid} (n={len(tr['t'])})")
        ax2.plot(tr["t"], tr["speed"], label=f"ID {tid}")

    ax1.set_title(f"Trajectory (Top-Down) - Top {TOP_K} Longest Tracks")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.grid()
    ax1.legend()

    ax2.set_title("Speed vs Time")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Speed (m/s)")
    ax2.grid()
    ax2.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_tracking()
