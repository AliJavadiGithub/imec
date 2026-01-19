"""
tracking-visualize.py

Plots trajectory and speed over time from tracking_results.json.
"""

import json
import matplotlib.pyplot as plt


def visualize_tracking(json_path="tracking_results.json"):
    with open(json_path) as f:
        data = json.load(f)

    tracks = {}

    for frame in data:
        t = frame["timestamp_ms"] / 1000.0
        for d in frame["detections"]:
            tracks.setdefault(d["id"], {
                "t": [], "x": [], "y": [], "speed": []
            })
            tracks[d["id"]]["t"].append(t)
            tracks[d["id"]]["x"].append(d["position"][0])
            tracks[d["id"]]["y"].append(d["position"][1])
            tracks[d["id"]]["speed"].append(d["speed"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    for tid, tr in tracks.items():
        ax1.plot(tr["x"], tr["y"], label=f"ID {tid}")
        ax2.plot(tr["t"], tr["speed"], label=f"ID {tid}")

    ax1.set_title("Trajectory (Top-Down)")
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

    # --- added code to save plots as separate PNG files ---
    fig_traj, ax_traj = plt.subplots(figsize=(10, 6))
    for tid, tr in tracks.items():
        ax_traj.plot(tr["x"], tr["y"], label=f"ID {tid}")
    ax_traj.set_title("Trajectory (Top-Down)")
    ax_traj.set_xlabel("X (m)")
    ax_traj.set_ylabel("Y (m)")
    ax_traj.grid()
    ax_traj.legend()
    fig_traj.tight_layout()
    fig_traj.savefig("trajectory.png")
    plt.close(fig_traj)

    fig_speed, ax_speed = plt.subplots(figsize=(10, 6))
    for tid, tr in tracks.items():
        ax_speed.plot(tr["t"], tr["speed"], label=f"ID {tid}")
    ax_speed.set_title("Speed vs Time")
    ax_speed.set_xlabel("Time (s)")
    ax_speed.set_ylabel("Speed (m/s)")
    ax_speed.grid()
    ax_speed.legend()
    fig_speed.tight_layout()
    fig_speed.savefig("speed.png")
    plt.close(fig_speed)
    # --- end added code ---

    plt.show()


if __name__ == "__main__":
    visualize_tracking()
