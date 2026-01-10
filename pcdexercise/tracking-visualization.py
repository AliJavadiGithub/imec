import json
import matplotlib.pyplot as plt
import numpy as np

def visualize_tracking_metrics(json_path="tracking_results.json"):
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Organize data by human_id
    tracks = {}
    for frame in data:
        ts = frame['timestamp_ms'] / 1000.0  # Convert to seconds
        for det in frame['detections']:
            h_id = det['id']
            if h_id not in tracks:
                tracks[h_id] = {'t': [], 'x': [], 'y': [], 'speed': []}
            
            tracks[h_id]['t'].append(ts)
            tracks[h_id]['x'].append(det['position'][0])
            tracks[h_id]['y'].append(det['position'][1])
            tracks[h_id]['speed'].append(det['speed'])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # 1. Trajectory Plot (X-Y Plane)
    for h_id, val in tracks.items():
        ax1.plot(val['x'], val['y'], marker='o', markersize=2, label=f'Human ID {h_id}')
        # Mark start and end
        ax1.scatter(val['x'][0], val['y'][0], color='green', s=50, label='Start' if h_id == 1 else "")
        ax1.scatter(val['x'][-1], val['y'][-1], color='red', s=50, label='End' if h_id == 1 else "")

    ax1.set_title("Human Trajectory (Top-Down View)")
    ax1.set_xlabel("X Position (m)")
    ax1.set_ylabel("Y Position (m)")
    ax1.legend()
    ax1.grid(True)

    # 2. Speed vs Time Plot
    for h_id, val in tracks.items():
        ax2.plot(val['t'], val['speed'], label=f'Human ID {h_id} Speed')

    ax2.set_title("Estimated Speed Over Time")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Speed (m/s)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_tracking_metrics()