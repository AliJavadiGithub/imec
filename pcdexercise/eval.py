import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.spatial.distance import euclidean

def load_results(filepath):
    with open(filepath) as f:
        return json.load(f)

def compute_metrics(results):
    metrics = {
        'id_switches': 0,
        'track_completeness': defaultdict(list),
        'velocity_stats': defaultdict(list),
        'trajectory_smoothness': defaultdict(list),
        'frame_gaps': [],
        'predicted_frames': 0,
        'total_frames': len(results),
        'unique_tracks': set(),
    }

    # Collect all unique track IDs
    for res in results:
        for d in res['detections']:
            metrics['unique_tracks'].add(d['human_id'])

    # Track positions per ID
    track_positions = defaultdict(list)
    track_timestamps = defaultdict(list)
    track_velocities = defaultdict(list)

    # First pass: collect positions and timestamps
    for res in results:
        ts = res['timestamp_ms']
        is_predicted = res.get('is_predicted', False)
        if is_predicted:
            metrics['predicted_frames'] += 1

        for d in res['detections']:
            track_id = d['human_id']
            track_positions[track_id].append(d['centroid'])
            track_timestamps[track_id].append(ts)
            track_velocities[track_id].append(d['speed'])

    # Compute metrics per track
    for track_id in metrics['unique_tracks']:
        positions = np.array(track_positions[track_id])
        timestamps = np.array(track_timestamps[track_id])
        velocities = np.array(track_velocities[track_id])

        # Track completeness (percentage of frames with this track)
        completeness = len(timestamps) / metrics['total_frames']
        metrics['track_completeness'][track_id] = completeness

        # Velocity statistics
        if len(velocities) > 0:
            metrics['velocity_stats'][track_id] = {
                'mean': np.mean(velocities),
                'std': np.std(velocities),
                'max': np.max(velocities),
            }

        # Trajectory smoothness (mean acceleration)
        if len(positions) > 2:
            diffs = np.diff(positions, axis=0)
            accelerations = np.diff(diffs, axis=0)
            accel_magnitudes = np.linalg.norm(accelerations, axis=1)
            metrics['trajectory_smoothness'][track_id] = np.mean(accel_magnitudes)

    # Second pass: detect ID switches
    prev_frame = {}
    for res in results:
        current_frame = {d['human_id']: d['centroid'] for d in res['detections']}

        for track_id, centroid in current_frame.items():
            if track_id in prev_frame:
                prev_centroid = prev_frame[track_id]
                dist = euclidean(centroid, prev_centroid)

                # If same ID but large distance jump, count as potential switch
                if dist > 0.5:  # Threshold for "same object"
                    metrics['id_switches'] += 1

        prev_frame = current_frame

    # Detect frame gaps
    timestamps = [res['timestamp_ms'] for res in results]
    if len(timestamps) > 1:
        gaps = np.diff(timestamps)
        metrics['frame_gaps'] = gaps.tolist()

    return metrics

def plot_metrics(metrics, results):
    # Plot trajectory smoothness
    track_ids = list(metrics['trajectory_smoothness'].keys())
    smoothness = [metrics['trajectory_smoothness'][tid] for tid in track_ids]

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.bar(track_ids, smoothness)
    plt.title('Trajectory Smoothness (mean acceleration)')
    plt.xlabel('Track ID')
    plt.ylabel('Acceleration')

    # Plot velocity statistics
    plt.subplot(2, 2, 2)
    for tid in metrics['velocity_stats']:
        plt.errorbar(tid, metrics['velocity_stats'][tid]['mean'],
                    yerr=metrics['velocity_stats'][tid]['std'],
                    fmt='o', label=f'Track {tid}')
    plt.title('Velocity Statistics')
    plt.xlabel('Track ID')
    plt.ylabel('Speed (m/s)')
    plt.legend()

    # Plot track completeness
    plt.subplot(2, 2, 3)
    completeness = [metrics['track_completeness'][tid] for tid in track_ids]
    plt.bar(track_ids, completeness)
    plt.title('Track Completeness')
    plt.xlabel('Track ID')
    plt.ylabel('Fraction of frames')

    # Plot frame gaps
    plt.subplot(2, 2, 4)
    if metrics['frame_gaps']:
        plt.hist(metrics['frame_gaps'], bins=20)
        plt.title('Frame Time Gaps')
        plt.xlabel('Time gap (ms)')
        plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

    # Plot trajectories in 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for track_id in metrics['unique_tracks']:
        positions = []
        for res in results:
            for d in res['detections']:
                if d['human_id'] == track_id:
                    positions.append(d['centroid'])

        if positions:
            xs, ys, zs = zip(*positions)
            ax.plot(xs, ys, zs, label=f'Track {track_id}')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Trajectories')
    ax.legend()
    plt.show()

def print_summary(metrics):
    print("\n=== Tracking Performance Summary ===")
    print(f"Total frames: {metrics['total_frames']}")
    print(f"Predicted frames: {metrics['predicted_frames']} ({100*metrics['predicted_frames']/metrics['total_frames']:.1f}%)")
    print(f"Unique tracks: {len(metrics['unique_tracks'])}")
    print(f"ID switches detected: {metrics['id_switches']}")

    print("\n=== Per-Track Metrics ===")
    for tid in metrics['unique_tracks']:
        print(f"\nTrack {tid}:")
        print(f"  Completeness: {metrics['track_completeness'][tid]:.2f}")
        if tid in metrics['velocity_stats']:
            vs = metrics['velocity_stats'][tid]
            print(f"  Velocity: {vs['mean']:.3f} ± {vs['std']:.3f} m/s")
        if tid in metrics['trajectory_smoothness']:
            print(f"  Smoothness: {metrics['trajectory_smoothness'][tid]:.4f} m/s²")

    if metrics['frame_gaps']:
        print(f"\nFrame gaps (ms): mean={np.mean(metrics['frame_gaps']):.1f}, max={np.max(metrics['frame_gaps']):.1f}")

# Main analysis
results = load_results('tracking_results.json')
metrics = compute_metrics(results)
print_summary(metrics)
plot_metrics(metrics, results)
