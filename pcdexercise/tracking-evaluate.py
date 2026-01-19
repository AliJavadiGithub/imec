import json
import numpy as np

JSON_PATH = "tracking_results.json"
MAX_HUMAN_SPEED = 3.0  # m/s
MAX_JUMP_DIST = 1.5   # meters (ID consistency proxy)

def load_data(path):
    with open(path, 'r') as f:
        return json.load(f)

def track_completeness(data):
    """Fraction of frames with at least one detection"""
    detected = sum(1 for f in data if len(f['detections']) > 0)
    return detected / len(data) if data else 0.0

def extract_dominant_track(data):
    """
    Assumes single human.
    If multiple detections exist, choose the one closest to previous position.
    """
    positions = []
    times = []

    last_pos = None

    for frame in data:
        if not frame['detections']:
            continue

        ts = frame['timestamp_ms'] / 1000.0
        dets = frame['detections']

        if last_pos is None or len(dets) == 1:
            pos = np.array(dets[0]['position'])
        else:
            # Choose closest detection
            pos = min(
                (np.array(d['position']) for d in dets),
                key=lambda p: np.linalg.norm(p - last_pos)
            )

        positions.append(pos)
        times.append(ts)
        last_pos = pos

    return np.array(positions), np.array(times)

def compute_speeds(positions, times):
    if len(positions) < 2:
        return np.array([])
    dt = np.diff(times)
    dp = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    return dp / dt

def id_consistency_proxy(positions):
    """
    Measures trajectory continuity.
    Large jumps indicate implicit ID switches.
    """
    if len(positions) < 2:
        return 1.0
    jumps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    stable = jumps < MAX_JUMP_DIST
    return np.mean(stable)

def velocity_smoothness(speeds, times):
    if len(speeds) < 2:
        return 0.0
    dv = np.diff(speeds)
    dt = np.diff(times[1:])
    jerk = np.abs(dv / dt)
    return float(np.mean(jerk))

def velocity_plausibility(speeds):
    if len(speeds) == 0:
        return 0.0
    return float(np.mean(speeds <= MAX_HUMAN_SPEED))

def main():
    data = load_data(JSON_PATH)
    positions, times = extract_dominant_track(data)
    speeds = compute_speeds(positions, times)

    print("\n📊 Tracking Evaluation Metrics (No GT, No IDs)")
    print("---------------------------------------------")
    print(f"Track Completeness        : {track_completeness(data):.3f}")
    print(f"ID Consistency (Proxy)    : {id_consistency_proxy(positions):.3f}")
    print(f"Velocity Smoothness       : {velocity_smoothness(speeds, times):.3f} m/s²")
    print(f"Velocity Plausibility     : {velocity_plausibility(speeds):.3f}")
    print("---------------------------------------------\n")

if __name__ == "__main__":
    main()
