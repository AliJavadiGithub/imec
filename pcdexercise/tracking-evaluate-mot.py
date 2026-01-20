import json
import numpy as np
from collections import defaultdict

JSON_PATH = "tracking_results.json"

MAX_HUMAN_SPEED = 3.5        # m/s
MAX_JUMP_DIST = 1.5          # meters per step (proxy for ID stability)
MIN_TRACK_LEN = 10           # frames


def load_data(path):
    with open(path, "r") as f:
        return json.load(f)


def track_completeness(data):
    detected = sum(1 for f in data if len(f["detections"]) > 0)
    return detected / len(data) if data else 0.0


def build_tracks(data):
    tracks = defaultdict(lambda: {"t": [], "pos": [], "speed": [], "vel": []})

    for frame in data:
        t = frame["timestamp_ms"] / 1000.0
        for d in frame["detections"]:
            tid = d["id"]
            tracks[tid]["t"].append(t)
            tracks[tid]["pos"].append(d["position"])
            tracks[tid]["speed"].append(d.get("speed", 0.0))
            tracks[tid]["vel"].append(d.get("velocity", [0.0, 0.0, 0.0]))

    for tid in list(tracks.keys()):
        tracks[tid]["t"] = np.asarray(tracks[tid]["t"], dtype=np.float32)
        tracks[tid]["pos"] = np.asarray(tracks[tid]["pos"], dtype=np.float32)
        tracks[tid]["speed"] = np.asarray(tracks[tid]["speed"], dtype=np.float32)
        tracks[tid]["vel"] = np.asarray(tracks[tid]["vel"], dtype=np.float32)

    return tracks


def per_track_jump_rate(pos):
    if len(pos) < 2:
        return 0.0
    jumps = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    return float(np.mean(jumps > MAX_JUMP_DIST))


def per_track_velocity_smoothness(speed, t):
    """
    Mean |dv/dt| from reported speed.
    Uses aligned diffs:
      dv length = N-1
      dt length = N-1
    """
    n = len(speed)
    if n < 3 or len(t) != n:
        return 0.0

    dv = np.diff(speed)          # (N-1,)
    dt = np.diff(t)              # (N-1,)
    dt = np.where(dt <= 0, 1e-3, dt)

    jerk = np.abs(dv / dt)       # (N-1,)
    return float(np.mean(jerk))


def per_track_velocity_plausibility(speed):
    if len(speed) == 0:
        return 0.0
    return float(np.mean(speed <= MAX_HUMAN_SPEED))


def summarize_mot(tracks):
    lengths = {tid: len(tr["t"]) for tid, tr in tracks.items()}
    valid_ids = [tid for tid, L in lengths.items() if L >= MIN_TRACK_LEN]

    if not valid_ids:
        return None

    jump_rates = []
    smoothness = []
    plaus = []
    durations = []

    for tid in valid_ids:
        tr = tracks[tid]
        jump_rates.append(per_track_jump_rate(tr["pos"]))
        smoothness.append(per_track_velocity_smoothness(tr["speed"], tr["t"]))
        plaus.append(per_track_velocity_plausibility(tr["speed"]))
        durations.append(float(tr["t"][-1] - tr["t"][0]))

    return {
        "num_tracks_total": len(tracks),
        "num_tracks_valid": len(valid_ids),
        "mean_track_len_frames": float(np.mean([lengths[tid] for tid in valid_ids])),
        "median_track_len_frames": float(np.median([lengths[tid] for tid in valid_ids])),
        "mean_track_duration_s": float(np.mean(durations)),
        "mean_jump_rate": float(np.mean(jump_rates)),
        "median_jump_rate": float(np.median(jump_rates)),
        "mean_velocity_smoothness": float(np.mean(smoothness)),
        "median_velocity_smoothness": float(np.median(smoothness)),
        "mean_velocity_plausibility": float(np.mean(plaus)),
        "median_velocity_plausibility": float(np.median(plaus)),
    }


def fragmentation_score(tracks):
    lengths = np.array([len(tr["t"]) for tr in tracks.values()], dtype=np.int32)
    if len(lengths) == 0:
        return 0.0
    return float(np.mean(lengths < MIN_TRACK_LEN))


def main():
    data = load_data(JSON_PATH)
    tracks = build_tracks(data)
    mot = summarize_mot(tracks)

    print("\n📊 Multi-Human Tracking Evaluation (MOT, no GT)")
    print("------------------------------------------------")
    print(f"Track Completeness              : {track_completeness(data):.3f}")

    if mot is None:
        print("No valid tracks found.")
        print("------------------------------------------------\n")
        return

    print(f"Total Track IDs                 : {mot['num_tracks_total']}")
    print(f"Valid Track IDs (len >= {MIN_TRACK_LEN})     : {mot['num_tracks_valid']}")
    print(f"Fragmentation (short-track frac): {fragmentation_score(tracks):.3f}")
    print("------------------------------------------------")
    print(f"Mean Track Length (frames)      : {mot['mean_track_len_frames']:.1f}")
    print(f"Median Track Length (frames)    : {mot['median_track_len_frames']:.1f}")
    print(f"Mean Track Duration (s)         : {mot['mean_track_duration_s']:.2f}")
    print("------------------------------------------------")
    print(f"Mean Jump Rate (> {MAX_JUMP_DIST}m step)      : {mot['mean_jump_rate']:.3f}")
    print(f"Median Jump Rate                : {mot['median_jump_rate']:.3f}")
    print("------------------------------------------------")
    print(f"Mean Velocity Smoothness        : {mot['mean_velocity_smoothness']:.3f} m/s²")
    print(f"Median Velocity Smoothness      : {mot['median_velocity_smoothness']:.3f} m/s²")
    print(f"Mean Velocity Plausibility      : {mot['mean_velocity_plausibility']:.3f}")
    print(f"Median Velocity Plausibility    : {mot['median_velocity_plausibility']:.3f}")
    print("------------------------------------------------\n")


if __name__ == "__main__":
    main()
