import os
import re
import json
from detection import HumanDetector
from tracker import MultiObjectTracker


def extract_timestamp(fname):
    m = re.search(r"(\d+)ms", fname)
    if not m:
        raise ValueError(f"Invalid filename: {fname}")
    return int(m.group(1))


def main():
    detector = HumanDetector()
    tracker = MultiObjectTracker()
    history = []

    data_dir = "mapAll"
    files = sorted(
        [f for f in os.listdir(data_dir) if f.endswith(".pcd")],
        key=extract_timestamp
    )

    prev_ts = None

    for i, fname in enumerate(files):
        ts = extract_timestamp(fname)
        dt = 0.033 if prev_ts is None else max((ts - prev_ts) / 1000.0, 0.001)
        prev_ts = ts

        detections = detector.detect(os.path.join(data_dir, fname))
        tracker.update(detections, dt)

        frame_out = []
        for t in tracker.tracks:
            if t.hits >= tracker.min_hits and t.skipped == 0:
                frame_out.append({
                    "id": t.id,
                    "position": [round(x, 3) for x in t.motion.position()],
                    "speed": round(t.motion.speed(), 3),
                    "status": "STATIC" if t.is_static else "MOVING"
                })

        history.append({
            "frame_id": i,
            "timestamp_ms": ts,
            "detections": frame_out
        })

    with open("tracking_results.json", "w") as f:
        json.dump(history, f, indent=4)

    print("✅ Tracking finished → tracking_results.json")


if __name__ == "__main__":
    main()
