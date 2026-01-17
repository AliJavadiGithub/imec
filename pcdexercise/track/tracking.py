import os
import re
import json
from detector import HumanDetector
from tracker import HumanTrackerMOT


def extract_ts(name):
    return int(re.search(r'(\d+)ms', name).group(1))


def main():
    detector = HumanDetector()
    tracker = HumanTrackerMOT()

    data_dir = "mapAll"
    files = sorted(
        [f for f in os.listdir(data_dir) if f.endswith(".pcd")],
        key=extract_ts
    )

    for i, fname in enumerate(files):
        ts = extract_ts(fname)
        detections = detector.detect(os.path.join(data_dir, fname))
        tracker.update(i, ts, detections)

        if i % 50 == 0:
            print(f"Processing Frame {i}/{len(files)}")

    with open("tracking_results.json", "w") as f:
        json.dump(tracker.history, f, indent=4)

    print("✅ Tracking Finished")


if __name__ == "__main__":
    main()
