import os
import re
import json
from .detector import HumanDetector
from .tracker import HumanTrackerMOT

def get_user_choice():
    """Prompt user to choose between human-only or entire map"""
    print("\n" + "="*60)
    print("Point Cloud Dataset")
    print("="*60)
    print("\nChoose Dataset:")
    print("  [1] Human Only (mapHumanOnly)")
    print("  [2] Entire Occupancy Map (mapAll)")
    print("")
    
    while True:
        choice = input("Enter your choice (1 or 2): ").strip()
        if choice == "1":
            return "mapHumanOnly"
        elif choice == "2":
            return "mapAll"
        else:
            print("Invalid choice. Please enter 1 or 2.")


def extract_ts(name):
    return int(re.search(r'(\d+)ms', name).group(1))


def main():
    detector = HumanDetector()
    tracker = HumanTrackerMOT()

    # Get user choice
    map_choice = get_user_choice()
    
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go up one level to the project root (.../pcdexercise/)
    project_root = os.path.dirname(current_dir)

    data_dir = os.path.join(project_root, map_choice)

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
