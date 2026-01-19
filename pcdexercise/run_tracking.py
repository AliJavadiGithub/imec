import os
from tracking.tracker import HumanTrackerMOT
from tracking.utils import extract_timestamp_ms

def get_user_choice():
    print("\nChoose Dataset:\n [1] Human Only\n [2] Entire Map")
    while True:
        c = input("Enter 1 or 2: ").strip()
        if c == "1": return "mapHumanOnly"
        if c == "2": return "mapAll"

def main():
    tracker = HumanTrackerMOT()
    base = os.path.dirname(os.path.abspath(__file__))
    dataset = get_user_choice()
    data_dir = os.path.join(base, dataset)

    files = sorted(
        [f for f in os.listdir(data_dir) if f.endswith(".pcd")],
        key=extract_timestamp_ms
    )

    for i, f in enumerate(files):
        ts = extract_timestamp_ms(f)
        tracker.update(i, ts, os.path.join(data_dir, f))
        if i % 50 == 0:
            print(f"Processing frame {i}/{len(files)}")

    tracker.finalize()

if __name__ == "__main__":
    main()
