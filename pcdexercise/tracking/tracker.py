import os
import json
import numpy as np
import open3d as o3d

from .detector import HumanDetector
from .track import Track
from .association import associate

class HumanTrackerMOT:
    def __init__(self):
        self.tracks = []
        self.lost_tracks = []
        self.next_id = 1
        self.history = []

        self.detector = HumanDetector()

        self.min_hits = 3
        self.max_skip_dynamic = 20
        self.max_skip_static = 50
        self.dist_weight = 0.8
        self.shape_weight = 0.2
        self.gating_threshold = 1.2

    def update(self, frame_id, timestamp_ms, pcd_path):
        if not os.path.exists(pcd_path):
            return

        pcd = o3d.io.read_point_cloud(pcd_path)
        detections = self.detector.detect(pcd)

        prev_ts = self.history[-1]["timestamp_ms"] if self.history else timestamp_ms - 33
        dt = max((timestamp_ms - prev_ts) / 1000.0, 1e-3)

        for t in self.tracks:
            t.predict(dt)

        matches, used = associate(
            self.tracks, detections,
            self.dist_weight, self.shape_weight, self.gating_threshold
        )

        for ti, di in matches.items():
            self.tracks[ti].update(detections[di]["centroid"], detections[di]["shape"])

        active = []
        for i, t in enumerate(self.tracks):
            if i not in matches:
                t.skipped_frames += 1

            speed = np.linalg.norm(t.velocity())
            t.is_static = speed < 0.15
            limit = self.max_skip_static if t.is_static else self.max_skip_dynamic

            if t.skipped_frames <= limit:
                active.append(t)
            else:
                self.lost_tracks.append(t)

        self.tracks = active

        for j, d in enumerate(detections):
            if j not in used:
                self.tracks.append(Track(self.next_id, d["centroid"], d["shape"]))
                self.next_id += 1

        self.history.append({
            "frame_id": frame_id,
            "timestamp_ms": timestamp_ms,
            "detections": [
                {
                    "id": t.id,
                    "position": [round(x, 3) for x in t.kf.x[:3].flatten()],
                    "velocity": [round(v, 3) for v in t.velocity()],
                    "speed": round(float(np.linalg.norm(t.velocity())), 3),
                    "status": "STATIC" if t.is_static else "MOVING",
                    "confidence": round(t.confidence(), 3)
                }
                for t in self.tracks
                if t.hits >= self.min_hits and t.skipped_frames == 0
            ]
        })

    def finalize(self, output="tracking_results.json"):
        with open(output, "w") as f:
            json.dump(self.history, f, indent=4)
        print(f"✅ Results written to {output}")
