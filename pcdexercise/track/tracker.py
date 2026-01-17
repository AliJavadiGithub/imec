import numpy as np
from scipy.optimize import linear_sum_assignment
from track import Track


class HumanTrackerMOT:
    def __init__(self):
        self.tracks = []
        self.lost_tracks = []
        self.next_id = 1
        self.history = []

        self.min_hits = 3
        self.max_skip_dynamic = 20
        self.max_skip_static = 50
        self.dist_weight = 0.80
        self.shape_weight = 0.20
        self.gating_threshold = 1.2

    def update(self, frame_id, timestamp, detections):
        prev_ts = self.history[-1]['timestamp_ms'] if self.history else timestamp - 33
        dt = max((timestamp - prev_ts) / 1000.0, 0.001)

        for t in self.tracks:
            for i in range(3):
                t.kf.F[i, i + 3] = dt
            t.kf.predict()
            t.age += 1

        assigned_tracks, assigned_dets = set(), set()

        if self.tracks and detections:
            cost = np.zeros((len(self.tracks), len(detections)))

            for i, track in enumerate(self.tracks):
                for j, det in enumerate(detections):
                    dist = np.linalg.norm(
                        track.kf.x[:3].flatten() - det['centroid']
                    )
                    shape_dist = np.linalg.norm(
                        track.get_avg_shape() - det['shape']
                    )

                    cost[i, j] = (
                        999.0 if dist > self.gating_threshold
                        else self.dist_weight * dist +
                             self.shape_weight * shape_dist
                    )

            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if cost[r, c] < self.gating_threshold:
                    t = self.tracks[r]
                    t.kf.update(detections[c]['centroid'])
                    t.shape_history.append(detections[c]['shape'])
                    t.last_seen_pos = detections[c]['centroid']
                    t.skipped_frames = 0
                    t.hits += 1
                    assigned_tracks.add(r)
                    assigned_dets.add(c)

        active = []
        for i, t in enumerate(self.tracks):
            if i not in assigned_tracks:
                t.skipped_frames += 1

            vel = np.linalg.norm(t.kf.x[3:])
            t.is_static = vel < 0.15
            limit = self.max_skip_static if t.is_static else self.max_skip_dynamic

            if t.skipped_frames <= limit:
                active.append(t)
            else:
                self.lost_tracks.append(t)

        self.tracks = active

        for j, det in enumerate(detections):
            if j not in assigned_dets:
                found = False
                for lt in self.lost_tracks:
                    if np.linalg.norm(lt.last_seen_pos - det['centroid']) < 1.5:
                        lt.kf.x[:3] = det['centroid'].reshape(3, 1)
                        lt.skipped_frames = 0
                        self.tracks.append(lt)
                        self.lost_tracks.remove(lt)
                        found = True
                        break

                if not found:
                    self.tracks.append(
                        Track(self.next_id, det['centroid'], det['shape'])
                    )
                    self.next_id += 1

        frame_out = []
        for t in self.tracks:
            if t.skipped_frames == 0 and t.hits >= self.min_hits:
                pos = t.kf.x[:3].flatten().tolist()
                speed = float(np.linalg.norm(t.kf.x[3:]))

                frame_out.append({
                    "id": t.id,
                    "position": [round(p, 3) for p in pos],
                    "speed": round(speed, 3),
                    "status": "STATIC" if t.is_static else "MOVING"
                })

        self.history.append({
            "frame_id": frame_id,
            "timestamp_ms": timestamp,
            "detections": frame_out
        })
