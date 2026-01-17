import numpy as np
from scipy.optimize import linear_sum_assignment
from motion import MotionModel


class Track:
    def __init__(self, track_id, centroid, shape):
        self.id = track_id
        self.motion = MotionModel(centroid)
        self.shape_history = [shape]

        self.hits = 1
        self.age = 1
        self.skipped = 0
        self.last_seen = centroid.copy()
        self.is_static = False

    def avg_shape(self):
        return np.mean(self.shape_history[-15:], axis=0)


class MultiObjectTracker:
    """
    Hungarian-based multi-object tracker.
    """

    def __init__(self):
        self.tracks = []
        self.lost = []
        self.next_id = 1

        self.min_hits = 3
        self.max_skip_dyn = 20
        self.max_skip_static = 50
        self.gating = 1.2

    def update(self, detections, dt):
        self._predict(dt)
        assigned_t, assigned_d = self._associate(detections)
        self._manage_tracks(detections, assigned_t, assigned_d)

    def _predict(self, dt):
        for t in self.tracks:
            t.motion.predict(dt)
            t.age += 1

    def _associate(self, detections):
        if not self.tracks or not detections:
            return set(), set()

        cost = np.full((len(self.tracks), len(detections)), 999.0)

        for i, t in enumerate(self.tracks):
            for j, d in enumerate(detections):
                dist = np.linalg.norm(t.motion.position() - d["centroid"])
                if dist <= self.gating:
                    shape_d = np.linalg.norm(t.avg_shape() - d["shape"])
                    cost[i, j] = 0.8 * dist + 0.2 * shape_d

        rows, cols = linear_sum_assignment(cost)
        assigned_t, assigned_d = set(), set()

        for r, c in zip(rows, cols):
            if cost[r, c] < self.gating:
                t = self.tracks[r]
                t.motion.update(detections[c]["centroid"])
                t.shape_history.append(detections[c]["shape"])
                t.last_seen = detections[c]["centroid"]
                t.skipped = 0
                t.hits += 1

                assigned_t.add(r)
                assigned_d.add(c)

        return assigned_t, assigned_d

    def _manage_tracks(self, detections, assigned_t, assigned_d):
        active = []
        for i, t in enumerate(self.tracks):
            if i not in assigned_t:
                t.skipped += 1

            t.is_static = t.motion.speed() < 0.15
            limit = self.max_skip_static if t.is_static else self.max_skip_dyn

            if t.skipped <= limit:
                active.append(t)
            else:
                self.lost.append(t)

        self.tracks = active

        for j, d in enumerate(detections):
            if j in assigned_d:
                continue
            self.tracks.append(
                Track(self.next_id, d["centroid"], d["shape"])
            )
            self.next_id += 1
