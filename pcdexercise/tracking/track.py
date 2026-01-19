import numpy as np
from .motion import build_imm_filter

class Track:
    def __init__(self, track_id, centroid, shape):
        self.id = track_id
        self.kf = build_imm_filter(centroid)
        self.shape_history = [shape]
        self.hits = 1
        self.age = 1
        self.skipped_frames = 0
        self.last_seen_pos = centroid
        self.is_static = False

    def predict(self, dt):
        for f in self.kf.filters:
            for i in range(3):
                f.F[i, i + 3] = dt
        self.kf.predict()
        self.age += 1

    def update(self, centroid, shape):
        self.kf.update(centroid)
        self.shape_history.append(shape)
        self.last_seen_pos = centroid
        self.skipped_frames = 0
        self.hits += 1

    def velocity(self):
        return self.kf.x[3:].flatten()

    def avg_shape(self):
        return np.mean(self.shape_history[-15:], axis=0)

    def confidence(self, eps=1e-6, p_max=50.0):
        P = self.kf.P[:3, :3]
        unc = np.sqrt(np.linalg.det(P) + eps)
        return float(np.clip(np.exp(-unc / p_max), 0, 1))
