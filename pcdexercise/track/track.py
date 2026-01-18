import numpy as np
from filterpy.kalman import KalmanFilter


class Track:
    def __init__(self, id, centroid, shape_feature):
        self.id = id
        self.kf = self.init_kalman(centroid)
        self.shape_history = [shape_feature]
        self.hits = 1
        self.age = 1
        self.skipped_frames = 0
        self.is_static = False
        self.last_seen_pos = centroid

    def init_kalman(self, pos):
        kf = KalmanFilter(dim_x=6, dim_z=3)
        kf.F = np.eye(6)
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        kf.x[:3] = pos.reshape(3, 1)
        kf.P *= 5.0
        kf.R *= 10.0
        kf.Q = np.eye(6) * 0.01
        return kf

    def get_avg_shape(self):
        return np.mean(self.shape_history[-15:], axis=0)


    def compute_confidence(self, min_hits, tau=10.0, v_max=2.0):
        # 1. Track maturity
        c_hits = min(1.0, self.hits / float(min_hits))

        # 2. Visibility / occlusion penalty
        c_visibility = np.exp(-self.skipped_frames / tau)

        # 3. Motion stability penalty
        speed = np.linalg.norm(self.kf.x[3:])
        c_motion = np.exp(-speed / v_max)

        # Weighted combination
        confidence = (0.5 * c_hits) + (0.3 * c_visibility) + (0.2 * c_motion)

        return float(np.clip(confidence, 0.0, 1.0))
