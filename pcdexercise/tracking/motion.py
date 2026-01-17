import numpy as np
from filterpy.kalman import KalmanFilter


class MotionModel:
    """
    Constant-velocity Kalman filter wrapper.
    """

    def __init__(self, position: np.ndarray):
        self.kf = KalmanFilter(dim_x=6, dim_z=3)
        self.kf.F = np.eye(6)
        self.kf.H = np.hstack((np.eye(3), np.zeros((3, 3))))
        self.kf.x[:3] = position.reshape(3, 1)

        self.kf.P *= 5.0
        self.kf.R *= 10.0
        self.kf.Q = np.eye(6) * 0.01

    def predict(self, dt: float):
        for i in range(3):
            self.kf.F[i, i + 3] = dt
        self.kf.predict()

    def update(self, position: np.ndarray):
        self.kf.update(position)

    def position(self):
        return self.kf.x[:3].flatten()

    def velocity(self):
        return self.kf.x[3:].flatten()

    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity()))
