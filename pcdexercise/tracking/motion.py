import numpy as np
from filterpy.kalman import KalmanFilter, IMMEstimator

def build_imm_filter(initial_pos):
    def make_kf(q):
        kf = KalmanFilter(dim_x=6, dim_z=3)
        kf.F = np.eye(6)
        kf.H = np.block([[np.eye(3), np.zeros((3, 3))]])
        kf.x[:3] = initial_pos.reshape(3, 1)
        kf.P *= 5.0
        kf.R *= 10.0
        kf.Q = np.eye(6) * q
        return kf

    cv = make_kf(0.01)
    rw = make_kf(0.1)

    mu = np.array([0.5, 0.5])
    trans = np.array([[0.95, 0.05], [0.05, 0.95]])

    return IMMEstimator([cv, rw], mu, trans)
