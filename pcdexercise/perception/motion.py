import numpy as np

def motion_state(vel):
    s = np.linalg.norm(vel)
    if s < 0.2: return "STILL"
    if s < 1.8: return "WALKING"
    return "RUNNING"
