import numpy as np

def is_human(cluster):
    h = cluster[:,2].max() - cluster[:,2].min()
    w = np.ptp(cluster[:,0])
    d = np.ptp(cluster[:,1])
    return (0.5 < h < 2.3) and (0.2 < w < 1.2) and (0.2 < d < 1.2)
