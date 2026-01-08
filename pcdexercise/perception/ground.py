import numpy as np

def estimate_ground(pcd, dist=0.03):
    plane, _ = pcd.segment_plane(dist, 3, 1000)
    a,b,c,d = plane
    normal = np.array([a,b,c])
    normal /= np.linalg.norm(normal)
    return normal
