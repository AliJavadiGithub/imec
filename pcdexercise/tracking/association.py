import numpy as np
from scipy.optimize import linear_sum_assignment

def associate(tracks, detections, dist_w, shape_w, gate):
    if not tracks or not detections:
        return {}, set()

    cost = np.full((len(tracks), len(detections)), 999.0)

    for i, t in enumerate(tracks):
        for j, d in enumerate(detections):
            dist = np.linalg.norm(t.kf.x[:3].flatten() - d["centroid"])
            if dist <= gate:
                shape_dist = np.linalg.norm(t.avg_shape() - d["shape"])
                cost[i, j] = dist_w * dist + shape_w * shape_dist

    rows, cols = linear_sum_assignment(cost)
    matches = {}

    for r, c in zip(rows, cols):
        if cost[r, c] < gate:
            matches[r] = c

    return matches, set(matches.values())
