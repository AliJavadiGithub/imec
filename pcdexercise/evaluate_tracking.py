# evaluation.py
"""
Simple evaluation utilities for single-person tracking
Compares with ground truth or visual inspection metrics
"""

import numpy as np
from typing import List, Tuple


def compute_trajectory_metrics(
    timestamps: List[float],
    positions: List[np.ndarray],
    min_frames: int = 20
) -> dict:
    """
    Very basic trajectory quality metrics
    """
    if len(positions) < min_frames:
        return {"valid": False, "msg": "too few frames"}

    pos = np.array(positions)
    deltas = np.diff(pos, axis=0)
    speeds = np.linalg.norm(deltas, axis=1)
    ts = np.array(timestamps)
    dt = np.diff(ts)

    valid = dt > 0
    speeds = speeds[valid]
    dt = dt[valid]

    mean_speed = np.mean(speeds)
    max_speed = np.max(speeds)
    acceleration = np.diff(speeds) / dt[1:]

    return {
        "valid": True,
        "num_frames": len(positions),
        "mean_speed_mps": float(mean_speed),
        "max_speed_mps": float(max_speed),
        "max_acceleration": float(np.max(np.abs(acceleration))) if len(acceleration) > 0 else 0.0,
        "trajectory_length_m": float(np.sum(speeds * dt))
    }


def id_consistency_score(active_ids_per_frame: List[List[int]]) -> float:
    """
    Very crude ID stability metric for single person tracking
    1.0 = perfect, 0.0 = completely broken
    """
    if not active_ids_per_frame:
        return 0.0

    main_id = None
    count = 0

    for frame_ids in active_ids_per_frame:
        if frame_ids:
            most_common = max(set(frame_ids), key=frame_ids.count)
            if main_id is None:
                main_id = most_common
            if most_common == main_id:
                count += 1

    return count / len(active_ids_per_frame) if active_ids_per_frame else 0.0