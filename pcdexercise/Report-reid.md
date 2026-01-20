
---

# REPORT.md — Point Cloud Human Tracking (pcdexercise)

This report describes the final tracking pipeline implemented in `tracking-reid.py` for the **pcdexercise** dataset (`mapHumanOnly/` and `mapAll/`). The goal is to maintain **consistent person IDs** over time and estimate **3D velocity** from timestamped point clouds.

---

## 1) Algorithm description

### Overview

The solution is a **multi-object tracking (MOT)** pipeline specialized for 3D point clouds:

1. Preprocess point cloud
2. Detect candidate human clusters
3. Track candidates over time using an IMM (CV + RW)
4. Associate detections to tracks using gating + Hungarian assignment
5. Re-identify lost tracks
6. Suppress static clutter using temporal motion confirmation
7. Output per-frame detections with position + velocity + confidence

The output schema remains unchanged:

```json
{
  "frame_id": int,
  "timestamp_ms": int,
  "detections": [
    {
      "id": int,
      "position": [x,y,z],
      "velocity": [vx,vy,vz],
      "speed": float,
      "status": "STATIC"|"MOVING",
      "confidence": float
    }
  ]
}
```

---

### 1.1 Detection (DBSCAN + strong human geometry constraints)

Each frame is processed as follows:

* **Voxel downsampling** to reduce noise and normalize sensor density variations
* **Statistical outlier removal** to remove isolated points
* **Optional ground plane removal** using RANSAC plane segmentation
* **DBSCAN clustering** on the remaining points

Each cluster is validated using geometry checks:

* Height range: `min_height ≤ height ≤ max_height`
* Max width: `width ≤ max_width`
* Density: `points / volume ≥ min_density`

To reduce false positives from the environment (e.g., poles/walls), a stronger “human-ness” constraint is used:

* **Aspect ratio constraint**: `height / width ≥ 1.4`
* **PCA verticality score**: require the cluster’s principal axis to be upright
  (verticality = alignment with Z axis), `verticality ≥ 0.85`
* **Minimum number of points**: `≥ 60`

This significantly reduces “human-like” static structures in `mapAll`.

---

### 1.2 Tracking (IMM Kalman Filter: CV + RW)

Each track uses an **Interacting Multiple Model (IMM)** filter with two models:

* **CV (Constant Velocity)**: smooth motion assumption
* **RW (Random Walk-ish)**: handles stop-and-go or jitter

State vector:

[
x = [p_x, p_y, p_z, v_x, v_y, v_z]^T
]

The filter is **dt-aware** using timestamps extracted from filenames (`*_1234ms.pcd`), and the process noise (Q) is built from a continuous white-noise acceleration model.

---

### 1.3 Data association (Mahalanobis gating + Hungarian, then appearance refinement)

Association is performed in two stages:

**Stage 1: motion-only assignment**

* Compute **Mahalanobis distance** between predicted track position and detection centroid
* Gate using a chi-square-inspired threshold (3D gate)
* Run **Hungarian assignment** on the gated cost matrix

**Stage 2: appearance refinement (only for still-ambiguous pairs)**
A secondary Hungarian step can use a weighted combination of:

* Euclidean distance (bounded)
* Shape descriptor distance
* **CLIP multi-view projection embedding cosine distance**
* **FPFH descriptor cosine distance**

This is “SOTA-ish” for a *no-training* setting: true SOTA 3D ReID typically requires task-specific training.

---

### 1.4 Re-identification (lost track recovery)

Tracks that exceed their skip limit are moved to a `lost_tracks` pool.

When a new detection appears, the tracker attempts re-ID using:

* spatial proximity to last known position
* CLIP and/or FPFH similarity (when available)

If matched, the lost track is revived and keeps its original ID.

---

### 1.5 Velocity estimation and smoothing

Velocity is estimated from the filter state, but raw Kalman velocities can be noisy.
To stabilize velocity without introducing heavy lag, the implementation uses:

* **measurement EMA** on position updates (reduces centroid jitter)
* **One Euro filter** on velocity magnitude and vector (smooth but responsive)
* speed is computed as (|v|)

---

### 1.6 Temporal motion confirmation gate (key improvement for mapAll)

Even with strong geometry constraints, occupancy-map clutter can produce occasional “human-ish” clusters.

To suppress these, tracks are only **emitted to the output** if they show evidence of actual motion:

* Each `Track` keeps a short position history (`deque(maxlen=60)` ≈ 2s)
* Compute displacement over this recent window
* Require either:

  * displacement ≥ **0.9 m** over ~2s, **OR**
  * smoothed speed ≥ **0.20 m/s**
* Additionally require **moving evidence counter**:

  * `moving_count` increments when speed > 0.18 m/s and must reach **≥ 8**
  * prevents “wiggle noise tracks” from appearing in results

This gate reduces static false positives while preserving stable IDs.

---

## 2) Implementation challenges and solutions

### Challenge A — Static environment clutter in `mapAll` creates false tracks

**Symptom:** many long-lived IDs that are not people (walls/pillars), visible in playback and trajectory plots.

**Solution:**

* Add PCA **verticality score** + stricter aspect ratio + minimum points
* Tighten DBSCAN parameters to reduce tiny junk clusters
* Add **temporal motion confirmation gate** and **moving_count** requirement to ensure only truly moving objects are reported

---

### Challenge B — Unknown sensor characteristics (density, noise, coordinate conventions)

**Solution:**

* Use **voxel downsampling** and **density thresholds** rather than absolute point counts alone
* Use **dt from timestamps** rather than assuming constant FPS
* Use robust gating (Mahalanobis) instead of pure Euclidean matching

---

### Challenge C — Velocity spikes from noisy centroids / irregular frame timing

**Solution:**

* EMA smoothing of the measurement update before feeding the Kalman update
* One Euro filtering of velocity and speed
* Clamp velocity norms to a physically plausible maximum

---

### Challenge D — Multi-human crossings (identity stability)

**Solution:**

* IMM prediction + Mahalanobis gating stabilizes motion association
* Appearance refinement (CLIP + FPFH) improves ID continuity during close interactions
* Re-ID from lost tracks recovers IDs after temporary disappearance

---

## 3) Quantitative results (MOT, no ground truth)

Using `tracking-evaluate-mot.py` on the final configuration:

* **Track Completeness:** `1.000`
* **Total Track IDs:** `9`
* **Valid Track IDs (len ≥ 10):** `9`
* **Fragmentation (short-track fraction):** `0.000`

Track statistics:

* **Mean Track Length:** `245.4 frames`
* **Median Track Length:** `167.0 frames`
* **Mean Track Duration:** `75.56 s`

Consistency proxies:

* **Mean Jump Rate (> 1.5m step):** `0.001`
* **Median Jump Rate:** `0.000`

Velocity quality proxies:

* **Mean Velocity Smoothness:** `0.696 m/s²`
* **Median Velocity Smoothness:** `0.411 m/s²`
* **Mean Velocity Plausibility:** `1.000`
* **Median Velocity Plausibility:** `1.000`

### Visual evidence

From the attached plots:

* **Top-Down trajectories** show 9 persistent track IDs with long continuous segments
* **Speed vs Time** curves remain mostly in plausible walking ranges with occasional peaks, but without frequent discontinuities

---

## 4) Limitations and future improvements

### Limitations

* **No ground truth**: reported metrics are proxies (smoothness, jump rate), not true IDF1/MOTA/HOTA
* **No learned 3D ReID training**: CLIP projection embeddings + FPFH are strong baselines, but not true SOTA for 3D ReID
* **Parameter sensitivity**: DBSCAN and geometry thresholds are still dataset-dependent
* **Static false positives can remain** if map structures move slightly or if the human motion is very slow
* **Runtime cost**: CLIP embedding computation can be expensive (especially on CPU)

---

### Future improvements

1. **Learned 3D ReID** (PointNet++ / DGCNN / Transformers)
2. **Semantic classification** to reject non-human clusters
3. **Advanced data association** (JPDA, MHT, tracklets)
4. **Adaptive thresholding** based on density statistics
5. **Ground-truth-based evaluation** (IDF1, MOTA, HOTA)
6. **Map-aware static suppression** using long-term occupancy priors

---

## Reproducibility notes

Run tracking:

```bash
python3 tracking-reid.py
```

Playback visualization:

```bash
python3 playback.py
```

Trajectory and speed plots:

```bash
python3 tracking-visualization.py
```

MOT-style evaluation (no GT):

```bash
python3 tracking-evaluate-mot.py
```

---

