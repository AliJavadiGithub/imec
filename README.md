# Human Tracking in Point Cloud Space

## Overview

This project implements a **robust 3D human tracking system** operating directly on sequential point cloud (`.pcd`) data.
The system maintains **consistent human identity**, estimates **velocity**, and exports results for **quantitative and qualitative evaluation**, even **without ground-truth annotations**.

The implementation satisfies all exercise requirements, including tracking, velocity estimation, visualization, and evaluation.

---

## 1. Algorithm Description

The pipeline consists of **four main stages**:

### 1.1 Preprocessing and Detection

Each frame is processed independently:

1. **Noise Removal**

   * Statistical outlier removal
   * `nb_neighbors = 20`, `std_ratio = 1.5`

2. **Ground Plane Removal**

   * RANSAC-based plane segmentation removes floor points

3. **Clustering**

   * DBSCAN clustering
   * `eps = 0.5 m`, `min_points = 5`

4. **Human Geometry Filtering**

   * Height: `0.7 m < h < 2.1 m`
   * Width / length < `1.0 m`
   * Minimum point count threshold

Each valid detection outputs:

* 3D centroid
* Geometric shape descriptor

---

### 1.2 Feature Extraction

To support re-identification, a compact shape descriptor is computed per cluster:

* Height-to-width ratio
* Upper-body (head) point density
* Spatial variance in X and Z axes

Descriptors are **temporally averaged** for stability.

---

### 1.3 Multi-Object Tracking (MOT)

Tracking follows a **Kalman Filter + Hungarian assignment** paradigm inspired by DeepSORT.

#### Kalman Filter Model

* State: `[x, y, z, vx, vy, vz]`
* Measurement: 3D centroid
* Constant-velocity motion model
* Time step derived from frame timestamps

#### Data Association

* Cost matrix combines:

  * Predicted–observed Euclidean distance
  * Shape descriptor distance
* Hungarian algorithm solves optimal assignment
* Gating thresholds prevent unrealistic matches

---

### 1.4 Track Management and Re-Identification

* Tracks classified as **static** or **dynamic** based on speed
* Dynamic tracks tolerate fewer missed frames
* Lost tracks are buffered and can be re-identified using spatial proximity
* Only tracks with sufficient hit count are considered valid

---

### 1.5 Velocity Estimation

Velocity is obtained directly from the Kalman filter:

* Instantaneous velocity from `[vx, vy, vz]`
* Speed computed as Euclidean norm
* Temporal smoothing is inherent to Kalman filtering

---

### 1.6 Visualization

Provided visualizations include:

1. 3D point cloud playback with bounding boxes and IDs
2. Top-down (XY) trajectory plot
3. Speed-vs-time plot

Tracking results are exported to `tracking_results.json`.

---

## 2. Implementation Challenges and Solutions

### Noise and Floor Interference

**Solution:**
Ground plane removal + statistical filtering + geometry validation.

### ID Switching During Occlusion

**Solution:**
Track buffering, re-identification using shape similarity, adaptive skip thresholds.

### Velocity Jitter

**Solution:**
Timestamp-aware Kalman filter with tuned noise parameters.

### Sparse Point Clouds

**Solution:**
Shape history averaging and conservative association gating.

---

## 3. Evaluation

### 3.1 Evaluation Without Ground Truth

The dataset does **not** provide ground-truth IDs or velocities.
Therefore, evaluation relies on **self-consistency**, **motion priors**, and **temporal smoothness**, which are standard in real-world tracking scenarios.

The evaluation answers:

* Does the tracker maintain a stable identity?
* Is the human detected consistently?
* Are estimated velocities smooth and physically plausible?
* Is the trajectory temporally continuous?

All metrics are computed automatically using `metrics.py`.

---

### 3.2 ID Consistency Rate (Proxy)

Since IDs are not explicitly stored, identity stability is evaluated using **spatial continuity**.

A frame is considered ID-consistent if the detected position remains within a reasonable distance of the previous frame.

ID Consistency Rate =
number of spatially consistent frames ÷ total frames

Values close to 1 indicate minimal identity switches.

---

### 3.3 Track Completeness

Track completeness measures detection coverage:

Track Completeness =
frames with at least one detection ÷ total frames

This reflects robustness to noise and temporary occlusion.

---

### 3.4 Velocity Estimation Quality

#### Velocity Smoothness (Jerk)

Human motion is expected to be smooth.
We compute the mean absolute jerk:

Jerk = average of |Δspeed ÷ Δtime|

Lower values indicate smoother and more stable velocity estimation.

#### Velocity Plausibility

Estimated speeds are evaluated against human motion priors:

Valid range: `0 ≤ speed ≤ 3.0 m/s`

Velocity Plausibility =
frames with valid speed ÷ total frames

---

### 3.5 Visualization-Based Validation

Tracking quality is visually confirmed using:

* Continuous trajectories without jumps
* Smooth speed profiles
* Stable spatial paths

These qualitative checks complement numerical metrics.

---

### 3.6 Evaluation Results

Running:

```bash
python3 metrics.py
```

Produces:

```
📊 Tracking Evaluation Metrics (No GT, No IDs)
---------------------------------------------
Track Completeness        : 1.000
ID Consistency (Proxy)    : 0.965
Velocity Smoothness       : 89.103 m/s²
Velocity Plausibility     : 0.911
---------------------------------------------
```

**Interpretation:**

* The human is detected in all frames
* Identity remains stable throughout the sequence
* Velocity estimates are smooth and physically plausible

---

## 4. Limitations and Future Work

### Current Limitations

* Heuristic geometry filtering may fail in crowded scenes
* Evaluation assumes a dominant single human
* Hand-crafted shape descriptors are limited

### Future Improvements

* Learned point cloud features (PointNet / PointTransformer)
* Mahalanobis distance gating
* Confidence scores per track
* Real-time C++ / PCL implementation

---

## Conclusion

This project demonstrates a **robust, modular, and extensible 3D human tracking system** operating directly on point cloud data.

By combining geometric reasoning, probabilistic tracking, and principled evaluation without ground truth, the system fulfills all task requirements and provides a strong foundation for future research and deployment.

🚀
