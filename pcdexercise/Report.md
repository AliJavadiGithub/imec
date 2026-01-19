# Robust 3D Human Tracking in Point Clouds

## Overview

This project implements a **robust multi-object human tracking system in 3D point clouds** using geometric detection, probabilistic motion modeling, and temporal data association.  
The system operates entirely without ground-truth labels and is designed for real-world LiDAR-style point cloud sequences.

The pipeline consists of:
- Human detection from raw point clouds
- Multi-model Kalman filtering for motion estimation
- Optimal data association using Hungarian assignment
- Lightweight re-identification and track management
- Visualization, evaluation, and playback tools

---

## Algorithm Description

### 1. Human Detection

Human candidates are detected using **DBSCAN clustering** applied to the input point cloud after noise removal and ground-plane segmentation.

Each cluster is validated using **human-specific geometric constraints**:
- Height range: 0.4–2.2 m  
- Maximum width: ≤ 1.0 m  
- Minimum spatial density  

For valid clusters, a **shape descriptor** is extracted:
- Vertical aspect ratio
- Upper-body (head) density
- Variance in horizontal and vertical axes  

These descriptors are later used for association and identity stability.

---

### 2. Motion Modeling (IMM Kalman Filter)

Each track is modeled using an **Interacting Multiple Model (IMM) Kalman Filter** consisting of:
- Constant Velocity (CV) model
- Random Walk (RW) model

State vector:
$$\mathbf{x} = [x, y, z, \dot{x}, \dot{y}, \dot{z}]^\top$$

Key properties:
- Adaptive time step $\Delta t$ computed from timestamps
- Model transition probabilities updated dynamically
- Combined posterior estimated from weighted model hypotheses

This enables smooth tracking during motion while remaining stable during pauses.

---

### 3. Data Association

Track-to-detection matching is solved using:
- **Gating** based on Euclidean distance
- **Cost matrix** combining:
  - Spatial distance (80%)
  - Shape similarity (20%)
- **Hungarian algorithm** for optimal assignment

Association cost:
$$C_{ij} = w_d \cdot \| \mathbf{p}_i - \mathbf{d}_j \| + w_s \cdot \| \mathbf{s}_i - \mathbf{s}_j \|$$

where:
- $\mathbf{p}$ = predicted position  
- $\mathbf{d}$ = detection centroid  
- $\mathbf{s}$ = shape descriptor  

Associations exceeding the gating threshold are rejected.

---

### 4. Track Management & Re-Identification

Tracks are classified as **dynamic** or **static** based on estimated speed:
$$v = \| \dot{\mathbf{p}} \|$$

- Static tracks are retained longer during occlusion
- Dynamic tracks are removed faster to prevent drift

A lightweight **spatial re-identification** allows recovery if:
$$\| \mathbf{p}_{\text{lost}} - \mathbf{p}_{\text{new}} \| < d_{\text{reID}}$$

Only tracks with sufficient temporal support are reported.

---

### 5. Output & Visualization

Tracking results are saved in a stable JSON schema (`tracking_results.json`) and used for:
- Real-time 3D playback with Open3D
- Annotated MP4 video generation
- Trajectory and speed plotting
- Quantitative evaluation without ground truth

---

## Implementation Challenges and Solutions

### Ground Plane & Noise Sensitivity
**Challenge:** Raw point clouds contain heavy noise and ground clutter.  
**Solution:** Statistical outlier removal followed by RANSAC plane segmentation.

---

### Identity Stability Without Appearance Features
**Challenge:** No RGB or appearance embeddings available.  
**Solution:** Shape descriptors + IMM-predicted motion + gating-based association.

---

### Variable Frame Rate
**Challenge:** Irregular timestamps cause unstable velocity estimates.  
**Solution:** Explicit timestamp parsing and dynamic $\Delta t$ handling.

---

### Visualization Consistency
**Challenge:** Open3D window resizing caused video corruption.  
**Solution:** Resolution locking at first frame capture.

---

## Quantitative Results

Evaluation is performed **without ground truth** using proxy metrics derived from physical plausibility and temporal consistency.

| Metric | Value |
|------|------|
| Track Completeness | **1.000** |
| ID Consistency (Proxy) | **0.903** |
| Velocity Smoothness | **298.7 m/s²** |
| Velocity Plausibility | **0.899** |

---

## Evaluation Metrics (Formulas)

### 1. Track Completeness

Fraction of frames with at least one valid detection:

$$\text{Completeness} = \frac{\sum_{t=1}^{T} \mathbb{1}(|D_t| > 0)}{T}$$

where:
- $D_t$ = detections at frame $t$
- $T$ = total number of frames

---

### 2. ID Consistency (Proxy)

Measures trajectory continuity using inter-frame displacement:

$$\text{ID Consistency} = \frac{1}{N-1} \sum_{i=1}^{N-1} \mathbb{1}(\| \mathbf{p}_{i+1} - \mathbf{p}_i \| < d_{\max})$$

Large jumps indicate implicit ID switches.

---

### 3. Velocity Estimation

Instantaneous speed:
$$v_i = \frac{\| \mathbf{p}_{i+1} - \mathbf{p}_i \|}{t_{i+1} - t_i}$$

---

### 4. Velocity Smoothness (Mean Jerk)

Measures temporal stability of motion:

$$\text{Smoothness} = \frac{1}{N-2} \sum_{i=1}^{N-2} \left| \frac{v_{i+1} - v_i}{t_{i+2} - t_{i+1}} \right|$$

Lower values indicate smoother motion.

---

### 5. Velocity Plausibility

Fraction of speeds within human limits:

$$\text{Plausibility} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}(v_i \leq v_{\text{max}})$$

with $v_{\text{max}} = 3.0 \, \text{m/s}$.

---

## Trajectory and Motion Analysis

### Trajectory (Top-Down)

Estimated ground-plane trajectory of tracked humans:

![Trajectory (Top-Down)](trajectory_top_down.png)

---

### Speed vs Time

Instantaneous speed over time:

![Speed vs Time](speed_vs_time.png)

---

## Limitations and Future Improvements

### Current Limitations
- No learned appearance or point-cloud embeddings
- Re-identification limited in dense crowds
- Shape descriptor is heuristic
- Vertical dynamics not explicitly constrained

---

### Future Improvements
- Learned 3D embeddings for Re-ID
- Joint Probabilistic Data Association (JPDA)
- Adaptive covariance-based gating
- Multi-hypothesis track management
- Ground-truth MOT metrics (MOTA, MOTP)
- GPU-accelerated clustering and filtering

---

## Conclusion

This work demonstrates that **robust 3D human tracking is achievable without learning or labeled data**, using principled geometry, probabilistic filtering, and careful engineering.  
The modular design enables future integration of learning-based and large-scale multi-object tracking methods.