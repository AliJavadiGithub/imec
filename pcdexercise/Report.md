# Robust Multi-Object Human Tracking in 3D Point Clouds

## 1. Algorithm Description

This project implements a complete pipeline for **multi-object human tracking in 3D point cloud sequences**, consisting of detection, tracking, data association, re-identification, visualization, and evaluation.

### 1.1 Detection

Human detection is performed directly in 3D space using geometric and statistical properties:

* **Preprocessing**: Statistical outlier removal and ground-plane segmentation using RANSAC.
* **Clustering**: DBSCAN is applied to the remaining points to extract candidate object clusters.
* **Geometric validation**: Each cluster is filtered using human-specific constraints:

  * Height range (0.4–2.2 m)
  * Width constraint (≤ 1.0 m)
  * Minimum point density
* **Shape descriptor**: A compact 4D descriptor is extracted, capturing:

  * Vertical aspect ratio
  * Head-region density
  * Horizontal and vertical variance

Each valid detection outputs a centroid and shape descriptor.

### 1.2 Tracking

Each detected human is tracked using an **Interacting Multiple Model (IMM) Kalman Filter**, combining:

* **Constant Velocity (CV) model** — for walking humans
* **Random Walk (RW) model** — for static or slowly moving humans

The IMM allows probabilistic switching between motion models, improving robustness to motion changes.

State vector:

```
[x, y, z, vx, vy, vz]
```

### 1.3 Data Association

Tracking-to-detection association is solved per frame using:

* **Gating**: Euclidean distance threshold on predicted position
* **Cost function**:

  * Weighted spatial distance
  * Shape descriptor distance
* **Hungarian algorithm** for optimal global assignment

### 1.4 Track Management & Re-Identification

* Tracks maintain hit count, age, velocity, and skipped frames
* Dynamic vs static tracks use different maximum skip thresholds
* Simple **spatial re-identification** revives recently lost tracks based on proximity
* Tracks are confirmed only after a minimum number of hits

### 1.5 Output & Visualization

* Results are stored in a stable JSON schema (`tracking_results.json`)
* Playback script renders:

  * Per-ID bounding boxes
  * 3D markers
  * Screen-projected ID and speed labels
* Annotated MP4 video is recorded with fixed resolution

---

## 2. Implementation Challenges and Solutions

### 2.1 No Ground Truth Availability

**Challenge**: No labeled ground truth or identity annotations were provided.

**Solution**:

* Designed proxy evaluation metrics focusing on temporal consistency and physical plausibility
* Avoided ID-based metrics that require GT (e.g., MOTA, IDF1)

### 2.2 Noisy and Sparse Point Clouds

**Challenge**: LiDAR point clouds contain noise, sparsity, and ground clutter.

**Solution**:

* Statistical outlier removal
* Ground plane segmentation
* Density-based clustering (DBSCAN) robust to variable point counts

### 2.3 Identity Switching

**Challenge**: Frequent occlusions and sparse detections can cause ID switches.

**Solution**:

* Shape descriptors added to association cost
* IMM filter smooths prediction during short detection gaps
* Lost-track re-identification based on spatial continuity

### 2.4 Visualization Stability

**Challenge**: Open3D window resizing caused corrupted video output.

**Solution**:

* Locked video resolution at first frame
* Explicit geometry removal and re-adding per frame

---

## 3. Quantitative Results

Evaluation was performed using the provided `tracking-evaluate.py` script.

```
📊 Tracking Evaluation Metrics (No GT, No IDs)
---------------------------------------------
Track Completeness        : 1.000
ID Consistency (Proxy)    : 0.977
Velocity Smoothness       : 95.170 m/s²
Velocity Plausibility     : 0.971
---------------------------------------------
```

### Interpretation

* **Track Completeness (1.000)**
  All tracks persist through their visible lifespan without fragmentation.

* **ID Consistency (0.977)**
  Very few identity switches, despite lack of explicit appearance features.

* **Velocity Smoothness (95.17 m/s²)**
  Indicates stable motion estimates with occasional acceleration spikes during detection loss.

* **Velocity Plausibility (0.971)**
  Most estimated speeds remain within realistic human motion bounds.

Overall, the tracker demonstrates **high temporal stability and physically plausible motion estimates**.

---

## 4. Limitations and Future Improvements

### Current Limitations

* No appearance-based Re-ID (only spatial proximity)
* Shape descriptor is simple and may fail for partial occlusions
* Static thresholds (distance, gating, geometry) are hand-tuned
* Evaluation relies on proxy metrics instead of true GT-based scores

### Future Improvements

* Integrate learned 3D appearance embeddings (e.g., PointNet-based Re-ID)
* Adaptive gating and noise covariance based on detection confidence
* Joint ground removal + semantic segmentation
* Track-level confidence decay and termination logic
* Support for official MOT metrics when GT becomes available

---

## Conclusion

This project delivers a **robust, interpretable, and fully self-contained 3D human tracking system** operating directly on point clouds. Despite the absence of ground truth, the system achieves strong consistency and smooth motion estimates, and provides a solid foundation for future learning-based extensions.
