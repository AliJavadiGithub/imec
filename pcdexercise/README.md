# 3D Point Cloud Human Tracking

Robust **multi-human tracking** from 3D point cloud sequences with strong suppression of static clutter.
This project detects human-like clusters in LiDAR / depth-map point clouds, tracks them over time using probabilistic motion models, applies lightweight re-identification, and provides visualization and evaluation tools — **without requiring ground truth or training data**.

---

## ✨ Features

### 🧍 3D Human Detection

* DBSCAN clustering on voxel-downsampled point clouds
* Statistical outlier removal and optional ground-plane removal
* Strong geometry-based human validation:

  * Height, width, density constraints
  * Aspect ratio filtering
  * PCA-based **verticality score** (uprightness)
* Designed to work on both:

  * `mapHumanOnly/` (clean)
  * `mapAll/` (full environment with clutter)

---

### 🔁 Multi-Object Tracking (MOT)

* **IMM Kalman Filter** with:

  * Constant Velocity (CV) model
  * Random Walk (RW) model
* Timestamp-aware prediction (`dt` from filenames)
* Track lifecycle management:

  * Active tracks
  * Temporarily lost tracks
  * Re-identification and recovery
* Distinguishes **STATIC** vs **MOVING** humans

---

### 🔗 Data Association & Re-Identification

* Two-stage association:

  1. Mahalanobis-distance gating + Hungarian assignment
  2. Appearance refinement for ambiguous cases
* Lightweight ReID cues (no training required):

  * CLIP multi-view projection embeddings (optional, off-the-shelf)
  * FPFH geometric descriptors
  * Shape descriptors
* Robust ID continuity through occlusions and crossings

---

### 🚫 Static Clutter Suppression (Key Feature)

* Temporal **motion-confirmation gate**:

  * Tracks must accumulate sufficient displacement (≈ 0.9 m over ~2 s)
  * **OR** sustain a minimum speed (≥ 0.20 m/s)
* Moving-evidence counter:

  * Track must be “moving” for several frames before being emitted
* Effectively removes:

  * Poles
  * Walls
  * Static vertical map artifacts

---

### 🎥 Visualization & Playback

* Open3D real-time playback
* Per-ID:

  * 3D bounding boxes
  * Markers
  * Speed labels
* Deterministic color per ID
* Automatic MP4 video recording with fixed resolution

---

### 📊 Analysis & Evaluation (No Ground Truth)

* Trajectory and speed visualization
* Multi-human MOT proxy metrics:

  * Track completeness
  * Fragmentation
  * Jump rate (ID stability proxy)
  * Velocity smoothness
  * Velocity plausibility
* Designed for **sensor-agnostic evaluation**

---

## 📁 Project Structure

```
.
├── Makefile
├── tracking-reid.py                    # Core detection + multi-human tracking
├── tracking-playback.py           # 3D playback + video recording
├── tracking-visualization.py      # Trajectory & speed plots
├── tracking-evaluate.py           # Single-human proxy evaluation
├── tracking-evaluate-mot.py       # Multi-human MOT evaluation (no GT)
├── reference_extractor.py         # Human size/speed statistics extraction
├── mapHumanOnly/                  # Human-only point clouds
├── mapAll/                        # Full environment point clouds
├── tracking_results.json          # Generated tracking output
└── tracking_playback.mp4          # Recorded playback video
```

---

## ⚙️ Requirements

* Python **3.8+**
* System dependencies for **Open3D** and **OpenCV**

Core Python packages (installed via Makefile):

* `open3d`
* `numpy`
* `scipy`
* `scikit-learn`
* `matplotlib`
* `filterpy`
* `opencv-python`

Optional (for appearance ReID):

* `torch`
* `open_clip_torch`
* `Pillow`

---

## 🚀 Quick Start

### 1. Create Virtual Environment

```bash
make venv
source .venv/bin/activate
```

---

### 2. Install Core Dependencies

```bash
make install
```

---

### 3. (Optional) Install ReID Dependencies

Only required if you want CLIP-based appearance ReID:

```bash
make install-reid
```

---

## 🧠 Run Human Tracking

Run the tracker on a dataset:

```bash
make track
```

You will be prompted to choose:

* **[1] Human Only** → `mapHumanOnly`
* **[2] Entire Map** → `mapAll`

Output:

```
tracking_results.json
```

Each detection contains:

* Consistent ID
* 3D position
* 3D velocity
* Speed
* Motion status
* Confidence score

---

## 🎥 Playback & Video Recording

Visualize tracking results and record an annotated video:

```bash
make playback
```

Output:

```
tracking_playback.mp4
```

Features:

* Per-track bounding boxes
* ID + speed labels
* Deterministic colors
* Stable video resolution

---

## 📈 Visualization

Plot trajectories and speed over time:

```bash
make visualize
```

Produces:

* Top-down (X–Y) trajectories
* Speed vs time plots per ID

---

## 📊 Evaluation (No Ground Truth)

### Multi-Human MOT Evaluation

```bash
make evaluate
```

Metrics include:

* Track completeness
* Number of valid track IDs
* Fragmentation (short-track fraction)
* Mean / median track duration
* Jump rate (ID stability proxy)
* Velocity smoothness
* Velocity plausibility

### Single-Human Proxy Evaluation

```bash
python tracking-evaluate.py
```

---

## 🧪 Output Format (`tracking_results.json`)

```json
{
  "frame_id": 42,
  "timestamp_ms": 1400,
  "detections": [
    {
      "id": 1,
      "position": [1.23, 0.45, 1.67],
      "velocity": [0.12, 0.01, 0.00],
      "speed": 0.12,
      "status": "MOVING",
      "confidence": 0.91
    }
  ]
}
```

---

## 📝 Notes

* Designed for **robustness** in cluttered occupancy maps
* No training data or ground truth required
* Strong suppression of static false positives
* Stable ID assignment for multiple humans
* Safe handling of empty or corrupted frames

