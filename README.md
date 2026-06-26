# 3D Point Cloud Human Tracking

**Computer Vision | 3D Point Cloud Processing | Multi-Object Tracking | Probabilistic State Estimation | Python | Open3D**

This repository contains a research-oriented implementation for robust human detection, multi-object tracking, visualization, and evaluation from sequential 3D point cloud data.

The project processes LiDAR/depth-map point cloud sequences, detects human candidates using clustering and geometric reasoning, estimates trajectories with probabilistic motion models, and provides tools for playback, visualization, and quantitative analysis.

## Research Relevance

This project is related to several active research areas in computer vision and artificial intelligence:

* 3D computer vision
* Point cloud processing
* Human detection
* Multi-object tracking
* Probabilistic state estimation
* Kalman filtering
* Data association
* Real-time perception
* Visualization and performance evaluation


---

## ✨ Features

- **3D Human Detection**
  - DBSCAN clustering on point clouds
  - Geometry-based human validation
- **Multi-Object Tracking (MOT)**
  - IMM Kalman Filter (Constant Velocity + Random Walk)
  - Hungarian assignment with gating
  - Track lifecycle management (static vs dynamic)
  - Lightweight re-identification
- **Visualization & Playback**
  - Open3D real-time playback
  - Per-ID bounding boxes, markers, and labels
  - Automatic MP4 video recording
- **Analysis & Evaluation**
  - Trajectory and speed plots
  - Heuristic tracking quality metrics (no ground truth required)
  - Reference statistic extraction from clean datasets

---

## 📁 Project Structure

```

.
├── Makefile
├── tracking.py                    # Core human detection + multi-object tracking
├── tracking-playback.py           # 3D playback + video recording
├── tracking-visualization.py      # Trajectory & speed plots
├── tracking-evaluate.py           # Quantitative tracking metrics (proxy)
├── reference_extractor.py         # Human size/speed statistics extraction
├── mapHumanOnly/                  # Point clouds with only humans
├── mapAll/                        # Full environment point clouds
├── tracking_results.json          # Generated tracking output
└── tracking_playback.mp4          # Recorded playback video

```

## ⚙️ Requirements

- Python **3.8+**
- System dependencies for **Open3D** and **OpenCV**

Python packages (installed via Makefile):

- `open3d`
- `numpy`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `filterpy`
- `opencv-python`



## 🚀 Quick Start

### 1. Create Virtual Environment

```bash
make venv
source .venv/bin/activate
```



### 2. Install Dependencies

```bash
make install
```

---

## 🧠 Run Human Tracking

Run the tracker over a dataset (`mapHumanOnly` or `mapAll`):

```bash
make track
```

You will be prompted to choose:

* **[1] Human Only**
* **[2] Entire Map**

This produces:

```
tracking_results.json
```

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
* Deterministic color per ID
* Stable video resolution

---

## 📈 Visualization

Plot trajectories and speed over time:

```bash
make visualize
```

Outputs:

* Top-down (X–Y) trajectories
* Speed vs time plots

---

## 📊 Evaluation (No Ground Truth)

Compute heuristic tracking quality metrics:

```bash
make evaluate
```

Metrics include:

* Track completeness
* ID consistency proxy
* Velocity smoothness
* Velocity plausibility

---

## 📐 Reference Statistics Extraction

Extract empirical human size and speed statistics from mapHumanOnly:

```bash
make reference
```

Outputs:

* Console report (height, width, speed)
* `ground_truth_path.png`

---

## 🧹 Cleanup

Remove Python cache files:

```bash
make clean
```

Remove virtual environment:

```bash
make clean-venv
```

Full cleanup:

```bash
make clean-all
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
      "velocity": [0.12, 0.01, 0.0],
      "speed": 0.12,
      "status": "MOVING",
      "confidence": 0.91
    }
  ]
}
```

---

## 📝 Notes

* Designed for robustness over noisy point clouds
* Works without ground truth or appearance features
* Optimized for human-scale motion and geometry
* Safe failure handling for corrupted or empty frames
