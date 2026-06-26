````markdown
# 3D Point Cloud Human Tracking

> **Computer Vision | 3D Point Cloud Processing | Multi-Object Tracking | Probabilistic State Estimation | Python | Open3D**

A research-oriented implementation for robust multi-object human detection, tracking, visualization, and evaluation from sequential 3D point cloud data.

The project processes LiDAR/depth-map point cloud sequences, detects human objects using geometric reasoning, estimates trajectories with probabilistic motion models, and provides visualization and quantitative evaluation tools. The implementation emphasizes robustness to noisy observations while remaining suitable for real-time processing.

---

# Research Topics

This project is related to several active research areas in Computer Vision and Artificial Intelligence:

- 3D Computer Vision
- Point Cloud Processing
- Human Detection
- Multi-Object Tracking (MOT)
- Probabilistic State Estimation
- Kalman Filtering
- Data Association
- Real-Time Perception
- Visualization and Performance Evaluation

---

# Main Features

## 3D Human Detection

- DBSCAN clustering for point cloud segmentation
- Geometry-based human candidate validation
- Noise filtering and outlier rejection

## Multi-Object Tracking

- Interacting Multiple Model (IMM) Kalman Filter
- Constant Velocity and Random Walk motion models
- Hungarian assignment with statistical gating
- Track initialization and termination
- Static/Dynamic track classification
- Lightweight re-identification

## Visualization

- Interactive Open3D playback
- 3D bounding boxes
- Object IDs and speed labels
- Automatic MP4 video generation
- Top-down trajectory visualization

## Evaluation

- Heuristic tracking quality metrics
- Velocity consistency analysis
- Track completeness estimation
- Reference statistic extraction from clean datasets

---

# Project Structure

```text
.
├── Makefile
├── tracking.py
├── tracking-playback.py
├── tracking-visualization.py
├── tracking-evaluate.py
├── reference_extractor.py
├── mapHumanOnly/
├── mapAll/
├── tracking_results.json
└── tracking_playback.mp4
```

---

# Requirements

- Python 3.8+
- Open3D
- OpenCV

Python packages

- open3d
- numpy
- scipy
- scikit-learn
- matplotlib
- filterpy
- opencv-python

Install everything using

```bash
make install
```

---

# Quick Start

Create a virtual environment

```bash
make venv
source .venv/bin/activate
```

Install dependencies

```bash
make install
```

Run tracking

```bash
make track
```

Choose one of the datasets:

```
[1] Human Only
[2] Entire Map
```

The tracker generates

```
tracking_results.json
```

---

# Visualization

Interactive playback

```bash
make playback
```

Output

```
tracking_playback.mp4
```

Visualization includes

- 3D bounding boxes
- Track IDs
- Estimated speed
- Consistent object colors

Trajectory analysis

```bash
make visualize
```

Outputs

- Top-down trajectories
- Speed profiles

---

# Quantitative Evaluation

Run

```bash
make evaluate
```

Computed metrics include

- Track completeness
- ID consistency proxy
- Velocity smoothness
- Velocity plausibility

---

# Reference Statistics

Extract empirical human statistics

```bash
make reference
```

Outputs

- Estimated human height
- Estimated human width
- Walking speed statistics
- Ground-truth reference path

---

# Output Format

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

# Technical Highlights

- Robust human detection in noisy point clouds
- Probabilistic multi-object tracking
- Motion estimation using IMM Kalman Filtering
- Hungarian data association with gating
- Real-time 3D visualization
- Modular Python implementation
- No appearance features or ground-truth labels required
- Safe handling of corrupted and incomplete frames

---

# Technologies

- Python
- Open3D
- NumPy
- SciPy
- Scikit-learn
- FilterPy
- OpenCV
- Matplotlib

---

# Disclaimer

This repository was developed as a technical assessment demonstrating practical implementation skills in 3D computer vision, point cloud processing, probabilistic tracking, and scientific software engineering.
````
