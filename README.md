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

## Main Features

### 3D Human Detection

* DBSCAN clustering for point cloud segmentation
* Geometry-based human candidate validation
* Noise filtering and outlier rejection

### Multi-Object Tracking

* Interacting Multiple Model (IMM) Kalman filter
* Constant Velocity and Random Walk motion models
* Hungarian assignment with statistical gating
* Track initialization and termination
* Static/dynamic track classification
* Lightweight re-identification

### Visualization

* Interactive Open3D playback
* 3D bounding boxes
* Object IDs and speed labels
* Automatic MP4 video generation
* Top-down trajectory visualization

### Evaluation

* Heuristic tracking quality metrics
* Velocity consistency analysis
* Track completeness estimation
* Reference statistic extraction from clean datasets

## Project Structure

```
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

## Requirements

* Python 3.8+
* Open3D
* OpenCV

Python packages:

* open3d
* numpy
* scipy
* scikit-learn
* matplotlib
* filterpy
* opencv-python

## Quick Start

Create a virtual environment:

```
make venv
source .venv/bin/activate
```

Install dependencies:

```
make install
```

Run tracking:

```
make track
```

You will be prompted to choose one of the datasets:

```
[1] Human Only
[2] Entire Map
```

The tracker generates:

```
tracking_results.json
```

## Visualization

Run interactive playback and video recording:

```
make playback
```

Output:

```
tracking_playback.mp4
```

Visualization includes:

* 3D bounding boxes
* Track IDs
* Estimated speed
* Consistent object colors

Plot trajectories and speed profiles:

```
make visualize
```

## Evaluation

Run quantitative evaluation:

```
make evaluate
```

Computed metrics include:

* Track completeness
* ID consistency proxy
* Velocity smoothness
* Velocity plausibility

## Reference Statistics

Extract empirical human statistics:

```
make reference
```

Outputs include:

* Estimated human height
* Estimated human width
* Walking speed statistics
* Ground-truth reference path

## Output Format

```
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

## Technical Highlights

* Robust human detection in noisy point clouds
* Probabilistic multi-object tracking
* Motion estimation using IMM Kalman filtering
* Hungarian data association with gating
* Real-time 3D visualization
* Modular Python implementation
* No appearance features or ground-truth labels required
* Safe handling of corrupted and incomplete frames

## Technologies

* Python
* Open3D
* NumPy
* SciPy
* Scikit-learn
* FilterPy
* OpenCV
* Matplotlib

## Disclaimer

This repository was developed as a technical assessment project demonstrating practical implementation skills in 3D computer vision, point cloud processing, probabilistic tracking, and scientific software engineering.
