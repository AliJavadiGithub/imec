# Point Cloud Human Tracking Exercise

This repository contains point cloud data for a human tracking exercise. The data consists of timestamped point cloud files (.pcd) captured from occupancy mapping.

## Project Structure

```yaml
.
├── mapAll/                 # Complete occupancy map (environment + human)
├── mapHumanOnly/          # Human-only point cloud data
├── playback.py            # Point cloud visualization script
├── Makefile               # Build automation
└── README.md              # This file
```

## Setup Instructions

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)

### Installation

1. Create and activate a virtual environment:
```bash
make venv
```

2. Activate the virtual environment:
```bash
source .venv/bin/activate
```

3. Install dependencies:
```bash
make install
```

## Visualization

To visualize the point cloud data, use the playback script:

```bash
make run
```

You will be prompted to choose:
- **Human Only**: Visualizes only the human point cloud data
- **Entire Map**: Visualizes the complete occupancy map (environment + human)

The visualization features:
- **Green points** on a **black background**
- Larger point size for better visibility
- 30Hz playback rate
- Real-time timestamp display

### Controls
- Close the window to stop playback
- The visualization will automatically loop through all frames

## Your Task: Human Tracking in Point Cloud Space

### Objective

Implement a robust human tracking system that maintains a **consistent ID** for the detected human and calculates their **velocity (speed)** throughout the sequence.

### Requirements

#### 1. Consistent Human ID Tracking
- Develop an algorithm to track the human across all frames
- Assign a unique, persistent ID to the human
- Handle occlusions and temporary disappearances
- Ensure the ID remains consistent throughout the entire sequence

#### 2. Velocity Estimation
- Calculate the human's velocity in 3D space (m/s)
- Compute both instantaneous and smoothed velocity
- Consider the timestamp information in the filenames (e.g., `occupied_1234ms.pcd`)
- Account for frame-to-frame variations

#### 3. Implementation Guidelines

**Suggested Approach:**
- Use point cloud clustering algorithms (e.g., DBSCAN, Euclidean clustering)
- Implement a tracking framework leveraging multi-object tracking techniques (e.g., Kalman Filter, Hungarian Algorithm for data association, DeepSort-inspired approaches)
- Extract features from point cloud clusters (centroid, bounding box, size, geometric features)
- Use temporal information to associate detections across frames
- Apply re-identification strategies to maintain consistent tracking IDs

**Implementation Notes:**
- Python prototyping is acceptable and recommended for initial development
- C++ implementation (using PCL, Open3D C++ API) is appreciated and earns extra points
- Consider point cloud-specific adaptations of modern tracking paradigms

**Data Structure Recommendations:**
```python
# Example tracking output structure
{
    "frame_id": int,
    "timestamp_ms": int,
    "detections": [
        {
            "human_id": int,          # Consistent ID (re-identification)
            "centroid": [x, y, z],    # 3D position
            "velocity": [vx, vy, vz], # 3D velocity
            "speed": float,            # Magnitude of velocity
            "confidence": float        # Tracking confidence
        }
    ]
}
```

#### 4. Evaluation Metrics

Your implementation should provide:
- ID consistency rate (percentage of frames with correct ID)
- Velocity estimation accuracy
- Track completeness (percentage of frames with successful detection)
- Visualization of tracked trajectory

#### 5. Deliverables

1. **Code** (`tracking.py`):
   - Clean, documented implementation
   - Modular design with separate functions for detection, tracking, and velocity estimation

2. **Results Visualization**:
   - Trajectory plot showing human path
   - Velocity plot over time
   - Visualization of tracking results overlaid on point clouds

3. **Report** (`REPORT.md`):
   - Algorithm description
   - Implementation challenges and solutions
   - Quantitative results
   - Limitations and future improvements

### Hints and Tips

- Start with the `mapHumanOnly/` data for easier development
- Use Open3D's clustering and filtering functions
- Consider the temporal aspect: consecutive frames are ~30ms apart
- Think about noise filtering and outlier rejection
- Test your algorithm on different segments of the sequence

### Useful Libraries

- **Open3D**: Point cloud processing and visualization
- **NumPy**: Numerical computations
- **SciPy**: Clustering algorithms (DBSCAN)
- **scikit-learn**: Additional machine learning tools
- **Matplotlib**: Plotting and visualization

### Evaluation Criteria

- **Correctness** (40%): Does the tracker maintain consistent ID?
- **Accuracy** (30%): How accurate is the velocity estimation?
- **Code Quality** (20%): Is the code clean, modular, and well-documented?
- **Innovation** (10%): Did you implement any creative solutions or optimizations?

## Additional Resources

- [Open3D Documentation](http://www.open3d.org/docs/)
- [Point Cloud Processing Tutorial](http://www.open3d.org/docs/latest/tutorial/geometry/pointcloud.html)
- Kalman Filter resources for tracking
- Hungarian Algorithm for data association

## Troubleshooting

### Virtual Environment Issues
If you encounter issues with the virtual environment:
```bash
make clean-venv
make venv
```

### Missing Dependencies
Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Point Cloud Loading Issues
Ensure you're using the correct path to the `.pcd` files in your implementation.

## Questions?

If you have questions about the exercise, please reach out to Hamed or Constantin

Good luck! 🚀
