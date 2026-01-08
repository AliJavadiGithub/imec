import os
import re
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

# ۱. حذف وارنینگ‌های Open3D
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

def extract_metadata():
    path = "mapHumanOnly/"
    files = sorted([f for f in os.listdir(path) if f.endswith('.pcd')],
                   key=lambda x: int(re.search(r'(\d+)ms', x).group(1)))

    dimensions = []
    point_counts = []
    centroids = []
    timestamps = []

    print(f"Analyzing {len(files)} frames...")

    for f in files:
        full_path = os.path.join(path, f)
        
        # ۲. بررسی فیزیکی فایل قبل از خواندن (جلوگیری از محاسبات غلط)
        if os.path.getsize(full_path) < 200: # فایل‌های خیلی کوچک خراب هستند
            continue
            
        try:
            pcd = o3d.io.read_point_cloud(full_path)
            
            # فیلتر: خوشه‌ای که کمتر از ۱۰ نقطه دارد انسان نیست
            if pcd.is_empty() or len(pcd.points) < 10:
                continue

            ts = int(re.search(r'(\d+)ms', f).group(1))
            bbox = pcd.get_axis_aligned_bounding_box()
            
            dimensions.append(bbox.get_extent())
            point_counts.append(len(pcd.points))
            centroids.append(pcd.get_center())
            timestamps.append(ts / 1000.0)
            
        except:
            continue

    if not centroids:
        print("No valid data found!")
        return

    dims = np.array(dimensions)
    pts = np.array(point_counts)
    pos = np.array(centroids)
    times = np.array(timestamps)

    # ۳. محاسبه دینامیک با حذف پرش‌های زمانی (Time Spikes)
    d_pos = np.diff(pos, axis=0)
    d_time = np.diff(times).reshape(-1, 1)
    
    # جلوگیری از تقسیم بر صفر یا زمان‌های خیلی کوتاه که سرعت را نجومی می‌کنند
    d_time[d_time <= 0] = 0.033 
    
    velocities = d_pos / d_time
    speeds = np.linalg.norm(velocities, axis=1)
    
    # فیلتر کردن سرعت‌های غیرمنطقی (بیش از ۵ متر بر ثانیه برای انسان غیرممکن است)
    valid_speed_idx = speeds < 5.0
    filtered_speeds = speeds[valid_speed_idx]

    print("\n" + "="*40)
    print("📊 CLEANED HUMAN PROFILE REPORT")
    print("="*40)
    print(f"1. Body Dimensions (Mean):")
    print(f"   - Width/Length: {np.mean(dims[:, 0]):.2f}m x {np.mean(dims[:, 1]):.2f}m")
    print(f"   - Height:       {np.mean(dims[:, 2]):.2f}m")
    print(f"\n2. Point Density:")
    print(f"   - Avg Points:   {int(np.mean(pts))}")
    print(f"   - Min Reliable: {int(np.percentile(pts, 5))} points") # ۵ درصد پایین داده‌ها
    print(f"\n3. Kinematics:")
    print(f"   - Normal Speed: {np.mean(filtered_speeds):.2f} m/s")
    print(f"   - Max Walking:  {np.max(filtered_speeds):.2f} m/s")
    print("="*40)

    # رسم مسیر
    plt.plot(pos[:, 0], pos[:, 1], 'g-')
    plt.title("Cleaned Ground Truth Path")
    plt.savefig("ground_truth_path.png")

if __name__ == "__main__":
    extract_metadata()