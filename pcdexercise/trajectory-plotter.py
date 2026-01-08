import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

def plot_individual_trajectories(json_file="tracking_results.json"):
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: JSON file not found.")
        return

    # ۱. استخراج مسیرها
    paths = {}
    for frame in data:
        for det in frame['detections']:
            obj_id = det['id']
            if obj_id not in paths:
                paths[obj_id] = {'x': [], 'y': [], 's': []}
            paths[obj_id]['x'].append(det['position'][0])
            paths[obj_id]['y'].append(det['position'][1])
            paths[obj_id]['s'].append(det.get('speed', 0.0))

    num_ids = len(paths)
    if num_ids == 0:
        print("No tracks found.")
        return

    # ۲. محاسبه تعداد سطر و ستون برای Subplots
    cols = 2
    rows = (num_ids + 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 5), squeeze=False)
    axes = axes.flatten()
    
    # پیدا کردن محدوده کلی محورها برای یکسان‌سازی مقیاس (اختیاری)
    all_x = [x for p in paths.values() for x in p['x']]
    all_y = [y for p in paths.values() for y in p['y']]
    x_lim = (min(all_x) - 0.5, max(all_x) + 0.5)
    y_lim = (min(all_y) - 0.5, max(all_y) + 0.5)

    norm = plt.Normalize(0, 0.7) # بازه سرعت بهینه شده

    # ۳. رسم هر ID در یک Subplot جداگانه
    for i, (obj_id, coords) in enumerate(paths.items()):
        ax = axes[i]
        
        points = np.array([coords['x'], coords['y']]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        lc = LineCollection(segments, cmap='jet', norm=norm, alpha=0.9, linewidth=2.5)
        lc.set_array(np.array(coords['s']))
        line = ax.add_collection(lc)
        
        # علامت‌گذاری شروع و پایان
        ax.scatter(coords['x'][0], coords['y'][0], color='green', s=60, label='Start', zorder=5)
        ax.scatter(coords['x'][-1], coords['y'][-1], color='red', marker='x', s=80, label='End', zorder=5)
        
        ax.set_title(f"Track Path - Person ID: {obj_id}", fontsize=14, color='darkblue')
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_aspect('equal')

    # حذف Subplot‌های خالی (اگر تعداد IDها فرد باشد)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # اضافه کردن یک Colorbar کلی برای کل شکل
    fig.subplots_adjust(right=0.85, hspace=0.4)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(line, cax=cbar_ax)
    cbar.set_label('Speed (m/s)', fontsize=12)

    plt.suptitle("Individual Human Trajectories Analysis", fontsize=20, y=0.98)
    
    output_filename = "subplots_trajectory_analysis.png"
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    print(f"✅ Subplot analysis saved as '{output_filename}'")
    plt.show()

if __name__ == "__main__":
    plot_individual_trajectories()