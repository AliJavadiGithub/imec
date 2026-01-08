import numpy as np
import open3d as o3d

def preprocess(pc, voxel=0.05, z_min=-2, z_max=3):
    pcd = pc.voxel_down_sample(voxel)
    pts = np.asarray(pcd.points)
    mask = (pts[:,2] > z_min) & (pts[:,2] < z_max)
    pcd.points = o3d.utility.Vector3dVector(pts[mask])
    return pcd
