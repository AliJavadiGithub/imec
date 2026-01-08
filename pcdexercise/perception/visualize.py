import open3d as o3d

def draw_frame(pcd, humans):
    geoms=[pcd]
    for h in humans:
        box=o3d.geometry.AxisAlignedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(h.cluster))
        box.color=(h.id*37%255/255,0.8,0.4)
        geoms.append(box)
    o3d.visualization.draw_geometries(geoms)
