import open3d as o3d
import numpy as np
import os

# ---------- Workspace utils ----------
def compute_workspace_bounds(pc_xyz, n_std=2):
    WORK_SPACE = [
        [0.6389609647247474, 4.405105314033407],      # X
        [-1.562868614993293, -1.0280730580106126],    # Y
        [1.2382057599132916, 2.6079001348631374]      # Z
    ]
    return WORK_SPACE

def crop_workspace(pc_xyz, workspace_bounds):
    mask = (
        (pc_xyz[:, 0] >= workspace_bounds[0][0]) & (pc_xyz[:, 0] <= workspace_bounds[0][1]) &
        (pc_xyz[:, 1] >= workspace_bounds[1][0]) & (pc_xyz[:, 1] <= workspace_bounds[1][1]) &
        (pc_xyz[:, 2] >= workspace_bounds[2][0]) & (pc_xyz[:, 2] <= workspace_bounds[2][1])
    )
    return mask

# ---------- Load point cloud ----------
PLY_PATH = "/home/aniruth/cloud_00232_filtered.ply"
assert os.path.exists(PLY_PATH)

pcd = o3d.io.read_point_cloud(PLY_PATH)
assert not pcd.is_empty()

pc_xyz = np.asarray(pcd.points)
print("Original points:", pc_xyz.shape[0])

# ---------- Compute workspace and crop ----------
workspace = compute_workspace_bounds(pc_xyz)
mask = crop_workspace(pc_xyz, workspace)
cropped_xyz = pc_xyz[mask]
print("Cropped points:", cropped_xyz.shape[0])

# ---------- Create Open3D clouds ----------
# Raw cloud (no color)
pcd_raw_nocolor = o3d.geometry.PointCloud()
pcd_raw_nocolor.points = o3d.utility.Vector3dVector(pc_xyz)
pcd_raw_nocolor.colors = o3d.utility.Vector3dVector([])

# Cropped cloud (no color)
pcd_cropped_nocolor = o3d.geometry.PointCloud()
pcd_cropped_nocolor.points = o3d.utility.Vector3dVector(cropped_xyz)
pcd_cropped_nocolor.colors = o3d.utility.Vector3dVector([])

# Cropped cloud (with original color)
pcd_cropped_color = o3d.geometry.PointCloud()
pcd_cropped_color.points = o3d.utility.Vector3dVector(cropped_xyz)

# Retain original colors for the cropped points
if np.asarray(pcd.colors).shape[0] == pc_xyz.shape[0]:
    original_colors = np.asarray(pcd.colors)
    pcd_cropped_color.colors = o3d.utility.Vector3dVector(original_colors[mask])
else:
    # If original cloud has no color, just leave empty
    pcd_cropped_color.colors = o3d.utility.Vector3dVector([])

# ---------- Create coordinate axes ----------
axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

# ---------- Visualizations ----------
print("1️⃣ Showing RAW point cloud (no color) with axes")
o3d.visualization.draw_geometries(
    [pcd_raw_nocolor, axes],
    window_name="Raw Point Cloud + Axes",
    width=1024, height=768
)

print("2️⃣ Showing CROPPED workspace point cloud (no color) with axes")
o3d.visualization.draw_geometries(
    [pcd_cropped_nocolor, axes],
    window_name="Cropped Point Cloud (No Color) + Axes",
    width=1024, height=768
)

print("3️⃣ Showing CROPPED workspace point cloud (with color) + axes")
o3d.visualization.draw_geometries(
    [pcd_cropped_color, axes],
    window_name="Cropped Point Cloud (With Color) + Axes",
    width=1024, height=768
)


