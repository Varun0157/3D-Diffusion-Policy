import open3d as o3d
import os

# ===== SET YOUR PLY FILE HERE =====
PLY_PATH = "env_pc.ply"
# =================================

assert os.path.exists(PLY_PATH), f"PLY file not found: {PLY_PATH}"

pcd = o3d.io.read_point_cloud(PLY_PATH)
assert not pcd.is_empty(), "Point cloud is empty"

print(pcd)
o3d.visualization.draw_geometries(
    [pcd],
    window_name="PLY Viewer",
    width=1024,
    height=768
)
