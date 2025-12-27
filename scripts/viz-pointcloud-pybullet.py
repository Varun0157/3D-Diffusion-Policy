import sys
import pathlib
import numpy as np
from termcolor import cprint
import open3d as o3d
# Project root, one level above scripts
# Current file: scripts/viz-pointcloud-pybullet.py
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent  # <-- parent of scripts
sys.path.insert(0, str(ROOT_DIR))  # insert at front for priority
sys.path.append(str(ROOT_DIR))  # now Python can see diffusion_policy_3d

print("Project root:", ROOT_DIR)


print("\n" + "=" * 80)
cprint("  Point Cloud Visualization", "cyan", attrs=["bold"])
print("=" * 80 + "\n")

# Import environment
try:
    from diffusion_policy_3d.env.pybullet.pybullet_wrapper import UR5PickPlaceEnv
    cprint("✓ Environment imported", "green")
except Exception as e:
    cprint(f"✗ Import failed: {e}", "red")
    raise



# Create environment with GUI for visual debugging
cprint("\nCreating environment with GUI...", "yellow")
cprint("(Close the PyBullet window when you're done)", "yellow")

env = UR5PickPlaceEnv(use_gui=True, num_points=2500, image_size=224)
cprint("✓ Environment created", "green")

# # Reset and get observation
obs = env.reset()
cprint("✓ Got initial observation", "green")

# Print point cloud statistics
pcd = obs['point_cloud']
print(f"\nPoint Cloud Statistics:")
print(f"  Shape: {pcd.shape}")
print(f"  X range: [{pcd[:, 0].min():.3f}, {pcd[:, 0].max():.3f}]")
print(f"  Y range: [{pcd[:, 1].min():.3f}, {pcd[:, 1].max():.3f}]")
print(f"  Z range: [{pcd[:, 2].min():.3f}, {pcd[:, 2].max():.3f}]")
print(f"  Valid points: {np.sum(np.abs(pcd).sum(axis=1) > 0.001)}/{pcd.shape[0]}")

# Sample a few points
print(f"\nSample points:")
for i in range(min(10, pcd.shape[0])):
    print(f"  Point {i}: [{pcd[i, 0]:.3f}, {pcd[i, 1]:.3f}, {pcd[i, 2]:.3f}]")

# Take a few steps and collect point clouds
cprint("\nCollecting point clouds from multiple timesteps...", "yellow")
point_clouds = [obs['point_cloud']]

for i in range(1):
    action = np.random.randn(13)
    obs, reward, done, info = env.step(action)
    point_clouds.append(obs['point_cloud'])
    print(f"  Step {i+1}: collected {obs['point_cloud'].shape[0]} points")

cprint("✓ Collected 6 point clouds", "green")

cprint("\nCreating 3D visualization with Open3D...", "yellow")

for idx, pcd in enumerate(point_clouds):

    # Convert to Open3D point cloud
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pcd[:, :3])
    
    # Optionally add colors based on Z height
    # colors = np.zeros_like(pcd[:, :3])
    # colors[:, 0] = (pcd[:, 2] - pcd[:, 2].min()) / (pcd[:, 2].max() - pcd[:, 2].min())
    # colors[:, 1] = 1.0 - colors[:, 0]
    # o3d_pcd.colors = o3d.utility.Vector3dVector(colors)
    
    o3d.visualization.draw_geometries([o3d_pcd], window_name=f"Timestep {idx}", width=800, height=600)

cprint("✓ Open3D visualization complete", "green")