"""
Convert master_data format to zarr format compatible with realdex_drill.zarr

Master data format:
- master_data/action/{traj_id}.txt: Text files with space-separated action values per line
- master_data/state/{traj_id}.txt: Text files with space-separated state values per line
- master_data/point_cloud/{traj_id}/frame_XXXX.ply: PLY point cloud files
- master_data/img/{traj_id}/frame_XXXX.png: PNG image files

Target zarr format (from realdex_drill.zarr):
- data/action: (T, action_dim) float32
- data/state: (T, state_dim) float32
- data/point_cloud: (T, num_points, 6) float64 [x, y, z, r, g, b]
- data/img: (T, H, W, 3) uint8
- meta/episode_ends: (num_episodes,) int64

Usage:
    python scripts/convert_master_data_to_zarr.py \
        --input_dir master_data \
        --output_path 3D-Diffusion-Policy/data/custom_data.zarr \
        --num_points 2500 \
        --img_height 720 \
        --img_width 1280
"""

import os
import argparse
import numpy as np
import zarr
from tqdm import tqdm
import cv2
import open3d as o3d


def load_trajectory_actions(action_file):
    """Load actions from text file, skipping the first line."""
    actions = []
    with open(action_file, "r") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue  # Skip first line (initial zero action)
            action = np.array(
                [float(x) for x in line.strip().split()], dtype=np.float32
            )
            actions.append(action)
    return np.array(actions)


def load_trajectory_states(state_file):
    """Load states from text file, skipping the first line."""
    states = []
    with open(state_file, "r") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue  # Skip first line (initial state)
            state = np.array([float(x) for x in line.strip().split()], dtype=np.float32)
            states.append(state)
    return np.array(states)


def load_point_cloud_ply(ply_file, expected_num_points):
    """Load point cloud from PLY file and validate point count."""
    pcd = o3d.io.read_point_cloud(ply_file)
    points = np.asarray(pcd.points, dtype=np.float64)

    if not pcd.has_colors():
        raise ValueError(
            f"Point cloud {ply_file} has no colors. RGB colors are required."
        )

    if len(points) != expected_num_points:
        raise ValueError(
            f"Point cloud {ply_file} has {len(points)} points, expected {expected_num_points}"
        )

    colors = np.asarray(pcd.colors, dtype=np.float64)  # Already in [0, 1] range
    point_cloud = np.concatenate([points, colors], axis=1)  # (N, 6) [x, y, z, r, g, b]

    return point_cloud


def load_image(img_file, expected_height, expected_width):
    """Load image and validate dimensions."""
    img = cv2.imread(img_file)
    if img is None:
        raise ValueError(f"Failed to load image: {img_file}")

    h, w, c = img.shape
    if h != expected_height or w != expected_width:
        raise ValueError(
            f"Image {img_file} has size {h}x{w}, expected {expected_height}x{expected_width}"
        )

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img.astype(np.uint8)


def convert_master_data_to_zarr(
    input_dir, output_path, num_points, img_height, img_width
):
    """
    Convert master_data format to zarr format.

    Args:
        input_dir: Path to master_data directory
        output_path: Path to output zarr file
        num_points: Expected number of points in point cloud
        img_height: Expected image height
        img_width: Expected image width
    """

    # Get list of trajectories
    action_dir = os.path.join(input_dir, "action")
    trajectory_files = sorted([f for f in os.listdir(action_dir) if f.endswith(".txt")])
    trajectory_ids = [f.replace(".txt", "") for f in trajectory_files]

    print(f"Found {len(trajectory_ids)} trajectories: {trajectory_ids}")

    # Collect all data
    all_actions = []
    all_states = []
    all_point_clouds = []
    all_images = []
    episode_ends = []

    current_timestep = 0

    for traj_id in tqdm(trajectory_ids, desc="Converting trajectories"):
        # Load actions and states
        action_file = os.path.join(input_dir, "action", f"{traj_id}.txt")
        state_file = os.path.join(input_dir, "state", f"{traj_id}.txt")

        actions = load_trajectory_actions(action_file)
        states = load_trajectory_states(state_file)
        traj_length = len(actions)

        if len(states) != traj_length:
            raise ValueError(
                f"Trajectory {traj_id}: state length {len(states)} != action length {traj_length}"
            )

        # Load point clouds
        pc_dir = os.path.join(input_dir, "point_cloud", traj_id)
        pc_files = sorted([f for f in os.listdir(pc_dir) if f.endswith(".ply")])

        if len(pc_files) != traj_length:
            raise ValueError(
                f"Trajectory {traj_id}: {len(pc_files)} point clouds != {traj_length} actions"
            )

        point_clouds = []
        for i in range(traj_length):
            pc_file = os.path.join(pc_dir, pc_files[i])
            pc = load_point_cloud_ply(pc_file, expected_num_points=num_points)
            point_clouds.append(pc)
        point_clouds = np.array(point_clouds)

        # Load images
        img_dir = os.path.join(input_dir, "img", traj_id)
        img_files = sorted(
            [f for f in os.listdir(img_dir) if f.endswith(".png") or f.endswith(".jpg")]
        )

        if len(img_files) != traj_length:
            raise ValueError(
                f"Trajectory {traj_id}: {len(img_files)} images != {traj_length} actions"
            )

        images = []
        for i in range(traj_length):
            img_file = os.path.join(img_dir, img_files[i])
            img = load_image(
                img_file, expected_height=img_height, expected_width=img_width
            )
            images.append(img)
        images = np.array(images)

        # Append to all data
        all_actions.append(actions)
        all_states.append(states)
        all_point_clouds.append(point_clouds)
        all_images.append(images)

        current_timestep += traj_length
        episode_ends.append(current_timestep)

    # Concatenate all trajectories
    all_actions = np.concatenate(all_actions, axis=0)
    all_states = np.concatenate(all_states, axis=0)
    all_point_clouds = np.concatenate(all_point_clouds, axis=0)
    all_images = np.concatenate(all_images, axis=0)
    episode_ends = np.array(episode_ends, dtype=np.int64)

    print(f"\nTotal timesteps: {len(all_actions)}")
    print(f"Total episodes: {len(episode_ends)}")
    print(f"Action shape: {all_actions.shape}")
    print(f"State shape: {all_states.shape}")
    print(f"Point cloud shape: {all_point_clouds.shape}")
    print(f"Image shape: {all_images.shape}")
    print(f"Episode ends: {episode_ends}")

    # Create zarr file
    print(f"\nSaving to {output_path}...")
    root = zarr.open(output_path, mode="w")

    # Create data group
    data_group = root.create_group("data")
    data_group.create_dataset(
        "action", data=all_actions, dtype=np.float32, chunks=(256, all_actions.shape[1])
    )
    data_group.create_dataset(
        "state", data=all_states, dtype=np.float32, chunks=(256, all_states.shape[1])
    )
    data_group.create_dataset(
        "point_cloud",
        data=all_point_clouds,
        dtype=np.float64,
        chunks=(256, all_point_clouds.shape[1], all_point_clouds.shape[2]),
    )
    data_group.create_dataset(
        "img",
        data=all_images,
        dtype=np.uint8,
        chunks=(256, all_images.shape[1], all_images.shape[2], all_images.shape[3]),
    )
    # Create dummy depth array (not used by policy but expected in zarr format)
    dummy_depth = np.zeros((len(all_actions), img_height, img_width), dtype=np.float64)
    data_group.create_dataset(
        "depth",
        data=dummy_depth,
        dtype=np.float64,
        chunks=(256, img_height, img_width),
    )

    # Create meta group
    meta_group = root.create_group("meta")
    meta_group.create_dataset("episode_ends", data=episode_ends, dtype=np.int64)

    print("Done!")
    print("\nZarr tree:")
    print(root.tree())


def main():
    parser = argparse.ArgumentParser(
        description="Convert master_data format to zarr format"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="master_data",
        help="Path to master_data directory",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="3D-Diffusion-Policy/data/custom_data.zarr",
        help="Path to output zarr file",
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=2500,
        help="Expected number of points in each point cloud (default: 2500)",
    )
    parser.add_argument(
        "--img_height",
        type=int,
        default=720,
        help="Expected image height (default: 720)",
    )
    parser.add_argument(
        "--img_width",
        type=int,
        default=1280,
        help="Expected image width (default: 1280)",
    )

    args = parser.parse_args()

    # Validate input directory
    if not os.path.exists(args.input_dir):
        raise ValueError(f"Input directory does not exist: {args.input_dir}")

    # Create output directory if needed
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    convert_master_data_to_zarr(
        input_dir=args.input_dir,
        output_path=args.output_path,
        num_points=args.num_points,
        img_height=args.img_height,
        img_width=args.img_width,
    )


if __name__ == "__main__":
    main()
