import zarr
import numpy as np
import matplotlib.pyplot as plt
import random
import os

def visualize_zarr_actions(zarr_path, num_episodes_to_plot=3):
    print(f"Opening Zarr file: {zarr_path}")
    data = zarr.open(zarr_path, mode='r')
    
    actions = data['data/action'][:]
    episode_ends = data['meta/episode_ends'][:]
    
    num_episodes = len(episode_ends)
    print(f"Total episodes found: {num_episodes}")
    
    starts = np.concatenate(([0], episode_ends[:-1]))
    lengths = episode_ends - starts
    
    # Randomly select episodes
    selected_indices = random.sample(range(num_episodes), min(num_episodes, num_episodes_to_plot))
    print(f"Selected episodes for visualization: {selected_indices}")
    
    for ep_idx in selected_indices:
        start = starts[ep_idx]
        end = episode_ends[ep_idx]
        ep_actions = actions[start:end]
        
        num_steps, num_dims = ep_actions.shape
        print(f"Plotting Episode {ep_idx} (Length: {num_steps}, Dimensions: {num_dims})")
        
        fig, axes = plt.subplots(num_dims, 1, figsize=(10, 2 * num_dims), sharex=True)
        if num_dims == 1:
            axes = [axes]
            
        for d in range(num_dims):
            axes[d].plot(ep_actions[:, d], label=f'Dim {d}')
            axes[d].set_ylabel(f'Dim {d}')
            axes[d].grid(True)
            
        axes[-1].set_xlabel('Step')
        plt.suptitle(f'Action Data - Episode {ep_idx}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        out_path = f'episode_{ep_idx}_actions.png'
        plt.savefig(out_path)
        print(f"Saved plot to: {out_path}")
        plt.close()

if __name__ == "__main__":
    zarr_path = '3D-Diffusion-Policy/3D-Diffusion-Policy/data/final_no_eef.zarr'
    visualize_zarr_actions(zarr_path)
