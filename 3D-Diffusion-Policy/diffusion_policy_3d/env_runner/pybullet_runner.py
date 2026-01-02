import wandb
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # For headless plotting
from termcolor import cprint
from diffusion_policy_3d.env import UR5PickPlaceEnv
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy_3d.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper

from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
import diffusion_policy_3d.common.logger_util as logger_util


class UR5PyBulletRunner(BaseRunner):
    """
    Extended runner that plots predicted vs ground truth joint angle deltas
    """
    def __init__(self,
                 output_dir,
                 n_train=10,
                 n_test=10,
                 max_steps=350,
                 n_obs_steps=2,
                 n_action_steps=8,
                 fps=10,
                 crf=22,
                 tqdm_interval_sec=5.0,
                 use_gui=False,
                 num_points=1024,
                 image_size=224,
                 use_workspace_crop=True,
                 workspace_std=2.0,
                 visualize_gt=True,
                 plot_joint_deltas=True,  # NEW: Enable joint delta plotting
                 ):
        super().__init__(output_dir)
        
        self.max_steps = max_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.fps = fps
        self.crf = crf
        self.tqdm_interval_sec = tqdm_interval_sec
        self.visualize_gt = visualize_gt
        self.plot_joint_deltas = plot_joint_deltas  # NEW
        
        # Environment factory function
        def env_fn():
            return MultiStepWrapper(
                SimpleVideoRecordingWrapper(
                    UR5PickPlaceEnv(
                        use_gui=use_gui,
                        num_points=num_points,
                        image_size=image_size,
                        use_workspace_crop=use_workspace_crop,
                        workspace_std=workspace_std,
                        visualize_gt=self.visualize_gt,
                    )
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method='sum',
            )
        
        self.env_test = env_fn()
        self.episode_test = n_test
        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        
    def plot_joint_deltas_comparison(self, pred_actions, gt_actions, episode_id):
        """
        Plot predicted vs ground truth joint angle deltas
        
        Args:
            pred_actions: (T, action_dim) numpy array of predicted actions
            gt_actions: (T, action_dim) numpy array of ground truth actions
            episode_id: Episode identifier for the plot title
        
        Returns:
            matplotlib figure object
        """
        # Debug: print shapes
        print(f"pred_actions shape: {pred_actions.shape}")
        print(f"gt_actions shape: {gt_actions.shape}")
        
        # Handle different action dimensions
        # If actions are already just joint deltas (7D), use them directly
        # If actions are full state (13D), extract last 7 dimensions
        if pred_actions.shape[1] == 7:
            pred_joint_deltas = pred_actions
            gt_joint_deltas = gt_actions
        elif pred_actions.shape[1] == 13:
            # Extract joint deltas (last 7 dimensions: 6 arm joints + 1 gripper)
            pred_joint_deltas = pred_actions[:, 6:13]  # Shape: (T, 7)
            gt_joint_deltas = gt_actions[:, 6:13]      # Shape: (T, 7)
        else:
            raise ValueError(f"Unexpected action dimension: {pred_actions.shape[1]}")
        
        T = pred_joint_deltas.shape[0]
        timesteps = np.arange(T)
        
        joint_names = [
            'Joint 1', 'Joint 2', 'Joint 3', 
            'Joint 4', 'Joint 5', 'Joint 6', 
            'Gripper'
        ]
        
        # Create a figure with subplots for each joint
        fig, axes = plt.subplots(4, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i in range(7):
            ax = axes[i]
            
            # Plot predicted and ground truth
            ax.plot(timesteps, gt_joint_deltas[:, i], 
                   label='Ground Truth', color='blue', linewidth=2, alpha=0.7)
            ax.plot(timesteps, pred_joint_deltas[:, i], 
                   label='Predicted', color='red', linewidth=2, alpha=0.7, linestyle='--')
            
            # Calculate error metrics
            mse = np.mean((pred_joint_deltas[:, i] - gt_joint_deltas[:, i])**2)
            mae = np.mean(np.abs(pred_joint_deltas[:, i] - gt_joint_deltas[:, i]))
            
            ax.set_title(f'{joint_names[i]}\nMSE: {mse:.6f}, MAE: {mae:.6f}')
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Joint Angle Delta (rad)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Remove the extra subplot
        fig.delaxes(axes[7])
        
        # Add overall title
        fig.suptitle(f'Episode {episode_id}: Predicted vs Ground Truth Joint Deltas', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def run(self, policy: BasePolicy, dataset=None):
        """
        Run evaluation with joint delta plotting
        
        Args:
            policy: Policy to evaluate
            dataset: Optional dataset to extract GT trajectories from
        """
        device = policy.device

        all_returns_test = []
        all_success_rates_test = []
        
        # Storage for plotting
        all_pred_actions = []
        all_gt_actions = []

        cprint("=" * 50, "cyan")
        cprint("Running on TEST environment with Joint Delta Plotting", "cyan")
        cprint("=" * 50, "cyan")

        for episode_id in range(self.episode_test):
            # Extract GT trajectory from dataset
            gt_actions_full = None
            if dataset is not None:
                try:
                    val_dataset = dataset.get_validation_dataset()
                    sample_idx = episode_id % len(val_dataset)
                    sample = val_dataset[sample_idx]
                    
                    gt_actions_full = sample['action']  # Shape: (T, 13)
                    
                    if isinstance(gt_actions_full, torch.Tensor):
                        gt_actions_full = gt_actions_full.cpu().numpy()
                    
                    # Set GT trajectory for visualization
                    if self.visualize_gt:
                        base_env = self.env_test.env.env
                        base_env.set_gt_trajectory(gt_actions_full)
                        base_env.enable_gt_visualization(True)
                    
                    cprint(f"[Episode {episode_id}] Loaded GT with {len(gt_actions_full)} waypoints", "cyan")
                except Exception as e:
                    cprint(f"[Warning] Could not load GT trajectory: {e}", "yellow")

            obs = self.env_test.reset()
            policy.reset()

            reward_sum = 0.0
            done = False
            
            # Storage for this episode
            episode_pred_actions = []
            episode_gt_actions = []

            for step_id in range(self.max_steps):
                np_obs_dict = dict(obs)

                obs_dict = dict_apply(
                    np_obs_dict,
                    lambda x: torch.from_numpy(x).to(device=device)
                )

                with torch.no_grad():
                    action_dict = policy.predict_action({
                        'point_cloud': obs_dict['point_cloud'].unsqueeze(0),
                        'agent_pos': obs_dict['agent_pos'].unsqueeze(0)
                    })

                # Extract action - handle both single and multi-step actions
                action = dict_apply(
                    action_dict,
                    lambda x: x.detach().cpu().numpy()
                )['action']
                
                # MultiStepWrapper may return (1, n_action_steps, action_dim)
                # or (n_action_steps, action_dim), we need to flatten properly
                if action.ndim == 3:
                    action = action.squeeze(0)  # Remove batch dim: (n_action_steps, action_dim)
                elif action.ndim == 2 and action.shape[0] == 1:
                    action = action.squeeze(0)  # Remove batch dim if present
                
                # Debug: print action shape on first step
                if step_id == 0 and episode_id == 0:
                    print(f"Action shape from policy: {action.shape}")
                    print(f"Action sample (first action): {action[0] if action.ndim == 2 else action}")
                
                # Store predicted action(s)
                # If multi-step (n_action_steps, action_dim), store each action
                if action.ndim == 2:
                    for single_action in action:
                        episode_pred_actions.append(single_action)
                else:
                    episode_pred_actions.append(action)
                
                # Store corresponding GT action(s) if available
                # Need to account for n_action_steps
                if gt_actions_full is not None:
                    if action.ndim == 2:
                        # Multi-step action, collect corresponding GT actions
                        for i in range(action.shape[0]):
                            gt_idx = step_id * action.shape[0] + i
                            if gt_idx < len(gt_actions_full):
                                gt_action = gt_actions_full[gt_idx]
                                if step_id == 0 and i == 0 and episode_id == 0:
                                    print(f"GT action shape: {gt_action.shape}")
                                    print(f"GT action sample: {gt_action}")
                                episode_gt_actions.append(gt_action)
                    else:
                        # Single-step action
                        if step_id < len(gt_actions_full):
                            gt_action = gt_actions_full[step_id]
                            if step_id == 0 and episode_id == 0:
                                print(f"GT action shape: {gt_action.shape}")
                                print(f"GT action sample: {gt_action}")
                            episode_gt_actions.append(gt_action)

                obs, reward, done, info = self.env_test.step(action)
                reward_sum += reward
                done = np.all(done)

                if done:
                    break

            all_returns_test.append(reward_sum)
            all_success_rates_test.append(self.env_test.env.is_success())
            
            # Store actions for this episode
            if len(episode_pred_actions) > 0:
                episode_pred_array = np.array(episode_pred_actions)
                all_pred_actions.append(episode_pred_array)
                if episode_id == 0:
                    print(f"Episode 0 pred actions stored shape: {episode_pred_array.shape}")
            if len(episode_gt_actions) > 0:
                episode_gt_array = np.array(episode_gt_actions)
                all_gt_actions.append(episode_gt_array)
                if episode_id == 0:
                    print(f"Episode 0 GT actions stored shape: {episode_gt_array.shape}")

            cprint(
                f"Test Episode {episode_id}: "
                f"Reward={reward_sum:.2f}, "
                f"Success={self.env_test.env.is_success()}",
                "yellow"
            )

        # ---- Metrics ----
        SR_mean_test = np.mean(all_success_rates_test)
        returns_mean_test = np.mean(all_returns_test)

        self.logger_util_test.record(SR_mean_test)

        log_data = {
            'mean_success_rates_test': SR_mean_test,
            'mean_returns_test': returns_mean_test,
            'SR_test_L3': self.logger_util_test.average_of_largest_K(),
            'test_mean_score': SR_mean_test
        }

        cprint("=" * 50, "green")
        cprint(
            f"Test - Mean SR: {SR_mean_test:.3f}, "
            f"Mean Return: {returns_mean_test:.3f}",
            "green"
        )
        cprint("=" * 50, "green")

        # ---- Joint Delta Plotting ----
        if self.plot_joint_deltas and len(all_pred_actions) > 0 and len(all_gt_actions) > 0:
            cprint("=" * 50, "magenta")
            cprint("Generating Joint Delta Plots", "magenta")
            cprint("=" * 50, "magenta")
            
            # Plot for first 3 episodes (or fewer if less episodes ran)
            num_plots = min(3, len(all_pred_actions), len(all_gt_actions))
            
            for i in range(num_plots):
                pred = all_pred_actions[i]
                gt = all_gt_actions[i]
                
                # Ensure same length (truncate to shorter)
                min_len = min(len(pred), len(gt))
                pred = pred[:min_len]
                gt = gt[:min_len]
                
                fig = self.plot_joint_deltas_comparison(pred, gt, episode_id=i)
                
                # Log to wandb
                log_data[f'joint_deltas_plot_ep{i}'] = wandb.Image(fig)
                plt.close(fig)
                
                # Calculate overall metrics
                joint_deltas_pred = pred[:, 6:13]
                joint_deltas_gt = gt[:, 6:13]
                
                overall_mse = np.mean((joint_deltas_pred - joint_deltas_gt)**2)
                overall_mae = np.mean(np.abs(joint_deltas_pred - joint_deltas_gt))
                
                log_data[f'joint_delta_mse_ep{i}'] = overall_mse
                log_data[f'joint_delta_mae_ep{i}'] = overall_mae
                
                cprint(f"Episode {i} - Overall Joint Delta MSE: {overall_mse:.6f}, MAE: {overall_mae:.6f}", "cyan")
            
            # Calculate average metrics across all episodes
            if len(all_pred_actions) > 0:
                all_mses = []
                all_maes = []
                
                for pred, gt in zip(all_pred_actions, all_gt_actions):
                    min_len = min(len(pred), len(gt))
                    pred = pred[:min_len]
                    gt = gt[:min_len]
                    
                    joint_deltas_pred = pred[:, 6:13]
                    joint_deltas_gt = gt[:, 6:13]
                    
                    mse = np.mean((joint_deltas_pred - joint_deltas_gt)**2)
                    mae = np.mean(np.abs(joint_deltas_pred - joint_deltas_gt))
                    
                    all_mses.append(mse)
                    all_maes.append(mae)
                
                log_data['avg_joint_delta_mse'] = np.mean(all_mses)
                log_data['avg_joint_delta_mae'] = np.mean(all_maes)
                
                cprint(f"Average across all episodes:", "green")
                cprint(f"  Joint Delta MSE: {log_data['avg_joint_delta_mse']:.6f}", "green")
                cprint(f"  Joint Delta MAE: {log_data['avg_joint_delta_mae']:.6f}", "green")

        # ---- Video logging ----
        try:
            videos_test = self.env_test.env.get_video()
            if len(videos_test.shape) == 5:
                videos_test = videos_test[:, 0]

            log_data['sim_video_test'] = wandb.Video(
                videos_test, fps=self.fps, format="mp4"
            )
            cprint("✓ Test video captured", "cyan")

        except Exception as e:
            cprint(f"⚠ Video capture failed: {e}", "yellow")

        try:
            _ = self.env_test.reset()
        except:
            pass

        return log_data
