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
                 ):
        super().__init__(output_dir)
        
        self.max_steps = max_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.fps = fps
        self.crf = crf
        self.tqdm_interval_sec = tqdm_interval_sec
        
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

    def run(self, policy: BasePolicy, dataset=None):
        """
        Run evaluation

        Args:
            policy: Policy to evaluate
            dataset: Optional dataset (not used)
        """
        device = policy.device

        all_returns_test = []
        all_success_rates_test = []

        cprint("=" * 50, "cyan")
        cprint("Running on TEST environment", "cyan")
        cprint("=" * 50, "cyan")

        for episode_id in range(self.episode_test):
            obs = self.env_test.reset()
            policy.reset()

            reward_sum = 0.0
            done = False

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

                obs, reward, done, info = self.env_test.step(action)
                reward_sum += reward
                done = np.all(done)

                if done:
                    break

            all_returns_test.append(reward_sum)
            all_success_rates_test.append(self.env_test.env.is_success())

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
