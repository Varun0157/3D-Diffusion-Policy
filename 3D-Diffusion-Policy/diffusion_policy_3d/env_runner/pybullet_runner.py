import wandb
import numpy as np
import torch
import collections
import tqdm
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
    Runner for UR5 Pick-and-Place task in PyBullet
    
    This runner wraps the base UR5PickPlaceEnv with MultiStepWrapper, which:
    1. Stacks multiple observations together (n_obs_steps)
    2. Executes multiple actions per policy prediction (n_action_steps)
    3. Aggregates rewards over the action sequence
    
    The MultiStepWrapper is compatible with both Gym and PyBullet environments
    that follow the gym.Env interface.
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
            """
            Creates environment with MultiStepWrapper
            
            MultiStepWrapper transforms:
            - Single obs (dict with arrays) -> Stacked obs (dict with arrays of shape (n_obs_steps, ...))
            - Single action (13,) -> Multi-step action (n_action_steps, 13)
            
            This wrapper works for ANY gym.Env, including PyBullet environments!
            """
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
        
        # Create train and test environments
        self.env_train = env_fn()
        self.env_test = env_fn()
        
        self.episode_train = n_train
        self.episode_test = n_test
        
        # Logging utilities
        self.logger_util_train = logger_util.LargestKRecorder(K=3)
        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        
    def run(self, policy: BasePolicy):
        """
        Run evaluation on both train and test environments
        """
        import time  # For timing
        
        device = policy.device
        dtype = policy.dtype
        
        all_returns_train = []
        all_success_rates_train = []
        all_returns_test = []
        all_success_rates_test = []
        
        ##############################
        # Train env loop
        ##############################
        cprint("="*50, "cyan")
        cprint("Running on TRAIN environment", "cyan")
        cprint("="*50, "cyan")
        
        for episode_id in tqdm.tqdm(
            range(self.episode_train), 
            desc=f"UR5 PyBullet Train Env",
            leave=False, 
            mininterval=self.tqdm_interval_sec
        ):
            episode_start = time.time()
            
            # Reset environment
            reset_start = time.time()
            obs = self.env_train.reset()
            reset_time = time.time() - reset_start
            print(f"[Episode {episode_id}] Reset time: {reset_time:.2f}s")
            
            policy.reset()
            
            done = False
            reward_sum = 0.
            
            step_times = []
            policy_times = []
            env_step_times = []
            
            for step_id in range(self.max_steps):
                step_start = time.time()
                
                # Prepare observation
                prep_start = time.time()
                np_obs_dict = dict(obs)
                
                # Device transfer
                obs_dict = dict_apply(
                    np_obs_dict,
                    lambda x: torch.from_numpy(x).to(device=device)
                )
                prep_time = time.time() - prep_start
                
                # Run policy
                policy_start = time.time()
                with torch.no_grad():
                    obs_dict_input = {}
                    obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)
                    obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)
                    
                    action_dict = policy.predict_action(obs_dict_input)
                policy_time = time.time() - policy_start
                policy_times.append(policy_time)
                
                # Convert action to numpy
                convert_start = time.time()
                np_action_dict = dict_apply(
                    action_dict,
                    lambda x: x.detach().to('cpu').numpy()
                )
                
                action = np_action_dict['action'].squeeze(0)
                convert_time = time.time() - convert_start
                # Step 0: total=17.187s, prep=0.000s, policy=0.089s, convert=0.000s, env_step=17.097s
                # Step environment
                env_step_start = time.time()
                obs, reward, done, info = self.env_train.step(action)
                env_step_time = time.time() - env_step_start
                env_step_times.append(env_step_time)
                
                reward_sum += reward
                done = np.all(done)
                
                step_time = time.time() - step_start
                step_times.append(step_time)
                
                # Print every 10 steps
                if step_id % 10 == 0:
                    print(f"  Step {step_id}: total={step_time:.3f}s, "
                          f"prep={prep_time:.3f}s, policy={policy_time:.3f}s, "
                          f"convert={convert_time:.3f}s, env_step={env_step_time:.3f}s")
                
                if done:
                    break
            
            episode_time = time.time() - episode_start
            
            all_returns_train.append(reward_sum)
            all_success_rates_train.append(self.env_train.env.is_success())
            
            print(f"[Episode {episode_id}] Total episode time: {episode_time:.2f}s")
            print(f"  Avg step time: {np.mean(step_times):.3f}s")
            print(f"  Avg policy time: {np.mean(policy_times):.3f}s")
            print(f"  Avg env step time: {np.mean(env_step_times):.3f}s")
            print(f"  Num steps: {len(step_times)}")
            cprint(f"Train Episode {episode_id}: Reward={reward_sum:.2f}, Success={self.env_train.env.is_success()}", "yellow")
            print("-"*50)
        
        ##############################
        # Test env loop
        ##############################
        cprint("="*50, "cyan")
        cprint("Running on TEST environment", "cyan")
        cprint("="*50, "cyan")
        
        for episode_id in tqdm.tqdm(
            range(self.episode_test), 
            desc=f"UR5 PyBullet Test Env",
            leave=False, 
            mininterval=self.tqdm_interval_sec
        ):
            # Reset environment
            obs = self.env_test.reset()
            policy.reset()
            
            done = False
            reward_sum = 0.
            
            for step_id in range(self.max_steps):
                # Prepare observation
                np_obs_dict = dict(obs)
                
                # Device transfer
                obs_dict = dict_apply(
                    np_obs_dict,
                    lambda x: torch.from_numpy(x).to(device=device)
                )
                
                # Run policy
                with torch.no_grad():
                    obs_dict_input = {}
                    obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)
                    obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)
                    
                    action_dict = policy.predict_action(obs_dict_input)
                
                # Convert action to numpy
                np_action_dict = dict_apply(
                    action_dict,
                    lambda x: x.detach().to('cpu').numpy()
                )
                
                action = np_action_dict['action'].squeeze(0)
                
                # Step environment
                obs, reward, done, info = self.env_test.step(action)
                reward_sum += reward
                done = np.all(done)
                
                if done:
                    break
            
            all_returns_test.append(reward_sum)
            all_success_rates_test.append(self.env_test.env.is_success())
            
            cprint(f"Test Episode {episode_id}: Reward={reward_sum:.2f}, Success={self.env_test.env.is_success()}", "yellow")
        
        ##############################
        # Compute metrics
        ##############################
        SR_mean_train = np.mean(all_success_rates_train)
        returns_mean_train = np.mean(all_returns_train)
        SR_mean_test = np.mean(all_success_rates_test)
        returns_mean_test = np.mean(all_returns_test)
        
        # Update loggers
        self.logger_util_train.record(SR_mean_train)
        self.logger_util_test.record(SR_mean_test)
        
        # Prepare log data
        log_data = dict()
        log_data['mean_success_rates_train'] = SR_mean_train
        log_data['mean_returns_train'] = returns_mean_train
        log_data['mean_success_rates_test'] = SR_mean_test
        log_data['mean_returns_test'] = returns_mean_test
        
        log_data['SR_train_L3'] = self.logger_util_train.average_of_largest_K()
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        
        # Main test metric
        log_data['test_mean_score'] = SR_mean_test
        
        # Print summary
        cprint("="*50, "green")
        cprint(f"Train - Mean SR: {SR_mean_train:.3f}, Mean Return: {returns_mean_train:.3f}", "green")
        cprint(f"Test  - Mean SR: {SR_mean_test:.3f}, Mean Return: {returns_mean_test:.3f}", "green")
        cprint("="*50, "green")
        
        # Get videos and create WandB Video objects
        try:
            videos_train = self.env_train.env.get_video()
            videos_test = self.env_test.env.get_video()
            
            # Handle video shape (remove extra dimension if needed)
            if len(videos_train.shape) == 5:
                videos_train = videos_train[:, 0]
            if len(videos_test.shape) == 5:
                videos_test = videos_test[:, 0]
            
            # Create WandB video objects
            sim_video_train = wandb.Video(videos_train, fps=self.fps, format="mp4")
            sim_video_test = wandb.Video(videos_test, fps=self.fps, format="mp4")
            
            log_data['sim_video_train'] = sim_video_train
            log_data['sim_video_test'] = sim_video_test
            
            cprint("✓ Videos captured and ready for WandB logging", "cyan")
            
        except Exception as e:
            cprint(f"⚠ Warning: Could not capture videos: {e}", "yellow")
            # Continue without videos - training metrics are more important
        
        # Clear video buffers
        try:
            _ = self.env_train.reset()
            _ = self.env_test.reset()
        except:
            pass
        
        return log_data