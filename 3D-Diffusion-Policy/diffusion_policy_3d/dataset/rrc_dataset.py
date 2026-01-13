from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset

class RRCDataset(BaseDataset):
    def __init__(self,
            zarr_path,
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            task_name=None,
            use_cumulative_action=False,
            ):
        super().__init__()
        self.task_name = task_name
        self.use_cumulative_action = use_cumulative_action
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['state', 'action', 'point_cloud'])
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        action_data = self.replay_buffer['action']
        if self.use_cumulative_action:
            # Fit on cumulative actions by sampling chunks
            sampled_actions = []
            n_samples = min(2000, len(self.sampler))
            indices = np.random.choice(len(self.sampler), n_samples, replace=False)
            for idx in indices:
                sample = self.sampler.sample_sequence(idx)
                action = sample['action'].astype(np.float32)
                # Apply cumsum as in _sample_to_data
                joints = action[:, :-1]
                gripper = action[:, -1:]
                joints_cumsum = np.cumsum(joints, axis=0)
                action_cumsum = np.concatenate([joints_cumsum, gripper], axis=-1)
                sampled_actions.append(action_cumsum)
            action_data = np.concatenate(sampled_actions, axis=0)

        data = {
            'action': action_data,
            'agent_pos': self.replay_buffer['state'][...,:],
            'point_cloud': self.replay_buffer['point_cloud'],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        # normalizer['point_cloud'] = SingleFieldLinearNormalizer.create_identity()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['state'][:,].astype(np.float32) # (T, 13)
        point_cloud = sample['point_cloud'][:,].astype(np.float32) # (T, 2500, 6)
        action = sample['action'].astype(np.float32) # T, 13

        if self.use_cumulative_action:
            # joint_state is all but last dim (gripper)
            joints = action[:, :-1]
            gripper = action[:, -1:]
            # Cumulative sum along time dimension (T)
            joints_cumsum = np.cumsum(joints, axis=0)
            action = np.concatenate([joints_cumsum, gripper], axis=-1)

        data = {
            'obs': {
                'point_cloud': point_cloud, # T, 2500, 6
                'agent_pos': agent_pos, # T, 13
            },
            'action': action # T, 13
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
