# Dataset Setup Guide for 3D-Diffusion-Policy

This guide explains how to configure and train the 3D-Diffusion-Policy model with a new dataset.

## Overview

The training system uses Hydra for configuration management. The main components you need to configure are:
1. Dataset structure inspection
2. Task configuration file
3. Main config file (dp3.yaml)
4. Path management

---

## Step 1: Inspect Your Dataset Structure

Before configuring anything, you need to understand your dataset's structure.

### Check Dataset Contents

```bash
python3 -c "import zarr; z = zarr.open('path/to/your/dataset.zarr', 'r'); \
print('Dataset structure:'); print('Keys:', list(z.keys())); \
print('\nData keys:', list(z['data'].keys())); \
print('\nData shapes:'); \
[print(f'  {k}: {z[\"data\"][k].shape}') for k in z['data'].keys()]"
```

### Expected Dataset Format

Your zarr dataset should have the following structure:
```
dataset.zarr/
├── data/
│   ├── state          # Robot state (e.g., joint positions)
│   ├── action         # Actions to execute
│   ├── point_cloud    # Point cloud observations
│   ├── img           # (Optional) RGB images
│   └── cube_pos      # (Optional) Object positions
└── meta/
    └── episode_ends   # Episode boundary indices
```

### Key Information to Note:
- **Point cloud shape**: `(N_samples, N_points, N_features)`
  - N_features = 3 for XYZ only
  - N_features = 6 for XYZ + RGB
- **State/Action dimensions**: Must match your robot's DOF
- **Number of episodes**: From `meta/episode_ends`

---

## Step 2: Create or Modify Task Configuration

Navigate to: `3D-Diffusion-Policy/diffusion_policy_3d/config/task/`

### Option A: Copy Existing Task Config

```bash
cd 3D-Diffusion-Policy/diffusion_policy_3d/config/task/
cp pybullet_pick_place.yaml your_task_name.yaml
```

### Option B: Create From Scratch

Create a new file: `your_task_name.yaml` with the following template:

```yaml
name: your_task_name

task_name: custom_task

shape_meta: &shape_meta
  obs:
    point_cloud:
      shape: [N_POINTS, N_FEATURES]  # Update based on your dataset
      type: point_cloud
    agent_pos:
      shape: [STATE_DIM]  # Update based on your dataset
      type: low_dimx
  action:
    shape: [ACTION_DIM]  # Update based on your dataset

env_runner:
  _target_: diffusion_policy_3d.env_runner.pybullet_runner.UR5PyBulletRunner
  n_train: 20
  max_steps: 365
  n_obs_steps: ${n_obs_steps}
  n_action_steps: ${n_action_steps}
  fps: 10
  action_dim: ${eval:'${shape_meta.action.shape}[0]'}
  # Set to null if you don't have a simulation environment for evaluation

dataset:
  _target_: diffusion_policy_3d.dataset.droid_dataset.DroidDataset
  zarr_path: /absolute/path/to/your/dataset.zarr  # Use absolute path
  horizon: ${horizon}
  pad_before: ${eval:'${n_obs_steps}-1'}
  pad_after: ${eval:'${n_action_steps}-1'}
  seed: 42
  val_ratio: 0.02
  max_train_episodes: null
```

### Configuration Notes:

1. **Point Cloud Shape**:
   - If your dataset has XYZ only: `[N_POINTS, 3]`
   - If your dataset has XYZ + RGB: `[N_POINTS, 6]`

2. **Dataset Class**:
   - Use `DroidDataset` for general robot manipulation datasets
   - Use `RRCDataset` for RRC-specific data
   - Use `AdroitDataset` for Adroit hand tasks
   - Check `diffusion_policy_3d/dataset/` for other options

3. **Env Runner**:
   - Set to `null` if you only want to train without evaluation
   - Use appropriate runner if you have a simulation environment
   - Available runners: `pybullet_runner`, `adroit_runner`, `metaworld_runner`, `dexart_runner`

4. **Paths**:
   - **Always use absolute paths** for `zarr_path` to avoid confusion
   - Example: `/home/user/project/3D-Diffusion-Policy/3D-Diffusion-Policy/data/your_dataset.zarr`

---

## Step 3: Update Main Configuration (dp3.yaml)

File location: `3D-Diffusion-Policy/diffusion_policy_3d/config/dp3.yaml`

### Key Parameters to Check:

```yaml
defaults:
  - task: your_task_name  # Change this to your task config name

# Training hyperparameters
horizon: 24              # Prediction horizon
n_obs_steps: 1          # Number of observation steps
n_action_steps: 12      # Number of action steps to predict

training:
  device: "cuda:0"
  seed: 42
  num_epochs: 1000
  addition_info: "default"  # REQUIRED: Used for logging
  
logging:
  name: ${training.addition_info}-${training.seed}
  project: dp3
  mode: online  # or 'offline' to disable wandb

# Output directories - use relative paths
multi_run:
  run_dir: outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}

hydra:
  run:
    dir: outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}
```

### Common Issues:

1. **Missing `training.addition_info`**: Add this field to avoid interpolation errors
2. **Hardcoded paths**: Always use relative paths like `outputs/` instead of `/scratch2/...` or `/home/user/...`
3. **Mismatched shapes**: Ensure all shape parameters match your dataset

---

## Step 4: Path Management Best Practices

### ❌ Avoid Hardcoded Paths

```yaml
# BAD - breaks when moving between machines
zarr_path: /scratch2/cross-emb/DP3_data/dataset.zarr
run_dir: /home/user/project/outputs
```

### ✅ Use Relative or Absolute Paths Appropriately

```yaml
# GOOD - for dataset paths (must be absolute for clarity)
zarr_path: /home/cross-emb/nitin_exp/3D-Diffusion-Policy/3D-Diffusion-Policy/data/final_velocity.zarr

# GOOD - for output directories (relative to project root)
run_dir: outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}
```

### Files That May Contain Hardcoded Paths

Common locations to check and update:
- `diffusion_policy_3d/config/dp3.yaml` - Lines 139-148 (output directories)
- `diffusion_policy_3d/config/task/*.yaml` - Dataset paths
- `scripts/train_policy.sh` - Line 19 (run_dir)
- `diffusion_policy_3d/env/pybullet/*.py` - URDF paths (lines 154, 160)

### Quick Search for Hardcoded Paths

```bash
cd 3D-Diffusion-Policy
# Find all hardcoded paths
grep -r "/scratch2\|/home/[a-z]" --include="*.yaml" --include="*.py" .
```

---

## Step 5: Run Training

### Basic Training Command

```bash
cd /path/to/3D-Diffusion-Policy/3D-Diffusion-Policy

python train.py \
    --config-name=dp3 \
    task=your_task_name \
    training.device=cuda:0 \
    training.seed=0
```

### Advanced Options

```bash
python train.py \
    --config-name=dp3 \
    task=your_task_name \
    training.device=cuda:0 \
    training.seed=0 \
    training.num_epochs=500 \
    training.debug=False \
    +training.addition_info=experiment_v1 \
    exp_name=my_experiment \
    logging.mode=online \
    checkpoint.save_ckpt=True
```

### Common Training Options

| Parameter | Description | Example |
|-----------|-------------|---------|
| `task` | Task configuration to use | `task=pybullet_pick_place` |
| `training.device` | GPU device | `training.device=cuda:0` |
| `training.seed` | Random seed | `training.seed=42` |
| `training.num_epochs` | Number of epochs | `training.num_epochs=1000` |
| `training.debug` | Debug mode | `training.debug=True` |
| `+training.addition_info` | Experiment tag | `+training.addition_info=exp01` |
| `logging.mode` | W&B logging mode | `logging.mode=online` or `offline` |
| `checkpoint.save_ckpt` | Save checkpoints | `checkpoint.save_ckpt=True` |

---

## Step 6: Verify Configuration

Before starting a long training run, verify your configuration:

### 1. Check Dataset Loading

```python
python3 -c "
from omegaconf import OmegaConf
import hydra
from hydra import compose, initialize

initialize(config_path='diffusion_policy_3d/config', version_base=None)
cfg = compose(config_name='dp3', overrides=['task=your_task_name'])

print('Task name:', cfg.task_name)
print('Point cloud shape:', cfg.shape_meta.obs.point_cloud.shape)
print('Action shape:', cfg.shape_meta.action.shape)
print('Dataset path:', cfg.dataset.zarr_path)
"
```

### 2. Test Data Loading

```python
# Test if dataset loads correctly
python3 -c "
from diffusion_policy_3d.dataset.droid_dataset import DroidDataset
dataset = DroidDataset(
    zarr_path='/path/to/your/dataset.zarr',
    horizon=24,
    pad_before=0,
    pad_after=11,
    seed=42,
    val_ratio=0.02
)
print(f'Dataset size: {len(dataset)}')
sample = dataset[0]
print('Sample keys:', sample.keys())
"
```

### 3. Check for Errors

```bash
# Dry run to check for configuration errors
python train.py --config-name=dp3 task=your_task_name training.debug=True training.num_epochs=1
```

---

## Troubleshooting

### Error: `Interpolation key 'training.addition_info' not found`

**Solution**: Add `addition_info: "default"` to the `training:` section in `dp3.yaml`

```yaml
training:
  # ... other fields ...
  addition_info: "default"
```

### Error: `TypeError: run() got an unexpected keyword argument 'dataset'`

**Solution**: This happens with some env_runners (like `adroit_runner`). Either:
1. Set `env_runner: null` in your task config if you don't need evaluation
2. Use a compatible runner like `pybullet_runner`

### Error: Dataset shape mismatch

**Solution**: Verify your dataset shapes match the config:
- Point cloud: Check if it's `[N, 3]` or `[N, 6]`
- State/Action: Check dimensions match your robot

### Error: Path not found

**Solution**: 
- Always use **absolute paths** for `zarr_path`
- Use **relative paths** for output directories
- Run training from the correct directory: `3D-Diffusion-Policy/3D-Diffusion-Policy/`

---

## Example: Setting Up final_velocity.zarr

Here's a complete example of setting up the `final_velocity.zarr` dataset:

### 1. Inspect Dataset
```bash
python3 -c "import zarr; z = zarr.open('data/final_velocity.zarr', 'r'); \
print('Data shapes:'); \
[print(f'  {k}: {z[\"data\"][k].shape}') for k in z['data'].keys()]"
```

Output:
```
  action: (13140, 7)
  state: (13140, 7)
  point_cloud: (13140, 6000, 6)  # Note: 6 features (xyz + rgb)
```

### 2. Update Task Config

Edit `diffusion_policy_3d/config/task/pybullet_pick_place.yaml`:

```yaml
shape_meta: &shape_meta
  obs:
    point_cloud:
      shape: [6000, 6]  # Changed from [6000, 3]
    agent_pos:
      shape: [7]
  action:
    shape: [7]

dataset:
  _target_: diffusion_policy_3d.dataset.droid_dataset.DroidDataset
  zarr_path: /home/cross-emb/nitin_exp/3D-Diffusion-Policy/3D-Diffusion-Policy/data/final_velocity.zarr
```

### 3. Update dp3.yaml

```yaml
training:
  addition_info: "default"  # Added this line

multi_run:
  run_dir: outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}
```

### 4. Run Training

```bash
cd 3D-Diffusion-Policy
python train.py --config-name=dp3 task=pybullet_pick_place training.device=cuda:0 training.seed=0
```

---

## Summary Checklist

- [ ] Inspect dataset structure and note dimensions
- [ ] Create/modify task configuration file with correct shapes
- [ ] Update dataset path to absolute path in task config
- [ ] Verify `training.addition_info` exists in dp3.yaml
- [ ] Change any hardcoded paths to relative paths (except dataset paths)
- [ ] Test configuration with a dry run
- [ ] Start training with appropriate parameters

---

## Additional Resources

- Hydra documentation: https://hydra.cc/
- For more examples, check existing task configs in `diffusion_policy_3d/config/task/`
- Dataset classes are in `diffusion_policy_3d/dataset/`
- Environment runners are in `diffusion_policy_3d/env_runner/`
