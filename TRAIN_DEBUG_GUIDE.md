# Understanding and Debugging train.py

## Overview

`train.py` is the main training script for the 3D Diffusion Policy (DP3) project. It handles the complete training pipeline including:
- Dataset loading and preprocessing
- Model initialization (DP3 policy)
- Training loop with validation
- Checkpoint management
- Logging with WandB
- Evaluation of trained policies

## File Location
```
3D-Diffusion-Policy/3D-Diffusion-Policy/train.py
```

## Key Components

### 1. TrainDP3Workspace Class
The main class that encapsulates the entire training workflow.

**Important Methods:**
- `__init__()`: Initializes the model, optimizer, and training state
- `run()`: Main training loop
- `eval()`: Evaluation mode (currently has two implementations commented/active)
- `save_checkpoint()`: Saves model checkpoints
- `load_checkpoint()`: Loads model checkpoints for resuming training

### 2. Main Configuration
The script uses **Hydra** for configuration management:
- Config files are located in: `3D-Diffusion-Policy/diffusion_policy_3d/config/`
- Main configs: `dp3.yaml`, `simple_dp3.yaml`
- Task-specific configs in: `config/task/`

### 3. Key Dependencies
```python
- hydra: Configuration management
- wandb: Experiment tracking and logging
- torch: Deep learning framework
- omegaconf: Configuration handling
- diffusion_policy_3d: Custom policy implementation
```

## Script Arguments

### Using Hydra Configuration

The script uses Hydra's decorator `@hydra.main()` which means arguments are passed via config files or command-line overrides.

**Common configuration overrides:**
```bash
# Override training device
python train.py training.device=cuda:0

# Override number of epochs
python train.py training.num_epochs=1000

# Override batch size
python train.py dataloader.batch_size=64

# Override task
python train.py task=adroit_hammer

# Override policy type
python train.py policy=dp3

# Override multiple parameters
python train.py training.device=cuda:0 training.num_epochs=500 task=adroit_hammer
```

### Important Configuration Parameters

**Training Parameters:**
- `training.seed`: Random seed (default varies)
- `training.device`: Device to use (e.g., 'cuda:0')
- `training.num_epochs`: Number of training epochs
- `training.use_ema`: Whether to use Exponential Moving Average
- `training.resume`: Resume from checkpoint
- `training.debug`: Enable debug mode

**Data Parameters:**
- `dataloader.batch_size`: Batch size for training
- `dataloader.num_workers`: Number of data loading workers
- `task.dataset`: Dataset configuration

**Checkpoint Parameters:**
- `checkpoint.topk.k`: Number of top checkpoints to keep
- `checkpoint.topk.mode`: 'max' or 'min' for checkpoint selection

## Debugging Setup

### Method 1: Using VS Code Python Debugger

#### Step 1: Create Launch Configuration

Create or edit `.vscode/launch.json` in your workspace root:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Train DP3",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/3D-Diffusion-Policy/3D-Diffusion-Policy/train.py",
            "console": "integratedTerminal",
            "justMyCode": false,
            "cwd": "${workspaceFolder}",
            "args": [
                "task=adroit_hammer",
                "training.device=cuda:0",
                "training.num_epochs=10"
            ],
            "env": {
                "PYTHONPATH": "${workspaceFolder}"
            }
        },
        {
            "name": "Python: Train DP3 (Debug Mode)",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/3D-Diffusion-Policy/3D-Diffusion-Policy/train.py",
            "console": "integratedTerminal",
            "justMyCode": false,
            "cwd": "${workspaceFolder}",
            "args": [
                "task=adroit_hammer",
                "training.device=cuda:0",
                "training.debug=true",
                "training.num_epochs=2"
            ],
            "env": {
                "PYTHONPATH": "${workspaceFolder}",
                "WANDB_MODE": "offline"
            }
        },
        {
            "name": "Python: Eval DP3",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/3D-Diffusion-Policy/3D-Diffusion-Policy/train.py",
            "console": "integratedTerminal",
            "justMyCode": false,
            "cwd": "${workspaceFolder}",
            "args": [
                "task=adroit_hammer",
                "training.device=cuda:0",
                "mode=eval"
            ],
            "env": {
                "PYTHONPATH": "${workspaceFolder}"
            }
        }
    ]
}
```

#### Step 2: Set Breakpoints
1. Open `train.py` in VS Code
2. Click on the left margin next to line numbers to set breakpoints
3. Common breakpoint locations:
   - Line ~198: Start of training loop
   - Line ~560: Start of eval method
   - Line ~90: Model initialization
   - Line ~310: Inside epoch training loop

#### Step 3: Start Debugging
1. Press `F5` or go to "Run and Debug" panel (Ctrl+Shift+D)
2. Select "Python: Train DP3" from the dropdown
3. Click the green play button
4. The debugger will stop at your breakpoints

### Method 2: Using pdb (Python Debugger)

#### Add breakpoint in code:
```python
# Add this line where you want to stop
import pdb; pdb.set_trace()
```

#### Run with arguments:
```bash
cd /home/cross-emb/nitin_exp/3D-Diffusion-Policy/3D-Diffusion-Policy
python train.py task=adroit_hammer training.device=cuda:0
```

**pdb Commands:**
- `n` (next): Execute next line
- `s` (step): Step into function
- `c` (continue): Continue execution
- `p variable_name`: Print variable
- `l` (list): Show current location in code
- `h` (help): Show help
- `q` (quit): Exit debugger

### Method 3: Using debugpy (Remote Debugging)

#### Add to train.py (at the top of main or run method):
```python
import debugpy
debugpy.listen(5678)
print("Waiting for debugger attach...")
debugpy.wait_for_client()
```

#### In VS Code, create attach configuration:
```json
{
    "name": "Python: Attach",
    "type": "debugpy",
    "request": "attach",
    "connect": {
        "host": "localhost",
        "port": 5678
    }
}
```

### Method 4: Using ipdb (Enhanced pdb)

```bash
pip install ipdb
```

```python
# Add this line in code
import ipdb; ipdb.set_trace()
```

## Common Debugging Scenarios

### 1. Debug Data Loading
Set breakpoint at line ~117 (dataset initialization):
```python
dataset: BaseDataset
dataset = hydra.utils.instantiate(cfg.task.dataset)
```

### 2. Debug Model Forward Pass
Set breakpoint inside the training loop where model is called (around line 220-250).

### 3. Debug Checkpoint Loading
Set breakpoint at line ~560 in the `eval()` method:
```python
self.load_checkpoint(path=lastest_ckpt_path)
```

### 4. Debug Environment Runner
Set breakpoint around line ~155:
```python
env_runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=self.output_dir)
```

### 5. Debug Action Normalization (Before/After)

Understanding how actions are normalized is crucial for debugging policy behavior. Here's how to observe the normalization process:

#### Location of Normalizer

The normalizer is typically set in two places:
1. **Training**: Around line ~145 in `train.py`
   ```python
   normalizer = dataset.get_normalizer()
   self.model.set_normalizer(normalizer)
   ```

2. **Inside the Policy**: The policy uses the normalizer in its forward pass to normalize/denormalize data

#### Method 1: Add Logging to See Before/After Normalization

Add this code snippet in the training loop (around line 220-250):

```python
# Inside the training loop, after getting batch data
batch = next(train_dataloader_iter)
batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))

# ===== ADD THIS DEBUG CODE =====
if global_step % 100 == 0:  # Log every 100 steps
    print("\n" + "="*60)
    print(f"DEBUG: Action Normalization at step {global_step}")
    print("="*60)
    
    # Get raw action from batch
    raw_action = batch['action']  # Shape: [batch_size, horizon, action_dim]
    print(f"Raw Action Shape: {raw_action.shape}")
    print(f"Raw Action [0,0,:] (first timestep of first batch):")
    print(f"  Values: {raw_action[0, 0, :].cpu().numpy()}")
    print(f"  Min: {raw_action[0, 0, :].min().item():.4f}")
    print(f"  Max: {raw_action[0, 0, :].max().item():.4f}")
    print(f"  Mean: {raw_action[0, 0, :].mean().item():.4f}")
    print(f"  Std: {raw_action[0, 0, :].std().item():.4f}")
    
    # Get normalizer stats
    if 'action' in self.model.normalizer.params_dict:
        action_stats = self.model.normalizer.params_dict['action']
        print(f"\nNormalizer Stats for 'action':")
        print(f"  Mean: {action_stats['mean']}")
        print(f"  Std: {action_stats['std']}")
        print(f"  Min: {action_stats.get('min', 'N/A')}")
        print(f"  Max: {action_stats.get('max', 'N/A')}")
    
    # Manually normalize to see the result
    normalized_action = self.model.normalizer.normalize(raw_action, key='action')
    print(f"\nNormalized Action [0,0,:] (after normalization):")
    print(f"  Values: {normalized_action[0, 0, :].cpu().numpy()}")
    print(f"  Min: {normalized_action[0, 0, :].min().item():.4f}")
    print(f"  Max: {normalized_action[0, 0, :].max().item():.4f}")
    print(f"  Mean: {normalized_action[0, 0, :].mean().item():.4f}")
    print(f"  Std: {normalized_action[0, 0, :].std().item():.4f}")
    
    # Denormalize to verify it matches original
    denormalized_action = self.model.normalizer.unnormalize(normalized_action, key='action')
    print(f"\nDenormalized Action [0,0,:] (should match raw):")
    print(f"  Values: {denormalized_action[0, 0, :].cpu().numpy()}")
    reconstruction_error = (raw_action[0, 0, :] - denormalized_action[0, 0, :]).abs().mean()
    print(f"  Reconstruction Error: {reconstruction_error.item():.6f}")
    print("="*60 + "\n")
# ===== END DEBUG CODE =====

# Continue with normal training
loss = self.model.compute_loss(batch)
```

#### Method 2: Using Breakpoints to Inspect Normalization

1. **Set breakpoint in the dataset's `get_normalizer()` method**:
   - File: `3D-Diffusion-Policy/diffusion_policy_3d/dataset/base_dataset.py` (or specific dataset file)
   - This shows how normalizer statistics are computed

2. **Set breakpoint in the policy's forward pass**:
   - File: `3D-Diffusion-Policy/diffusion_policy_3d/policy/dp3.py`
   - Look for `normalize()` or `unnormalize()` calls
   - Inspect `naction` (normalized action) vs raw action

3. **Set breakpoint in normalizer class**:
   - File: `3D-Diffusion-Policy/diffusion_policy_3d/common/normalize_util.py` (or similar)
   - Methods: `normalize()`, `unnormalize()`

#### Method 3: Add Custom Logging Function

Create a helper function to log normalization details:

```python
def debug_normalization(data, normalizer, key, step, name=""):
    """Debug helper to log normalization process"""
    import numpy as np
    
    print(f"\n{'='*70}")
    print(f"Normalization Debug - {name} (Step {step})")
    print(f"{'='*70}")
    
    # Original data
    if isinstance(data, torch.Tensor):
        data_np = data[0, 0, :].detach().cpu().numpy()  # First batch, first timestep
    else:
        data_np = data[0, 0, :]
    
    print(f"BEFORE Normalization ({key}):")
    print(f"  Shape: {data.shape}")
    print(f"  Sample values: {data_np}")
    print(f"  Statistics:")
    print(f"    Min:  {data_np.min():.6f}")
    print(f"    Max:  {data_np.max():.6f}")
    print(f"    Mean: {data_np.mean():.6f}")
    print(f"    Std:  {data_np.std():.6f}")
    
    # Normalizer parameters
    if key in normalizer.params_dict:
        params = normalizer.params_dict[key]
        print(f"\nNormalizer Parameters ({key}):")
        for param_key, param_val in params.items():
            if isinstance(param_val, torch.Tensor):
                print(f"  {param_key}: {param_val.cpu().numpy()}")
            else:
                print(f"  {param_key}: {param_val}")
    
    # Normalized data
    normalized = normalizer.normalize(data, key=key)
    if isinstance(normalized, torch.Tensor):
        norm_np = normalized[0, 0, :].detach().cpu().numpy()
    else:
        norm_np = normalized[0, 0, :]
    
    print(f"\nAFTER Normalization:")
    print(f"  Sample values: {norm_np}")
    print(f"  Statistics:")
    print(f"    Min:  {norm_np.min():.6f}")
    print(f"    Max:  {norm_np.max():.6f}")
    print(f"    Mean: {norm_np.mean():.6f}")
    print(f"    Std:  {norm_np.std():.6f}")
    
    # Verify denormalization
    denormalized = normalizer.unnormalize(normalized, key=key)
    if isinstance(denormalized, torch.Tensor):
        denorm_np = denormalized[0, 0, :].detach().cpu().numpy()
    else:
        denorm_np = denormalized[0, 0, :]
    
    reconstruction_error = np.abs(data_np - denorm_np).mean()
    print(f"\nDenormalization Check:")
    print(f"  Reconstruction error: {reconstruction_error:.8f}")
    print(f"  {'✓ PASS' if reconstruction_error < 1e-5 else '✗ FAIL'}")
    print(f"{'='*70}\n")

# Usage in training loop:
debug_normalization(batch['action'], self.model.normalizer, 'action', global_step, "Training Action")
```

#### Method 4: Save Normalization Data to File

For detailed analysis, save the data to a file:

```python
import json
import numpy as np

# In training loop
if global_step == 0:  # First step only
    debug_data = {
        'step': global_step,
        'raw_action': batch['action'][0].cpu().numpy().tolist(),
        'normalized_action': self.model.normalizer.normalize(
            batch['action'], key='action'
        )[0].cpu().numpy().tolist(),
        'normalizer_params': {
            k: {pk: pv.cpu().numpy().tolist() if isinstance(pv, torch.Tensor) else pv
                for pk, pv in v.items()}
            for k, v in self.model.normalizer.params_dict.items()
        }
    }
    
    with open('normalization_debug.json', 'w') as f:
        json.dump(debug_data, f, indent=2)
    
    print("Saved normalization debug data to normalization_debug.json")
```

#### VS Code Debugger Configuration for Normalization

Add this specialized debug configuration to `.vscode/launch.json`:

```json
{
    "name": "Python: Debug Normalization",
    "type": "debugpy",
    "request": "launch",
    "program": "${workspaceFolder}/3D-Diffusion-Policy/3D-Diffusion-Policy/train.py",
    "console": "integratedTerminal",
    "justMyCode": false,
    "cwd": "${workspaceFolder}",
    "args": [
        "task=adroit_hammer",
        "training.device=cuda:0",
        "training.num_epochs=1",
        "training.debug=true"
    ],
    "env": {
        "PYTHONPATH": "${workspaceFolder}",
        "WANDB_MODE": "offline"
    },
    "stopOnEntry": false,
    "logToFile": true
}
```

**Set breakpoints at:**
1. Line ~145: Where `normalizer` is set
2. Line ~220-250: Inside training loop after getting batch
3. Inside `diffusion_policy_3d/policy/dp3.py` where actions are normalized

#### Expected Output Example

```
==================================================================
Normalization Debug - Training Action (Step 0)
==================================================================
BEFORE Normalization (action):
  Shape: torch.Size([64, 16, 22])
  Sample values: [ 0.234  -0.123   0.456   0.789  -0.234  ...]
  Statistics:
    Min:  -0.856234
    Max:   0.923451
    Mean:  0.045123
    Std:   0.234567

Normalizer Parameters (action):
  mean: [ 0.012  -0.034   0.056  ...]
  std:  [ 0.234   0.345   0.456  ...]
  
AFTER Normalization:
  Sample values: [ 0.948  -0.258   0.877   1.234  -0.543  ...]
  Statistics:
    Min:  -3.245123
    Max:   3.876543
    Mean:  0.001234
    Std:   1.002345

Denormalization Check:
  Reconstruction error: 0.00000012
  ✓ PASS
==================================================================
```

#### Key Things to Check

1. **Normalized data should be roughly zero-mean, unit-variance** (mean ≈ 0, std ≈ 1)
2. **Reconstruction error should be very small** (< 1e-5)
3. **Normalizer params should match dataset statistics**
4. **No NaN or Inf values** after normalization
5. **Action dimensions should match** expected action space size

## Logging and Monitoring

### WandB Integration
- Training logs are automatically sent to WandB
- To debug offline: Set environment variable `WANDB_MODE=offline`
- To disable WandB: Set `WANDB_MODE=disabled`

### Output Directory
- Default: `./outputs/YYYY-MM-DD/HH-MM-SS/`
- Checkpoints saved in: `./outputs/.../checkpoints/`
- Logs saved in: `./outputs/.../logs.json.txt`

## Troubleshooting

### Issue 1: CUDA Out of Memory
**Solutions:**
- Reduce batch size: `dataloader.batch_size=32`
- Reduce number of workers: `dataloader.num_workers=2`
- Use gradient accumulation: `training.gradient_accumulate_every=2`

### Issue 2: Hydra Config Not Found
**Solutions:**
- Ensure you're in the correct directory
- Check config path is correct
- Verify config files exist in `diffusion_policy_3d/config/`

### Issue 3: Dataset Not Found
**Solutions:**
- Generate demonstrations first using scripts in `scripts/`
- Check data path in task config file
- Verify zarr files exist in `3D-Diffusion-Policy/data/`

## Example Debugging Workflow

1. **Start with small dataset:**
   ```bash
   python train.py task=adroit_hammer training.num_epochs=2 training.debug=true
   ```

2. **Add breakpoint at model initialization** (line ~70)

3. **Step through to verify:**
   - Config is loaded correctly
   - Dataset is initialized
   - Model architecture is correct
   - Normalizer is set properly

4. **Continue to training loop** and verify:
   - Batch shapes are correct
   - Loss is computed properly
   - Gradients are reasonable

5. **Check validation** works correctly

## Quick Reference Commands

```bash
# Basic training run
python train.py task=adroit_hammer

# Training with custom args
python train.py task=adroit_hammer training.device=cuda:0 training.num_epochs=100

# Debug mode (faster, less data)
python train.py task=adroit_hammer training.debug=true

# Resume from checkpoint
python train.py task=adroit_hammer training.resume=true

# Evaluation mode (requires checkpoint)
python train.py task=adroit_hammer mode=eval
```

## Additional Resources

- **Hydra Documentation**: https://hydra.cc/docs/intro/
- **WandB Documentation**: https://docs.wandb.ai/
- **VS Code Python Debugging**: https://code.visualstudio.com/docs/python/debugging
- **Project README**: See main README.md for setup and usage

## Notes

- The `eval()` method in the current file has two implementations (one commented out, one active)
- The checkpoint path in `eval()` is hardcoded and may need adjustment
- Gripper action discretization is implemented but may not be used depending on task
- EMA (Exponential Moving Average) model is optional but recommended for better performance
