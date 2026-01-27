# Action Normalization Logging

This document explains the normalization logging feature that has been added to the training pipeline.

## Overview

During training, actions are normalized before being used in the diffusion model. This logging feature helps you understand and debug the normalization process by saving detailed statistics about actions before and after normalization.

## What Gets Logged

### 1. Normalizer Statistics (at the start of training)
- **Input statistics**: Min, max, mean, std of the raw data used to fit the normalizer
- **Statistics for all fields**: `action`, `agent_pos`, `point_cloud`

### 2. Per-Batch Action Data (first batch of each epoch)
- **Before Normalization (Raw Actions)**:
  - Shape of the action tensor
  - Min/Max values per dimension
  - Mean values per dimension
  - Standard deviation per dimension
  - First sample showing first 3 timesteps
  
- **After Normalization (Normalized Actions)**:
  - Same statistics as above but for normalized values
  - First sample showing how normalization transformed the data

## Log File Location

Logs are saved in: `outputs/<date>/<time>_<experiment>/normalization_logs/`

File format: `action_normalization_epoch_<epoch_number>.log`

Example:
```
outputs/2026.01.27/12.40.58_train_dp3_pick_place/normalization_logs/action_normalization_epoch_0.log
```

## Code Implementation

### Location
The logging code is implemented in [train.py](3D-Diffusion-Policy/train.py):

1. **Lines 127-155**: Creates log directory and writes normalizer statistics
2. **Lines 239-265**: Logs first batch actions before/after normalization

### How It Works

```python
# 1. After creating the normalizer, log its statistics
normalizer = dataset.get_normalizer()
norm_log_dir = pathlib.Path(self.output_dir) / "normalization_logs"
norm_log_dir.mkdir(parents=True, exist_ok=True)
norm_log_file = norm_log_dir / f"action_normalization_epoch_{self.epoch}.log"

# Log normalizer input stats
input_stats = normalizer.get_input_stats()
# ... write to file ...

# 2. In training loop, log first batch
if batch_idx == 0:
    # Raw actions (before normalization)
    raw_actions = batch['action'].cpu().numpy()
    
    # Normalized actions (after normalization)
    normalized_actions = normalizer['action'].normalize(batch['action']).cpu().numpy()
    
    # Write statistics to log file
```

## Understanding the Normalization Process

### Normalization Types

The dataset uses `LinearNormalizer` which can operate in different modes:

1. **'limits' mode** (default): Normalizes to [-1, 1] range
   - Formula: `normalized = 2 * (x - min) / (max - min) - 1`

2. **'gaussian' mode**: Standardizes to zero mean, unit variance
   - Formula: `normalized = (x - mean) / std`

### What to Look For

1. **Check Raw Action Range**:
   - Are actions in the expected range?
   - Any suspicious outliers?

2. **Check Normalized Range**:
   - Should typically be in [-1, 1] for limits mode
   - Should have mean≈0, std≈1 for gaussian mode

3. **Check Per-Dimension Statistics**:
   - Are some dimensions always near zero? (might indicate unused actions)
   - Are some dimensions saturated at min/max? (might indicate clipping)

## Example Log Output

```
Normalization Statistics - Epoch 0
Timestamp: 2026-01-27 12:40:58.123456
================================================================================

Normalizer Input Stats (Raw Data Statistics):

action:
  min: shape=(7,), values=[-1. -1. -1. -1. -1. -1. -1.]
  max: shape=(7,), values=[1. 1. 1. 1. 1. 1. 1.]
  mean: shape=(7,), values=[0.1234 -0.0567 0.2345 ...]
  std: shape=(7,), values=[0.4567 0.3456 0.5678 ...]

agent_pos:
  min: shape=(7,), values=[-2.22 -1.55 0.00 ...]
  max: shape=(7,), values=[2.45 1.23 0.87 ...]

================================================================================

Batch 0 - Action Normalization Details:
================================================================================

RAW ACTIONS (Before Normalization):
  Shape: (16, 24, 7)
  Min: [-0.9876 -0.8765 -0.7654 -0.6543 -0.5432 -0.4321 -0.3210]
  Max: [0.9876 0.8765 0.7654 0.6543 0.5432 0.4321 0.3210]
  Mean: [0.0123 -0.0234 0.0345 -0.0456 0.0567 -0.0678 0.0789]
  Std: [0.4567 0.3456 0.5678 0.2345 0.6789 0.1234 0.8901]

  First sample (first 3 timesteps):
[[ 0.123 -0.234  0.345 -0.456  0.567 -0.678  0.789]
 [ 0.234 -0.345  0.456 -0.567  0.678 -0.789  0.890]
 [ 0.345 -0.456  0.567 -0.678  0.789 -0.890  0.901]]

NORMALIZED ACTIONS (After Normalization):
  Shape: (16, 24, 7)
  Min: [-0.9999 -0.9876 -0.9753 -0.9630 -0.9507 -0.9384 -0.9261]
  Max: [0.9999 0.9876 0.9753 0.9630 0.9507 0.9384 0.9261]
  Mean: [0.0012 -0.0023 0.0034 -0.0045 0.0056 -0.0067 0.0078]
  Std: [0.5123 0.4567 0.6234 0.3456 0.7345 0.2345 0.8456]

  First sample (first 3 timesteps):
[[ 0.246 -0.468  0.690 -0.912  0.134 -0.356  0.578]
 [ 0.468 -0.690  0.912 -0.134  0.356 -0.578  0.790]
 [ 0.690 -0.912  0.134 -0.356  0.578 -0.790  0.902]]

================================================================================
```

## Debugging Tips

### Issue: Actions not in expected range after normalization

**Possible Causes:**
1. Dataset has outliers affecting min/max
2. Normalization mode mismatch
3. Data not loaded correctly

**Solution:**
- Check raw action statistics in the log
- Verify dataset preprocessing
- Consider using robust normalization (ignore outliers)

### Issue: Normalized actions have very small range

**Possible Causes:**
1. Actions are already normalized in the dataset
2. Very small variation in training data
3. Incorrect normalization parameters

**Solution:**
- Check if dataset actions are already in [-1, 1]
- Increase data diversity
- Verify normalizer fitting process

### Issue: Some dimensions always near zero

**Possible Causes:**
1. Unused action dimensions in the dataset
2. Robot doesn't use certain DOFs
3. Gripper state included but constant

**Solution:**
- This might be expected behavior
- Consider masking unused dimensions
- Check if dimensions correspond to actual robot capabilities

## Customization

### Change Logging Frequency

To log every N batches instead of just the first batch:

```python
# In train.py, line 239
if batch_idx % N == 0:  # Change from batch_idx == 0
    # ... logging code ...
```

### Log Additional Fields

To log observations or other fields:

```python
# After normalizing actions, add:
raw_obs = batch['obs']['agent_pos'].cpu().numpy()
normalized_obs = normalizer['agent_pos'].normalize(batch['obs']['agent_pos']).cpu().numpy()
f.write(f"\nRAW OBSERVATIONS:\n")
f.write(f"  Mean: {raw_obs.mean(axis=(0,1))}\n")
# ... etc
```

### Change Normalization Mode

In your dataset configuration (e.g., `pybullet_pick_place.yaml`), you can't directly change it, but you can modify the dataset class to accept a parameter:

```python
# In droid_dataset.py, line 64
def get_normalizer(self, mode='limits', **kwargs):
    # Change 'limits' to 'gaussian' for standardization
```

## Related Files

- [train.py](3D-Diffusion-Policy/train.py) - Main training loop with logging
- [normalizer.py](3D-Diffusion-Policy/diffusion_policy_3d/model/common/normalizer.py) - Normalization implementation
- [droid_dataset.py](3D-Diffusion-Policy/diffusion_policy_3d/dataset/droid_dataset.py) - Dataset with normalizer creation
- [dp3.py](3D-Diffusion-Policy/diffusion_policy_3d/policy/dp3.py) - Policy that uses normalized actions

## Performance Impact

The logging has minimal performance impact:
- Only logs the first batch of each epoch
- File I/O is done on CPU while GPU continues training
- Typical overhead: < 0.1 seconds per epoch

You can disable logging by commenting out lines 239-265 in train.py if needed.
