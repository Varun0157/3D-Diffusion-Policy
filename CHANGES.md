# Changes: Joint Deltas → Absolute Joint Positions

**Date:** January 31, 2026

**Files Modified:** 
- `3D-Diffusion-Policy/diffusion_policy_3d/env/pybullet/pybullet_wrapper.py`

---

## Summary

Changed PyBullet UR5 environment from **delta-based** to **absolute joint position control**.

**Before:** Actions = joint deltas `[Δθ₁...Δθ₆]`, range `[-0.1, 0.1]`  
**After:** Actions = absolute positions `[θ₁...θ₆]`, range `[-π, π]`

---

## Code Changes

### 1. Action Space Definition (Lines ~478-492)

```python
# OLD
self.action_space = spaces.Box(
    low=-0.1, high=0.1, shape=(self.action_dim,), dtype=np.float32
)

# NEW
limits_lower, limits_upper = self.robot.get_joint_limits(include_gripper=self.include_gripper)
if self.include_gripper:
    action_low = np.concatenate([limits_lower[:-1], [-1.0]])
    action_high = np.concatenate([limits_upper[:-1], [1.0]])
else:
    action_low = np.array(limits_lower)
    action_high = np.array(limits_upper)

self.action_space = spaces.Box(
    low=action_low, high=action_high, shape=(self.action_dim,), dtype=np.float32
)
```

**Reason:** Use actual joint limits instead of fixed delta range.

**Bug Fix (Jan 31, 2026):** Changed from `limits_lower[:-1] + [-1.0]` to `np.concatenate([limits_lower[:-1], [-1.0]])` to fix numpy array concatenation issue that caused `AssertionError: low.shape doesn't match provided shape`.

### 2. Step Function Execution (Lines ~645-648)

```python
# OLD
current_joint_pos = self.robot.get_joint_positions(include_gripper=self.include_gripper)
target_arm = current_joint_pos[:6] + arm_deltas
self.robot.set_arm_joints(target_arm)

# NEW
target_arm = np.clip(arm_absolute, self.robot.arm_lower_limits, self.robot.arm_upper_limits)
self.robot.set_arm_joints(target_arm)
```

**Reason:** Direct position assignment with safety clamping, no delta addition.

### 3. Variable Names (Lines ~615-630)

```python
# OLD: arm_deltas
# NEW: arm_absolute
```

**Reason:** Accurately reflect absolute position control.

### 4. Docstring (Lines ~600-611)

Updated action descriptions from `arm_deltas(6)` to `arm_absolute_joints(6)`.

---

## Required Actions Before Training

### ⚠️ Critical Prerequisites

1. **Dataset:** Must contain absolute joint positions, not deltas
   - Check: Action values should be in `[-π, π]` range, not `[-0.1, 0.1]`

2. **Normalizer:** Will automatically retrain with new statistics
   - No code changes needed

3. **Policy:** Must retrain from scratch
   - Old checkpoints are incompatible

### Quick Verification

```python
import zarr
root = zarr.open('your_dataset.zarr', 'r')
actions = root['data']['action'][0:10]
print("Action range:", actions[:, :6].min(), actions[:, :6].max())
# Expected: ~[-3.14, 3.14], NOT ~[-0.1, 0.1]
```

---

## Rollback

```bash
git checkout HEAD -- 3D-Diffusion-Policy/diffusion_policy_3d/env/pybullet/pybullet_wrapper.py
```
