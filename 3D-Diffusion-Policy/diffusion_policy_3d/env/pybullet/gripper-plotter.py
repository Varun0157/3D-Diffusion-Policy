import re
import matplotlib.pyplot as plt

# path to your txt file
log_path = "/home/aniruth/Desktop/RRC/3D-Diffusion-Policy/3D-Diffusion-Policy/data/outputs/pybullet_pick_place-dp3-no-table-envenvenvenvenv_seed42/gripper_comparisons/gripper_gt_vs_pred_20260120_134656.txt"

with open(log_path, "r") as f:
    text = f.read()

timesteps = []
gt_vals = []
pred_vals = []

pattern = re.compile(
    r"GT Timestep (\d+):\s+GT Gripper:\s+([-\d\.eE]+)\s+Pred Gripper:\s+([-\d\.eE]+)",
    re.MULTILINE
)

for t, gt, pred in pattern.findall(text):
    timesteps.append(int(t))
    gt_vals.append(float(gt))
    pred_vals.append(float(pred))

plt.figure()
plt.plot(timesteps, gt_vals, label="GT Gripper")
plt.plot(timesteps, pred_vals, label="Pred Gripper")
plt.xlabel("Timestep")
plt.ylabel("Gripper Value")
plt.legend()
plt.tight_layout()
plt.show()
