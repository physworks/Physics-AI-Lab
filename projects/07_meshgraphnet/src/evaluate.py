import numpy as np
import matplotlib.pyplot as plt

r = np.load("../models/results_mgn.npz")
history = r["history"]
rollout_positions = r["rollout_positions"]  # (n_steps+1, N, 2)
test_traj = r["test_traj"]
onestep_mse = float(r["onestep_mse"])
rollout_mse = float(r["rollout_mse"])

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

ax = axes[0]
ax.plot(history)
ax.set_yscale("log")
ax.set_xlabel("Epoch")
ax.set_ylabel("Train loss (log scale)")
ax.set_title("MeshGraphNet Training Loss")
ax.grid(alpha=0.3)

ax = axes[1]
n_steps = len(test_traj)
sample_idx = [0, n_steps // 4, n_steps // 2, 3 * n_steps // 4, n_steps - 1]
colors = plt.cm.viridis(np.linspace(0, 1, len(sample_idx)))
for c, idx in zip(colors, sample_idx):
    ax.plot(test_traj[idx, :, 0], test_traj[idx, :, 1], color=c, linewidth=2,
             label=f"t={idx} (ground truth)")
    ax.plot(rollout_positions[idx, :, 0], rollout_positions[idx, :, 1], color=c,
             linestyle="--", linewidth=1.3, alpha=0.8)
ax.plot([], [], color="gray", linestyle="--", label="MeshGraphNet rollout")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Ground Truth vs Autoregressive Rollout\n(held-out bump configuration)")
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

ax = axes[2]
step_errs = np.mean((rollout_positions - test_traj) ** 2, axis=(1, 2))
ax.plot(step_errs, color="#d62728", linewidth=1.5)
ax.axhline(onestep_mse, color="gray", linestyle=":", label=f"One-step MSE ({onestep_mse:.2e})")
ax.set_yscale("log")
ax.set_xlabel("Rollout step")
ax.set_ylabel("MSE vs ground truth (log scale)")
ax.set_title(f"Rollout Error Accumulation\nFinal/One-step ratio: {rollout_mse/onestep_mse:.0f}x")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("../assets/meshgraphnet_results.png", dpi=150)
print("Saved ../assets/meshgraphnet_results.png")
