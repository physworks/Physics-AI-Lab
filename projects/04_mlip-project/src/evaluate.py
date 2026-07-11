import numpy as np
import matplotlib.pyplot as plt

r = np.load("../models/mlip_results.npz")
pred_E, true_E = r["pred_E"], r["true_E"]
pred_F, true_F = r["pred_F"], r["true_F"]
hist_total, hist_e, hist_f = r["history_total"], r["history_energy"], r["history_force"]
e_rmse, f_rmse = float(r["e_rmse"]), float(r["f_rmse"])

print(f"Force magnitude stats (true, held-out set): "
      f"mean={np.abs(true_F).mean():.3f}, std={true_F.std():.3f}, max={np.abs(true_F).max():.3f}")
print(f"Force RMSE relative to std: {f_rmse / true_F.std():.2%}")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

ax = axes[0]
ax.plot(hist_total, label="Total")
ax.plot(hist_e, label="Energy")
ax.plot(hist_f, label="Force")
ax.set_yscale("log")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss (log scale)")
ax.set_title("MLIP Training Loss")
ax.legend()
ax.grid(alpha=0.3)

ax = axes[1]
ax.scatter(true_E, pred_E, s=10, alpha=0.5)
lims = [min(true_E.min(), pred_E.min()), max(true_E.max(), pred_E.max())]
ax.plot(lims, lims, "k--", linewidth=1)
ax.set_xlabel("True Energy (LJ)")
ax.set_ylabel("Predicted Energy (MLIP)")
ax.set_title(f"Energy Parity (held-out T=0.8)\nRMSE={e_rmse:.3f}")
ax.grid(alpha=0.3)

ax = axes[2]
tf_flat = true_F.flatten()
pf_flat = pred_F.flatten()
sample_idx = np.random.default_rng(0).choice(len(tf_flat), size=3000, replace=False)
ax.scatter(tf_flat[sample_idx], pf_flat[sample_idx], s=5, alpha=0.3)
flims = [tf_flat.min(), tf_flat.max()]
ax.plot(flims, flims, "k--", linewidth=1)
ax.set_xlabel("True Force component (LJ)")
ax.set_ylabel("Predicted Force component (MLIP)")
ax.set_title(f"Force Parity (held-out T=0.8)\nRMSE={f_rmse:.3f}")
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("../assets/mlip_validation.png", dpi=150)
print("Saved ../assets/mlip_validation.png")
