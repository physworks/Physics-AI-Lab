import numpy as np
import matplotlib.pyplot as plt

r = np.load("../models/fno_surrogate_results.npz")
vg_grid = r["vg_grid"]
na_test = r["na_test"]
pred_test = r["pred_test"]
true_test = r["true_test"]
history = r["history"]
speedup = float(r["speedup"])
solver_t = float(r["solver_time_per_curve"]) * 1000
fno_t = float(r["fno_time_per_curve"]) * 1000

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.plot(history)
ax.set_yscale("log")
ax.set_xlabel("Epoch")
ax.set_ylabel("Train MSE (log scale)")
ax.set_title("FNO Training Loss (TCAD Surrogate)")
ax.grid(alpha=0.3)

ax = axes[1]
colors = plt.cm.plasma(np.linspace(0, 1, len(na_test)))
for i, na in enumerate(na_test):
    ax.plot(vg_grid, true_test[i, 0], color=colors[i], linewidth=2,
             label=f"Na={na:.1e} (solver)")
    ax.plot(vg_grid, pred_test[i, 0], color=colors[i], linestyle="--",
             linewidth=1.5, label=f"Na={na:.1e} (FNO)")
ax.set_xlabel("Vg (V)")
ax.set_ylabel("Qs (C/m^2)")
ax.set_title(f"FNO Generalization to Held-out Na Values\n"
              f"Solver: {solver_t:.1f} ms/curve  |  FNO: {fno_t:.3f} ms/curve  |  Speedup: {speedup:.0f}x")
ax.legend(fontsize=7, ncol=2)
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("../assets/fno_tcad_surrogate.png", dpi=150)
print("Saved ../assets/fno_tcad_surrogate.png")
