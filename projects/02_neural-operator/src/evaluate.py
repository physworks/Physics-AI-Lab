import os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
MODELS_DIR = os.path.join(HERE, "..", "models")
ASSETS_DIR = os.path.join(HERE, "..", "assets")

r = np.load(os.path.join(MODELS_DIR, "fno_results.npz"))
vgs_grid = r["vgs_grid"]
vth_test = r["vth_test"]
pred_test = r["pred_test"]  # (n_test, 1, N)
true_test = r["true_test"]
history = r["history"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.plot(history)
ax.set_yscale("log")
ax.set_xlabel("Epoch")
ax.set_ylabel("Train MSE (log scale)")
ax.set_title("FNO Training Loss")
ax.grid(alpha=0.3)

ax = axes[1]
colors = plt.cm.viridis(np.linspace(0, 1, len(vth_test)))
for i, vth in enumerate(vth_test):
    ax.plot(vgs_grid, true_test[i, 0] * 1e3, color=colors[i], linewidth=2,
             label=f"Vth={vth:.2f}V (true)")
    ax.plot(vgs_grid, pred_test[i, 0] * 1e3, color=colors[i], linestyle="--",
             linewidth=1.5, label=f"Vth={vth:.2f}V (FNO pred)")
ax.set_xlabel("Vgs (V)")
ax.set_ylabel("Id (mA)")
ax.set_title("FNO Generalization to Held-out Vth Values\n(not seen during training)")
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.3)

fig.tight_layout()
out_path = os.path.join(ASSETS_DIR, "fno_generalization.png")
fig.savefig(out_path, dpi=150)
print(f"Saved {out_path}")
