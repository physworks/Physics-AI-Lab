import numpy as np
import matplotlib.pyplot as plt

r = np.load("../models/scf_results.npz")
x = r["x"]
n_xc, n_h = r["n_xc"], r["n_h"]
v_ext = r["v_ext"]
history_xc, history_h = r["history_xc"], r["history_h"]
eps0_xc, eps0_h = float(r["eps0_xc"]), float(r["eps0_h"])

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

ax = axes[0]
ax.plot(history_h, label="Hartree-only", color="#d62728", linewidth=1.5)
ax.plot(history_xc, label="Hartree + exchange", color="#1f77b4", linewidth=1.5)
ax.set_yscale("log")
ax.set_xlabel("SCF iteration")
ax.set_ylabel("max density change (log scale)")
ax.set_title("Self-Consistent Field Convergence\n(1D two-electron system)")
ax.legend()
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(x, v_ext, color="gray", linewidth=1.5, linestyle=":", label="v_ext (soft-Coulomb, Z=2)")
ax.plot(x, n_h, color="#d62728", linewidth=2, label=f"n(x), Hartree-only (eps_0={eps0_h:.3f})")
ax.plot(x, n_xc, color="#1f77b4", linewidth=2, label=f"n(x), +exchange (eps_0={eps0_xc:.3f})")
ax.set_xlim(-6, 6)
ax.set_xlabel("x (Bohr)")
ax.set_ylabel("electron density n(x) / potential")
ax.set_title("Self-Consistent Electron Density\n(2-electron 1D He-like atom)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[2]
ax.plot(x, n_h, color="#d62728", linewidth=2, label="Hartree-only")
ax.plot(x, n_xc, color="#1f77b4", linewidth=2, label="Hartree + exchange")
ax.set_xlim(-4, 4)
ax.set_xlabel("x (Bohr)")
ax.set_ylabel("electron density n(x)")
ax.set_title("Density Comparison (zoomed)\nExchange contracts the density")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("../assets/dft_scf_results.png", dpi=150)
print("Saved ../assets/dft_scf_results.png")
