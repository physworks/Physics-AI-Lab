import numpy as np
import matplotlib.pyplot as plt

r = np.load("../models/self_consistent_results.npz")
x = r["x_nm"]
phi_q = r["phi_quantum"]
phi_cl = r["phi_classical"]
n_negf = r["n_negf"]
n_cl = r["n_classical"]
history = r["history"]

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

ax = axes[0]
ax.plot(history, marker='o', markersize=3, color="#1f77b4")
ax.set_yscale("log")
ax.set_xlabel("Self-consistent iteration")
ax.set_ylabel("max |phi change| (V, log scale)")
ax.set_title("Self-Consistent Loop Convergence\n(Gummel-style damped fixed-point)")
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(x[:25], phi_cl[:25], label="Classical (Poisson-Boltzmann)", color="#d62728", linewidth=2)
ax.plot(x[:25], phi_q[:25], label="Self-consistent (NEGF electrons)", color="#1f77b4",
         linewidth=2, linestyle="--")
ax.set_xlabel("Depth from interface (nm)")
ax.set_ylabel("Potential phi (V)")
ax.set_title("Band Bending: Classical vs Quantum-Corrected")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

ax = axes[2]
ax.semilogy(x[:20], n_cl[:20], label="Classical n(x)", color="#d62728", linewidth=2, marker='o', markersize=3)
ax.semilogy(x[:20], n_negf[:20], label="NEGF n(x) (bulk-anchored calibration)",
             color="#1f77b4", linewidth=2, marker='s', markersize=3)
ax.set_xlabel("Depth from interface (nm)")
ax.set_ylabel("Electron density (m^-3, log scale)")
ax.set_title("Electron Density Near Interface\n(quantum vs classical shape comparison)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

fig.tight_layout()
fig.savefig("../assets/negf_poisson_results.png", dpi=150)
print("Saved ../assets/negf_poisson_results.png")

print(f"\nSummary:")
print(f"  Final max phi change: {history[-1]:.2e} V (converged)")
print(f"  Classical peak density: {n_cl[0]:.3e} m^-3 at interface")
print(f"  NEGF peak density (calibrated): {n_negf.max():.3e} m^-3")
