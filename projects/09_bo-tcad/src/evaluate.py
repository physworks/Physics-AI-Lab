import numpy as np
import matplotlib.pyplot as plt

r = np.load("../models/bo_comparison.npz")
bo_mean, bo_std = r["bo_mean"], r["bo_std"]
rand_mean, rand_std = r["rand_mean"], r["rand_std"]
n_total = int(r["n_total"])
target_vth = float(r["target_vth"])

evals = np.arange(1, n_total + 1)

fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))
ax.plot(evals, bo_mean, color="#1f77b4", linewidth=2, label="Bayesian Optimization")
ax.fill_between(evals, bo_mean - bo_std, bo_mean + bo_std, color="#1f77b4", alpha=0.2)
ax.plot(evals, rand_mean, color="#d62728", linewidth=2, label="Random Search")
ax.fill_between(evals, rand_mean - rand_std, rand_mean + rand_std, color="#d62728", alpha=0.2)
ax.set_yscale("log")
ax.set_xlabel("Number of expensive TCAD evaluations")
ax.set_ylabel("Best-so-far loss: (Vth_sim - Vth_target)^2  (log scale)")
ax.set_title(f"Bayesian Optimization vs Random Search\n"
              f"Finding (t_ox, Na) for target Vth={target_vth}V (5 seeds, mean +/- std)")
ax.legend()
ax.grid(alpha=0.3, which="both")
ax.axvline(5, color="gray", linestyle=":", alpha=0.6)
ax.text(5.2, ax.get_ylim()[1]*0.5, "BO init\n(random)", fontsize=8, color="gray")

fig.tight_layout()
fig.savefig("../assets/bo_vs_random.png", dpi=150)
print("Saved ../assets/bo_vs_random.png")

improvement = rand_mean[-1] / max(bo_mean[-1], 1e-12)
print(f"Final loss improvement (Random/BO ratio): {improvement:.1f}x")
