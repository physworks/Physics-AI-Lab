"""
이온주입 MC 검증:
1. Rp(투사 범위)가 입사 에너지에 따라 단조증가하는지 확인 (LSS 이론의 기본 예측)
2. 여러 에너지에서 깊이 분포(implant profile) 형태 확인 (근사적으로 Gaussian/skewed)
3. 대표 궤적 몇 개를 시각화해 BCA의 zigzag 산란 경로 특성 확인
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from ion_implant_mc import run_ensemble, simulate_ion, MEAN_SPACING

ENERGIES_KEV = [10, 20, 30, 50, 70, 100, 150]
N_IONS = 1500

print("Running ensembles across energies...")
results = {}
for E_keV in ENERGIES_KEV:
    t0 = time.time()
    depths, n_back = run_ensemble(E_keV * 1000, n_ions=N_IONS, seed=7)
    results[E_keV] = (depths, n_back)
    print(f"  {E_keV:4d} keV: Rp={depths.mean()/10:6.1f} nm  dRp={depths.std()/10:5.1f} nm  "
          f"backscatter={n_back}/{N_IONS}  [{time.time()-t0:.1f}s]")

Rp_array = np.array([results[E][0].mean() / 10 for E in ENERGIES_KEV])
dRp_array = np.array([results[E][0].std() / 10 for E in ENERGIES_KEV])
E_array = np.array(ENERGIES_KEV)

# 단조증가 검증
is_monotonic = np.all(np.diff(Rp_array) > 0)
print(f"\nMonotonic Rp(E) increase: {is_monotonic}")

# power-law fit: Rp ~ E^n (LSS 이론에서 예상되는 스케일링 형태)
log_fit = np.polyfit(np.log(E_array), np.log(Rp_array), 1)
n_exponent = log_fit[0]
print(f"Power-law fit: Rp ~ E^{n_exponent:.2f} (LSS theory typically predicts n in range ~0.6-1.0 for this energy regime)")

# --- 시각화 ---
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

ax = axes[0]
ax.errorbar(E_array, Rp_array, yerr=dRp_array, fmt='o-', color="#1f77b4", capsize=3)
ax.set_xlabel("Incident Energy (keV)")
ax.set_ylabel("Projected Range Rp (nm)")
ax.set_title(f"Rp vs Energy (validation: monotonic)\nPower-law fit: Rp ~ E^{n_exponent:.2f}")
ax.grid(alpha=0.3)

ax = axes[1]
colors = ["#2ca02c", "#d62728", "#9467bd"]
for E_keV, c in zip([30, 60, 100], colors):
    if E_keV not in results:
        depths, _ = run_ensemble(E_keV * 1000, n_ions=N_IONS, seed=7)
    else:
        depths = results[E_keV][0]
    ax.hist(depths / 10, bins=30, alpha=0.5, color=c, label=f"{E_keV} keV", density=True)
ax.set_xlabel("Depth (nm)")
ax.set_ylabel("Probability density")
ax.set_title("Implant Depth Profile\n(validation: roughly Gaussian/skewed shape)")
ax.legend()
ax.grid(alpha=0.3)

ax = axes[2]
rng = np.random.default_rng(3)
for i in range(6):
    _, _, _, path = simulate_ion(60000, rng, record_path=True)
    path = np.array(path)
    ax.plot(path[:, 2] / 10, path[:, 0] / 10, alpha=0.7, linewidth=1)
ax.set_xlabel("Depth z (nm)")
ax.set_ylabel("Lateral x (nm)")
ax.set_title("Sample Trajectories (E=60 keV)\n(zigzag scattering path, BCA characteristic)")
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("../assets/mc_implant_validation.png", dpi=150)
print("\nSaved ../assets/mc_implant_validation.png")
