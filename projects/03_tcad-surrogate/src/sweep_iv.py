import numpy as np
import matplotlib.pyplot as plt
from diode_dd import solve_diode_iv, VT

Va_forward = np.linspace(0.0, 0.75, 40)
Va_full = np.concatenate([np.linspace(-0.3, -0.02, 8), Va_forward])

x, J, J_eq = solve_diode_iv(Va_full, verbose=False)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# (a) 선형 스케일 I-V (교과서적 다이오드 정류 특성)
ax = axes[0]
ax.plot(Va_full, J, color="#1f77b4", linewidth=1.5, marker='o', markersize=3)
ax.axhline(0, color="gray", linewidth=0.5)
ax.set_xlabel("Va (V)")
ax.set_ylabel("J (A/m^2)")
ax.set_title("Diode I-V (linear scale)\nSelf-consistent Poisson-Drift-Diffusion")
ax.grid(alpha=0.3)

# (b) 순방향 반로그 스케일 -- Shockley 지수 법칙 검증 (직선이면 성공)
ax = axes[1]
mask = J > 0
ax.semilogy(Va_full[mask], J[mask], color="#d62728", linewidth=1.5,
             marker='o', markersize=3, label="Simulated (Poisson-DD)")

# 이상적 Shockley 기울기(exp(Va/Vt)) 참고선 (임의 스케일, 기울기 비교용)
va_ref = np.linspace(0.35, 0.75, 20)
j_ref = J[mask][np.argmin(np.abs(Va_full[mask] - 0.5))] * np.exp((va_ref - 0.5) / VT)
ax.semilogy(va_ref, j_ref, color="black", linestyle=":", linewidth=1.5,
             label="Ideal exp(Va/Vt) slope (reference)")

ax.set_xlabel("Va (V)")
ax.set_ylabel("J (A/m^2, log scale)")
ax.set_title("Forward Bias I-V (semilog)\nValidation: should follow exp(Va/Vt)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

fig.tight_layout()
fig.savefig("../assets/diode_iv_validation.png", dpi=150)
print("Saved ../assets/diode_iv_validation.png")
print(f"\nEquilibrium current check: J_eq = {J_eq:.3e} A/m^2 (should be ~0)")
