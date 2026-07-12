import numpy as np
import matplotlib.pyplot as plt
from negf_core import device_hamiltonian, transmission_spectrum

t = 1.0

# --- Milestone 1: 균일한 체인 (검증 기준선, T(E)=1 이어야 함) ---
N1 = 30
H_uniform = device_hamiltonian(np.zeros(N1), t)
E_array = np.linspace(-2.8, 2.8, 500)
T_uniform = transmission_spectrum(E_array, H_uniform, t_lead=t, t_coupling=t)

# --- Milestone 2: Double-barrier 공명터널링 구조 ---
N2 = 40
onsite2 = np.zeros(N2)
Vb = 2.0          # 장벽 높이
barrier_width = 3
well_start = 15
well_width = 8

onsite2[well_start - barrier_width: well_start] = Vb                       # 왼쪽 장벽
onsite2[well_start + well_width: well_start + well_width + barrier_width] = Vb  # 오른쪽 장벽
# 사이(well_start ~ well_start+well_width)는 우물(onsite=0)

H_barrier = device_hamiltonian(onsite2, t)
T_barrier = transmission_spectrum(E_array, H_barrier, t_lead=t, t_coupling=t)

# 검증용: 우물만 고립계로 놓고 대각화 (hard-wall boundary)해서 준속박상태 에너지 추정
H_well_isolated = device_hamiltonian(np.zeros(well_width), t)
well_eigenvalues = np.sort(np.linalg.eigvalsh(H_well_isolated))
# 장벽보다 낮은 에너지의 준속박상태만 공명으로 나타날 가능성이 높음
resonance_candidates = well_eigenvalues[well_eigenvalues < Vb]

print("Milestone 1 (uniform chain) validation:")
in_band = np.abs(E_array) < 1.9 * t
print(f"  T(E) in-band mean = {T_uniform[in_band].mean():.6f} (should be 1.0)")

print("\nMilestone 2 (double barrier) resonance check:")
print(f"  Isolated well eigenvalues (< barrier height {Vb}): {resonance_candidates}")
peak_idx = np.where((T_barrier[1:-1] > T_barrier[:-2]) & (T_barrier[1:-1] > T_barrier[2:]) &
                       (T_barrier[1:-1] > 0.3))[0] + 1
print(f"  T(E) peaks found at E = {E_array[peak_idx]}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.plot(E_array, T_uniform, color="#1f77b4", linewidth=1.5)
ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)
ax.set_xlabel("Energy E (units of t)")
ax.set_ylabel("Transmission T(E)")
ax.set_title("Milestone 1: Uniform Chain\n(validation: T(E)=1 across entire band)")
ax.set_ylim(-0.05, 1.15)
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(E_array, T_barrier, color="#d62728", linewidth=1.5, label="T(E), double barrier")
for ev in resonance_candidates:
    ax.axvline(ev, color="gray", linestyle="--", linewidth=1, alpha=0.7)
ax.plot([], [], color="gray", linestyle="--", label="Isolated-well eigenvalues\n(resonance prediction)")
ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.5)
ax.set_xlabel("Energy E (units of t)")
ax.set_ylabel("Transmission T(E)")
ax.set_title("Milestone 2: Double-Barrier Resonant Tunneling\n(validation: peaks align with isolated-well bound states)")
ax.set_ylim(-0.05, 1.15)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("../assets/negf_transmission.png", dpi=150)
print("\nSaved ../assets/negf_transmission.png")
