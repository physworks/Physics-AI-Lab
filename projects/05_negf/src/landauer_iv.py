"""
Landauer 공식으로 NEGF I-V 특성 계산.

I(V) = (2e/h) * Integral[ T(E,V) * (f(E-muL) - f(E-muR)) ] dE
muL = EF + V/2,  muR = EF - V/2  (대칭 전압 강하 가정)

중요한 단순화: 진짜 self-consistent NEGF-Poisson 계산은 전하 재분포에 따라
전위가 어떻게 바뀌는지까지 풀지만, 여기서는 교과서 수준의 근사로 디바이스에
"선형 포텐셜 램프"(왼쪽 +V/2 ~ 오른쪽 -V/2)를 강제로 걸어 바이어스에 따라
장벽/우물의 에너지 준위가 이동하는 효과만 흉내낸다. 이 정도 근사로도
공명터널링 다이오드의 핵심 특징인 NDR(negative differential resistance)을
정성적으로 재현할 수 있다.
"""

import numpy as np
import matplotlib.pyplot as plt
from negf_core import device_hamiltonian, transmission_spectrum

t = 1.0
N = 60  # 오른쪽에 버퍼를 추가해 디바이스를 비대칭으로 만듦
Vb = 2.0
barrier_width = 3
well_start = 15
well_width = 3

onsite_base = np.zeros(N)
onsite_base[well_start - barrier_width: well_start] = Vb
onsite_base[well_start + well_width: well_start + well_width + barrier_width] = Vb
# 우물+장벽 구조는 왼쪽에 치우쳐 있고(인덱스 12~26), 오른쪽 버퍼(27~59)가 길어서
# 디바이스 전체의 기하학적 중심(i=29.5)이 우물 중심(~19)과 겹치지 않음 —
# 램프의 "0점"이 우물 밖에 위치하게 되어, 바이어스에 따라 우물의 에너지 준위가
# 실제로 이동하게 됨 (대칭 구조에서는 이 효과가 정확히 상쇄되어 사라짐)


def biased_onsite(onsite_base, V):
    """왼쪽 +V/2 ~ 오른쪽 -V/2 로 선형으로 떨어지는 포텐셜 램프를 더함."""
    N = len(onsite_base)
    ramp = V * (0.5 - np.arange(N) / (N - 1))
    return onsite_base + ramp


def fermi(E, mu, kT):
    x = (E - mu) / kT
    x = np.clip(x, -60, 60)
    return 1.0 / (1.0 + np.exp(x))


def landauer_current(E_array, T_E, muL, muR, kT):
    integrand = T_E * (fermi(E_array, muL, kT) - fermi(E_array, muR, kT))
    return np.trapezoid(integrand, E_array)  # 단위: 2e/h = 1 (reduced units)


if __name__ == "__main__":
    EF = -1.6     # Fermi level: 좁은 우물의 낮은 준속박상태보다 살짝 아래
    kT = 0.05      # 열적 broadening (격자 해상도 대비 적분 안정성 확보를 위해 확대)
    E_array = np.linspace(-3.0, 3.0, 3000)

    V_array = np.linspace(0.0, 2.0, 100)
    I_array = np.zeros_like(V_array)

    for i, V in enumerate(V_array):
        onsite_V = biased_onsite(onsite_base, V)
        H_V = device_hamiltonian(onsite_V, t)
        T_E = transmission_spectrum(E_array, H_V, t_lead=t, t_coupling=t)

        muL, muR = EF + V / 2, EF - V / 2
        I_array[i] = landauer_current(E_array, T_E, muL, muR, kT)

    dI_dV = np.gradient(I_array, V_array)
    ndr_mask = dI_dV < 0
    print(f"NDR region detected: {ndr_mask.sum()} / {len(V_array)} points have dI/dV < 0")
    if ndr_mask.any():
        v_ndr = V_array[ndr_mask]
        print(f"  NDR voltage range: {v_ndr.min():.2f}V ~ {v_ndr.max():.2f}V")
    peak_idx = np.argmax(I_array)
    print(f"  Peak current at V={V_array[peak_idx]:.2f}, I={I_array[peak_idx]:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(V_array, I_array, color="#1f77b4", linewidth=2, marker='o', markersize=3)
    if ndr_mask.any():
        ax.axvspan(v_ndr.min(), v_ndr.max(), color="red", alpha=0.1, label="NDR region")
    ax.set_xlabel("Bias Voltage V (units of t/e)")
    ax.set_ylabel("Current I (units of 2e/h)")
    ax.set_title("NEGF I-V via Landauer Formula\n(Resonant Tunneling Diode toy model)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    sample_Vs = [0.0, V_array[peak_idx], V_array[-1]]
    colors = ["#2ca02c", "#d62728", "#9467bd"]
    for V, c in zip(sample_Vs, colors):
        onsite_V = biased_onsite(onsite_base, V)
        H_V = device_hamiltonian(onsite_V, t)
        T_E = transmission_spectrum(E_array, H_V, t_lead=t, t_coupling=t)
        ax.plot(E_array, T_E, color=c, linewidth=1.3, label=f"V={V:.2f}")
        muL, muR = EF + V / 2, EF - V / 2
        ax.axvspan(muR, muL, color=c, alpha=0.08)
    ax.set_xlim(-3, 2)
    ax.set_xlabel("Energy E (units of t)")
    ax.set_ylabel("Transmission T(E)")
    ax.set_title("T(E) at selected bias points\n(shaded = conduction window [muR, muL])")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("../assets/negf_iv_curve.png", dpi=150)
    print("\nSaved ../assets/negf_iv_curve.png")
