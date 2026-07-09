"""
Vg 스윕 → 전위 프로파일 여러 개 + C-V 곡선 계산.
C-V 곡선(축적-공핍-반전)이 교과서적인 형태로 나오는지가 솔버 정확성의
핵심 검증 포인트.
"""

import numpy as np
import matplotlib.pyplot as plt
from poisson_1d import solve_poisson, total_semiconductor_charge, EPS_OX

HERE = "."
ASSETS_DIR = "../assets"
DATA_DIR = "../data"

T_OX = 5e-9
T_SEMI = 3e-7
NA_VAL = 1e23


def sweep_cv(vg_array, t_ox=T_OX, t_semi=T_SEMI, na_val=NA_VAL):
    Qs = []
    profiles = []
    for vg in vg_array:
        x, phi, is_semi, vol, phi_p, iters = solve_poisson(
            Vg=vg, t_ox=t_ox, t_semi=t_semi, Na_val=na_val
        )
        q = total_semiconductor_charge(x, phi, is_semi, vol, na_val, t_ox)
        Qs.append(q)
        profiles.append((x, phi))
    return np.array(Qs), profiles


if __name__ == "__main__":
    vg_array = np.linspace(-2.0, 2.0, 81)
    Qs, profiles = sweep_cv(vg_array)

    # C = dQs/dVg (Qs는 반도체 쪽 전하이므로, 게이트 전하는 -Qs;
    # C_total ~ dQ_gate/dVg = -dQs/dVg, 여기선 크기 비교를 위해 |dQs/dVg| 사용)
    C_semi = -np.gradient(Qs, vg_array)  # F/m^2
    Cox = EPS_OX / T_OX
    C_total = 1.0 / (1.0 / Cox + 1.0 / np.abs(C_semi + 1e-30))  # 직렬 커패시턴스 근사 (정성적 확인용)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (a) 몇 개 Vg에서의 전위 프로파일
    ax = axes[0]
    sample_idx = [0, 20, 40, 60, 80]
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(sample_idx)))
    for c, idx in zip(colors, sample_idx):
        x, phi = profiles[idx]
        ax.plot(x[x < 50e-9] * 1e9, phi[x < 50e-9], color=c,
                 label=f"Vg={vg_array[idx]:.1f}V")
    ax.axvline(T_OX * 1e9, color="gray", linestyle=":", label="Oxide/Si interface")
    ax.set_xlabel("Position (nm)")
    ax.set_ylabel("Potential phi (V)")
    ax.set_title("Band Bending vs Gate Voltage\n(1D Nonlinear Poisson, Newton-Raphson)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (b) C-V curve (정성적: 축적/공핍/반전 영역이 보여야 함)
    ax = axes[1]
    ax.plot(vg_array, np.abs(C_semi) / Cox, color="#1f77b4", linewidth=2)
    ax.set_xlabel("Vg (V)")
    ax.set_ylabel("C_semiconductor / Cox (normalized)")
    ax.set_title("Quasi-static C-V Curve\n(validation: accumulation-depletion-inversion shape)")
    ax.grid(alpha=0.3)
    ax.axvline(0, color="gray", linestyle=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(f"{ASSETS_DIR}/poisson_cv_validation.png", dpi=150)
    print(f"Saved {ASSETS_DIR}/poisson_cv_validation.png")

    np.savez(f"{DATA_DIR}/cv_sweep.npz", vg=vg_array, Qs=Qs, C_semi=C_semi)
    print(f"Saved sweep data to {DATA_DIR}/cv_sweep.npz")
