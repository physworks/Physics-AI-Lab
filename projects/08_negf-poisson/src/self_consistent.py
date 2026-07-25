"""
Self-consistent Schrodinger(NEGF)-Poisson 루프.

핵심 결과: 반전층 전자를 classical Boltzmann 대신 NEGF로 계산하면, 실제
device 물리에서 잘 알려진 "quantum confinement로 인한 전하 setback" 현상
(전하 피크가 계면에서 물러나고, 피크 값이 낮아지며 퍼짐)이 나타난다.

Self-consistency: NEGF 밀도의 dphi에 대한 해석적 미분이 없어 Newton 대신
damped fixed-point iteration(under-relaxation)으로 수렴시킨다.

Normalization 한계 (정직하게 기록): 1D tight-binding 체인의 국소 상태밀도는
실제 3D 반전층 밀도에 필요한 transverse(in-plane) 2D 상태밀도 인자가 빠져있다.
이를 완전히 유도하는 대신, bulk 깊은 곳(quantum confinement가 없어야 하는 영역)에서
NEGF 밀도가 classical 값과 일치하도록 전체 프로파일을 한 번 스케일링하는
"bulk-anchoring calibration"을 사용한다 — Ec_offset calibration과 같은 철학.
"""

import numpy as np
from qpoisson_negf import (
    build_grid, classical_poisson_newton, solve_linear_poisson_fixed_charge,
    negf_electron_density, EC_OFFSET, NA_VAL, NI, VT, Q, A_ANG,
)


def run_self_consistent(Vg, L_nm=15.0, n_iter=25, beta=0.3, n_energy=200, verbose=True):
    a_nm = A_ANG / 10
    x, N = build_grid(L_nm, a_nm)
    x_full_nm = np.concatenate([[0.0], x, [x[-1] + a_nm]])

    # 1) classical 초기값 (좋은 시작점 + 비교 기준)
    phi_full_classical, phi_p_eq = classical_poisson_newton(Vg, NA_VAL, x_full_nm)
    phi_full = phi_full_classical.copy()
    n_classical_full = NI * np.exp(phi_full_classical / VT)

    history = []
    for it in range(n_iter):
        phi_q = phi_full[1:-1]
        Ec_site = -phi_q + EC_OFFSET
        E_grid = np.linspace(Ec_site.min() - 0.1, 12 * VT, n_energy)

        n_negf_raw = negf_electron_density(phi_q, EC_OFFSET, 0.0, E_grid, a_nm, eta=1e-3)

        # bulk-anchoring calibration (마지막 5개 site 평균으로 anchor, 노이즈 완화)
        n_classical_q = NI * np.exp(phi_q / VT)
        scale = np.mean(n_classical_q[-5:]) / max(np.mean(n_negf_raw[-5:]), 1e-30)
        n_negf = n_negf_raw * scale

        p_classical_q = NI * np.exp(-phi_q / VT)
        charge_q = Q * (p_classical_q - n_negf - NA_VAL)

        charge_full = np.zeros(len(phi_full))
        charge_full[1:-1] = charge_q

        phi_poisson = solve_linear_poisson_fixed_charge(x_full_nm, charge_full, Vg, phi_p_eq)

        d_phi = phi_full + beta * (phi_poisson - phi_full)
        max_change = np.max(np.abs(d_phi - phi_full))
        phi_full = d_phi

        history.append(max_change)
        if verbose and (it % 5 == 0 or it == n_iter - 1):
            print(f"  iter {it:3d}: max phi change = {max_change:.6f} V, calib scale = {scale:.3e}")

    phi_q_final = phi_full[1:-1]
    Ec_site = -phi_q_final + EC_OFFSET
    E_grid = np.linspace(Ec_site.min() - 0.1, 12 * VT, n_energy)
    n_negf_final_raw = negf_electron_density(phi_q_final, EC_OFFSET, 0.0, E_grid, a_nm, eta=1e-3)
    n_classical_final = NI * np.exp(phi_q_final / VT)
    scale_final = np.mean(n_classical_final[-5:]) / max(np.mean(n_negf_final_raw[-5:]), 1e-30)
    n_negf_final = n_negf_final_raw * scale_final

    return {
        "x_nm": x,
        "phi_quantum": phi_q_final,
        "phi_classical": phi_full_classical[1:-1],
        "n_negf": n_negf_final,
        "n_classical": n_classical_final,
        "history": np.array(history),
    }


if __name__ == "__main__":
    import time
    t0 = time.time()
    print("Running self-consistent NEGF-Poisson (Vg=0.3V)...")
    result = run_self_consistent(Vg=0.3, n_iter=25, beta=0.3, n_energy=200)
    print(f"Total time: {time.time()-t0:.1f}s")

    x = result["x_nm"]
    n_negf = result["n_negf"]
    n_cl = result["n_classical"]

    peak_idx_negf = np.argmax(n_negf[:20])  # 계면 근처(첫 20 site)에서 피크 탐색
    peak_idx_cl = np.argmax(n_cl[:20])
    print(f"\nClassical peak position: x={x[peak_idx_cl]:.2f} nm (site {peak_idx_cl})")
    print(f"NEGF (quantum) peak position: x={x[peak_idx_negf]:.2f} nm (site {peak_idx_negf})")
    print(f"Setback: {x[peak_idx_negf]-x[peak_idx_cl]:.2f} nm")
    print(f"Classical peak value: {n_cl[peak_idx_cl]:.3e} m^-3")
    print(f"NEGF peak value: {n_negf[peak_idx_negf]:.3e} m^-3")

    np.savez("../models/self_consistent_results.npz", **result)
    print("\nSaved results.")
