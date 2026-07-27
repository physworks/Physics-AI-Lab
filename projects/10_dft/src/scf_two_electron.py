"""
Milestone 2: 1D 2전자 "헬륨류 원자" self-consistent Kohn-Sham DFT.

닫힌 껍질(closed-shell) 가정: 2개 전자가 스핀이 반대로 같은 최저 공간
궤도함수를 이중 점유 (spin-restricted). 전자밀도 n(x) = 2*|psi_0(x)|^2.

Kohn-Sham 유효 포텐셜:
    v_KS(x) = v_ext(x) + v_Hartree(x) + v_xc(x)

v_Hartree(x) = integral n(x') / sqrt((x-x')^2 + a^2) dx'  (직접 수치적분, soft-Coulomb)

v_xc(x): 정확한 1D LDA 함수형(QMC로 피팅된 exchange-correlation, Baker et al.)은
이 프로젝트의 범위를 넘어서는 별도 유도가 필요하므로, 여기서는 **국소 밀도에
선형 비례하는 단순화된 LDA형 exchange 항** v_xc(x) = -C_x * n(x)를 사용한다.
이는 exchange가 self-consistent loop에 미치는 정성적 효과(에너지를 낮추고
밀도를 재분배하는 방향)를 보여주기 위한 것이며, 정량적으로 정확한 문헌
벤치마크 재현을 주장하지 않는다 — 이 한계를 README에 명시한다.

Self-consistency: Gummel/damped fixed-point (linear density mixing),
프로젝트 03/08에서 쓴 것과 같은 철학의 SCF 안정화 기법.
"""

import numpy as np
from dft_core import build_grid, soft_coulomb_ext, solve_eigenstates

N_ELECTRONS = 2
C_X = 0.8  # 단순화된 LDA형 exchange 계수 (임의의 예시 값, 정성적 데모용)


def hartree_potential(n, x, dx, a=1.0):
    """v_Hartree(x_i) = sum_j n(x_j) / sqrt((x_i-x_j)^2+a^2) * dx  (직접 수치적분)"""
    diff = x[:, None] - x[None, :]
    kernel = 1.0 / np.sqrt(diff ** 2 + a ** 2)
    return kernel @ n * dx


def xc_potential(n, C_x=C_X):
    """단순화된 LDA형 국소 exchange potential (선형 근사, 정성적 데모용)."""
    return -C_x * n


def run_scf(Z=2.0, a=1.0, L=10.0, n_points=400, n_iter=100, beta=0.3, tol=1e-8,
             use_xc=True, verbose=True):
    x, dx = build_grid(L, n_points)
    v_ext = soft_coulomb_ext(x, Z=Z, a=a)

    # 초기 밀도: v_ext만으로 얻은 비상호작용 바닥상태를 2전자로 채움
    energies0, psis0 = solve_eigenstates(v_ext, dx, n_states=1)
    n = N_ELECTRONS * psis0[:, 0] ** 2

    history = {"density_change": [], "eps0": []}

    for it in range(n_iter):
        v_H = hartree_potential(n, x, dx, a=a)
        v_xc = xc_potential(n) if use_xc else np.zeros_like(n)
        v_KS = v_ext + v_H + v_xc

        energies, psis = solve_eigenstates(v_KS, dx, n_states=1)
        n_new_orbital = N_ELECTRONS * psis[:, 0] ** 2

        # 밀도 정규화 확인 (닫힌 껍질 궤도이므로 자동으로 N_ELECTRONS가 되어야 함)
        norm = np.sum(n_new_orbital) * dx

        n_mixed = (1 - beta) * n + beta * n_new_orbital
        density_change = np.max(np.abs(n_mixed - n)) * dx
        n = n_mixed

        history["density_change"].append(density_change)
        history["eps0"].append(energies[0])

        if verbose and (it % 10 == 0 or it == n_iter - 1):
            print(f"  iter {it:3d}: eps_0={energies[0]:.5f}  max|dn|*dx={density_change:.2e}  "
                  f"norm(n)={norm:.4f}")

        if density_change < tol:
            break

    return {
        "x": x, "n": n, "v_ext": v_ext, "v_H": v_H, "v_xc": v_xc,
        "eps0": energies[0], "history": history, "converged_iter": it,
    }


if __name__ == "__main__":
    print("=== SCF: Hartree + simplified LDA-exchange (Z=2, 1D He-like atom) ===")
    result_xc = run_scf(Z=2.0, use_xc=True)

    print("\n=== SCF: Hartree-only (no exchange, for comparison) ===")
    result_h = run_scf(Z=2.0, use_xc=False)

    print(f"\nComparison:")
    print(f"  Hartree-only eps_0:        {result_h['eps0']:.5f} Hartree")
    print(f"  Hartree+exchange eps_0:    {result_xc['eps0']:.5f} Hartree")
    print(f"  Exchange lowers eigenvalue by: {result_h['eps0']-result_xc['eps0']:.5f} Hartree "
          f"(expected direction: exchange is attractive/stabilizing)")

    n_xc, n_h = result_xc["n"], result_h["n"]
    x = result_xc["x"]
    spread_xc = np.sqrt(np.sum(x ** 2 * n_xc) / np.sum(n_xc) -
                          (np.sum(x * n_xc) / np.sum(n_xc)) ** 2)
    spread_h = np.sqrt(np.sum(x ** 2 * n_h) / np.sum(n_h) -
                         (np.sum(x * n_h) / np.sum(n_h)) ** 2)
    print(f"  Density spread (std): Hartree-only={spread_h:.4f}, with exchange={spread_xc:.4f}")

    np.savez("../models/scf_results.npz",
             x=result_xc["x"], n_xc=result_xc["n"], n_h=result_h["n"],
             v_ext=result_xc["v_ext"], v_H_xc=result_xc["v_H"], v_xc=result_xc["v_xc"],
             history_xc=np.array(result_xc["history"]["density_change"]),
             history_h=np.array(result_h["history"]["density_change"]),
             eps0_xc=result_xc["eps0"], eps0_h=result_h["eps0"])
    print("\nSaved results.")
