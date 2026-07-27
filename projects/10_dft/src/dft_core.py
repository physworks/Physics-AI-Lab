"""
1D Kohn-Sham DFT 솔버 (soft-Coulomb 포텐셜).

3D 쿨롱 포텐셜(-Z/|x|)은 1D에서 원점 특이점 때문에 그대로 쓸 수 없어,
1D DFT 문헌(Baker, Wasserman, Burke 등)의 표준 관행인 soft-Coulomb
포텐셜을 사용한다:

    v_ext(x) = -Z / sqrt(x^2 + a^2)
    v_ee(x, x') = 1 / sqrt((x-x')^2 + a^2)   (전자-전자 상호작용에도 동일 완화)

원자 단위(atomic units, hbar=m_e=e=1) 사용: Hamiltonian의 운동에너지 항은
T = -1/2 d^2/dx^2.

이산화: 3점 유한차분(finite difference) stencil으로 tridiagonal
Hamiltonian 행렬을 만들고 numpy.linalg.eigh로 직접 대각화 (격자 크기가
작아 반복 고유값 솔버 없이도 충분히 빠름).
"""

import numpy as np


def build_grid(L=10.0, n_points=400):
    x = np.linspace(-L, L, n_points)
    dx = x[1] - x[0]
    return x, dx


def soft_coulomb_ext(x, Z, a=1.0):
    return -Z / np.sqrt(x ** 2 + a ** 2)


def kinetic_matrix(n_points, dx):
    """3점 유한차분 -1/2 d^2/dx^2 의 tridiagonal 행렬."""
    T = np.zeros((n_points, n_points))
    diag = 1.0 / dx ** 2
    offdiag = -0.5 / dx ** 2
    np.fill_diagonal(T, diag)
    T += np.diag(np.full(n_points - 1, offdiag), k=1)
    T += np.diag(np.full(n_points - 1, offdiag), k=-1)
    return T


def solve_eigenstates(v_total, dx, n_states=3):
    """H = T + diag(v_total)를 대각화해 가장 낮은 n_states개 (에너지, 파동함수) 반환."""
    n_points = len(v_total)
    H = kinetic_matrix(n_points, dx) + np.diag(v_total)
    eigvals, eigvecs = np.linalg.eigh(H)

    # 파동함수 정규화 (trapezoidal 근사: sum(psi^2)*dx = 1)
    psis = eigvecs[:, :n_states].copy()
    for i in range(n_states):
        norm = np.sqrt(np.sum(psis[:, i] ** 2) * dx)
        psis[:, i] /= norm

    return eigvals[:n_states], psis


if __name__ == "__main__":
    # Milestone 1: 단일 전자 1D "수소원자" (soft-Coulomb, Z=1, a=1)
    x, dx = build_grid(L=10.0, n_points=400)
    v_ext = soft_coulomb_ext(x, Z=1.0, a=1.0)
    energies, psis = solve_eigenstates(v_ext, dx, n_states=3)

    print("1D soft-Coulomb hydrogen atom (Z=1, a=1) eigenvalues:")
    for i, E in enumerate(energies):
        print(f"  State {i}: E = {E:.4f} Hartree")

    # 문헌 참고값 (Baker, Wasserman, Burke 계열의 1D DFT 튜토리얼에서 흔히 인용되는
    # soft-Coulomb a=1 기저상태 에너지). 우리 격자 해상도에서 이 값에 얼마나
    # 가까운지 확인 (정확한 재현이 아니라, 알려진 기준 근처인지 검증).
    E0_reference = -0.6698
    print(f"\nLiterature reference ground state energy: {E0_reference:.4f} Hartree")
    print(f"Our result: {energies[0]:.4f} Hartree  (diff: {abs(energies[0]-E0_reference):.4f})")

    # 정규화 확인
    norm_check = np.sum(psis[:, 0] ** 2) * dx
    print(f"\nGround state normalization check: integral|psi_0|^2 dx = {norm_check:.6f} (should be 1.0)")
