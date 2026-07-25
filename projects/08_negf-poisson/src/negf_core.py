"""
NEGF core (project 05 재사용 + 확장): 전하밀도 계산 추가.

project 05는 투과율 T(E)과 Landauer 전류만 계산했다. 여기서는 self-consistent
Poisson 결합을 위해, 각 사이트의 전자 밀도 n_i를 NEGF의 상관함수(correlation
function) G^n으로부터 직접 계산하는 기능을 추가한다:

    G^n(E) = G^R(E) [Gamma_L f_L(E) + Gamma_R f_R(E)] G^A(E)
    n_i = (1/2pi) * Integral G^n_ii(E) dE

이것이 project 03(평형 Boltzmann 통계 n=ni*exp(phi/Vt))과 근본적으로 다른 점 —
비평형 상태에서도, 그리고 양자역학적 결맞음(coherence) 효과까지 포함해 정확한
전하밀도를 준다. 평형(zero bias)에서는 이 NEGF 밀도가 Boltzmann 통계와
일치해야 하며, 이것이 이번 프로젝트의 핵심 검증 포인트 중 하나다.
"""

import numpy as np

HBAR = 1.0


def surface_greens_function(E, t, eta=1e-6):
    Ec = E + 1j * eta
    disc = np.asarray(4 * t**2 - Ec**2, dtype=complex)
    sqrt_term = np.sqrt(disc)
    g = (Ec - 1j * sqrt_term) / (2 * t**2)
    if np.imag(g) > 0:
        g = (Ec + 1j * sqrt_term) / (2 * t**2)
    return g


def lead_self_energy(E, t_lead, t_coupling, eta=1e-6):
    g = surface_greens_function(E, t_lead, eta)
    return t_coupling**2 * g


def fermi(E, mu, kT):
    x = np.clip((E - mu) / kT, -60, 60)
    return 1.0 / (1.0 + np.exp(x))


def build_hamiltonian(onsite, t):
    N = len(onsite)
    H = np.diag(onsite).astype(complex)
    idx = np.arange(N - 1)
    H[idx, idx + 1] = -t
    H[idx + 1, idx] = -t
    return H


def greens_functions(E, H, t_lead, t_coupling, eta=1e-6):
    """G^R, Gamma_L, Gamma_R를 계산."""
    N = H.shape[0]
    Sigma_L = lead_self_energy(E, t_lead, t_coupling, eta)
    Sigma_R = lead_self_energy(E, t_lead, t_coupling, eta)

    Sigma_L_mat = np.zeros((N, N), dtype=complex)
    Sigma_R_mat = np.zeros((N, N), dtype=complex)
    Sigma_L_mat[0, 0] = Sigma_L
    Sigma_R_mat[-1, -1] = Sigma_R

    G_R = np.linalg.inv((E + 1j * eta) * np.eye(N) - H - Sigma_L_mat - Sigma_R_mat)
    Gamma_L = 1j * (Sigma_L_mat - Sigma_L_mat.conj().T)
    Gamma_R = 1j * (Sigma_R_mat - Sigma_R_mat.conj().T)
    return G_R, Gamma_L, Gamma_R


def transmission_at(E, H, t_lead, t_coupling, eta=1e-6):
    G_R, Gamma_L, Gamma_R = greens_functions(E, H, t_lead, t_coupling, eta)
    G_A = G_R.conj().T
    T = np.trace(Gamma_L @ G_R @ Gamma_R @ G_A)
    return np.real(T)


def electron_density(H, t_lead, t_coupling, E_array, muL, muR, kT, eta=1e-6):
    """
    각 사이트의 전자 밀도 n_i를 에너지 적분으로 계산.
    (1/2pi) * Integral [G^R (Gamma_L f_L + Gamma_R f_R) G^A]_ii dE
    """
    N = H.shape[0]
    n_of_E = np.zeros((len(E_array), N))

    for k, E in enumerate(E_array):
        G_R, Gamma_L, Gamma_R = greens_functions(E, H, t_lead, t_coupling, eta)
        G_A = G_R.conj().T
        fL, fR = fermi(E, muL, kT), fermi(E, muR, kT)
        G_n = G_R @ (Gamma_L * fL + Gamma_R * fR) @ G_A
        n_of_E[k] = np.real(np.diag(G_n))

    n = np.trapezoid(n_of_E, E_array, axis=0) / (2 * np.pi)
    return n


def current_from_H(H, t_lead, t_coupling, E_array, muL, muR, kT, eta=1e-6):
    """Landauer 공식으로 전류 계산 (project 05와 동일한 정의)."""
    T_E = np.array([transmission_at(E, H, t_lead, t_coupling, eta) for E in E_array])
    integrand = T_E * (fermi(E_array, muL, kT) - fermi(E_array, muR, kT))
    return np.trapezoid(integrand, E_array)
