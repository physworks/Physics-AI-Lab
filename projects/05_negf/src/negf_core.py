"""
1D Tight-Binding 체인의 NEGF(Non-Equilibrium Green's Function) 양자수송 계산.

물리적 설정:
- 디바이스: N개 사이트로 이루어진 1D tight-binding 체인 (onsite energy eps_i, hopping t)
- 좌/우 리드: 디바이스와 동일한 hopping을 갖는 반무한(semi-infinite) 1D 체인

핵심 아이디어: 리드를 명시적으로 다루는 대신, 디바이스에 미치는 영향을
self-energy(Sigma_L, Sigma_R)로 "요약"해 유한 크기 행렬 문제로 축소한다.
이게 NEGF의 핵심 트릭이며, 실제 반도체 소자 양자수송 시뮬레이션(예: 나노와이어
FET, 터널링 다이오드)에서 쓰이는 것과 동일한 프레임워크.

반무한 1D 체인의 표면 Green's function은 닫힌 형태(analytic closed form)로
알려져 있어 (Datta, "Electronic Transport in Mesoscopic Systems"), 이를 이용해
self-energy를 정확하게 계산한다 (반복적 계산인 Sancho-Rubio 알고리즘 없이도 정확).
"""

import numpy as np

HBAR = 1.0  # reduced units


def surface_greens_function(E, t, eta=1e-8):
    """
    반무한 1D tight-binding 체인(hopping t, onsite 0)의 표면(끝점) Green's function.
    닫힌 형태 해:  g(E) = (E - i*sqrt(4t^2 - E^2)) / (2t^2),  |E| < 2t (밴드 내부)
    밴드 밖(|E|>2t)에서는 evanescent(소산) 해로 부드럽게 연결.

    retarded Green's function이므로 허수부는 항상 <= 0.
    """
    Ec = E + 1j * eta
    disc = np.asarray(4 * t**2 - Ec**2, dtype=complex)
    sqrt_term = np.sqrt(disc)
    # retarded 해: Im(g) <= 0가 되도록 부호 선택
    g = (Ec - 1j * sqrt_term) / (2 * t**2)
    if np.imag(g) > 0:
        g = (Ec + 1j * sqrt_term) / (2 * t**2)
    return g


def lead_self_energy(E, t_lead, t_coupling, eta=1e-8):
    """
    디바이스 끝 사이트에 결합된 리드의 self-energy.
    Sigma(E) = t_coupling^2 * g_surface(E)
    """
    g = surface_greens_function(E, t_lead, eta)
    return t_coupling**2 * g


def device_hamiltonian(onsite, t):
    """
    onsite: (N,) 배열, 각 사이트의 on-site energy
    t: hopping (nearest-neighbor, 균일하다고 가정)
    반환: (N, N) tight-binding Hamiltonian
    """
    N = len(onsite)
    H = np.diag(onsite).astype(complex)
    for i in range(N - 1):
        H[i, i + 1] = -t
        H[i + 1, i] = -t
    return H


def transmission(E, H_device, t_lead, t_coupling, eta=1e-8):
    """
    주어진 에너지 E에서의 투과율 T(E) = Tr[Gamma_L G^R Gamma_R G^A].
    """
    N = H_device.shape[0]
    Sigma_L = lead_self_energy(E, t_lead, t_coupling, eta)
    Sigma_R = lead_self_energy(E, t_lead, t_coupling, eta)

    Sigma_L_mat = np.zeros((N, N), dtype=complex)
    Sigma_R_mat = np.zeros((N, N), dtype=complex)
    Sigma_L_mat[0, 0] = Sigma_L
    Sigma_R_mat[-1, -1] = Sigma_R

    G_R = np.linalg.inv((E + 1j * eta) * np.eye(N) - H_device - Sigma_L_mat - Sigma_R_mat)
    G_A = G_R.conj().T

    Gamma_L = 1j * (Sigma_L_mat - Sigma_L_mat.conj().T)
    Gamma_R = 1j * (Sigma_R_mat - Sigma_R_mat.conj().T)

    T = np.trace(Gamma_L @ G_R @ Gamma_R @ G_A)
    return np.real(T)


def transmission_spectrum(E_array, H_device, t_lead, t_coupling, eta=1e-8):
    return np.array([transmission(E, H_device, t_lead, t_coupling, eta) for E in E_array])


if __name__ == "__main__":
    t = 1.0
    N = 30
    onsite = np.zeros(N)
    H = device_hamiltonian(onsite, t)

    E_array = np.linspace(-2.5, 2.5, 400)
    T_E = transmission_spectrum(E_array, H, t_lead=t, t_coupling=t)

    in_band = np.abs(E_array) < 1.9 * t
    print(f"Uniform chain check: T(E) within band, mean={T_E[in_band].mean():.6f}, "
          f"max deviation from 1 = {np.max(np.abs(T_E[in_band] - 1)):.6f}")
    print(f"Outside band (E=2.4t): T = {transmission(2.4*t, H, t, t):.6f} (should be ~0)")
