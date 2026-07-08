"""
1D MOS Capacitor Poisson Equation Solver (box-integration / FEM, Newton-Raphson)

TCAD의 가장 기본 단위인 MOS 구조에서, 게이트 전압에 따른 반도체 표면의
band bending을 비선형 Poisson 방정식을 풀어 계산한다.

지배 방정식 (평형, Boltzmann 통계 가정):
  d/dx( eps(x) dphi/dx ) = -q * [p(phi) - n(phi) - Na(x)]
  n(phi) = ni * exp(phi / Vt)
  p(phi) = ni * exp(-phi / Vt)

이산화: box-integration (제어체적) 방식 — Sentaurus를 포함한 대부분의
상용 TCAD 툴이 실제로 사용하는 이산화 기법. 1D에서는 선형 FEM과
수학적으로 동일한 결과를 준다 (등가성 확인은 README 참고).

경계조건:
  x=0 (게이트):        phi = Vg (Dirichlet, ideal MOS, flatband voltage=0 가정)
  x=L (bulk contact):  phi = phi_p (Dirichlet, ohmic contact, 평형 bulk potential)
"""

import numpy as np

# ---------------------------------------------------------------
# 물리 상수 및 재료 파라미터
# ---------------------------------------------------------------
Q = 1.602176634e-19       # C, elementary charge
EPS0 = 8.8541878128e-12   # F/m, vacuum permittivity
KB = 1.380649e-23         # J/K
T = 300.0                 # K
VT = KB * T / Q           # thermal voltage (~0.02585 V at 300K)

EPS_SI = 11.7 * EPS0      # silicon permittivity
EPS_OX = 3.9 * EPS0       # SiO2 permittivity
NI = 1.5e16                # m^-3, silicon intrinsic carrier concentration (300K)


def build_mesh(t_ox, t_semi, n_ox=15, n_semi_near=60, n_semi_far=40,
                near_frac=0.05, grade=1.08):
    """
    비균일 메시 생성.
    - 산화막: charge-free이므로 균일 메시로 충분
    - 반도체: 계면 근처(Debye length 스케일)는 매우 촘촘하게,
      멀어질수록 기하급수적으로 성기게 (grading)
    """
    x_ox = np.linspace(0, t_ox, n_ox, endpoint=False)

    near_len = t_semi * near_frac
    x_near = np.linspace(0, near_len, n_semi_near, endpoint=False)

    # 나머지 구간은 기하급수적으로 증가하는 간격
    remaining = t_semi - near_len
    steps = []
    h = remaining / n_semi_far * (1 - grade) / (1 - grade**n_semi_far) if grade != 1 else remaining / n_semi_far
    pos = 0.0
    for _ in range(n_semi_far):
        steps.append(h)
        pos += h
        h *= grade
    x_far = near_len + np.cumsum([0] + steps[:-1])

    x_semi = np.concatenate([x_near, x_far]) + t_ox
    x = np.concatenate([x_ox, x_semi, [t_ox + t_semi]])
    x = np.unique(np.sort(x))
    return x


def material_region(x, t_ox):
    """각 노드가 산화막(0)인지 반도체(1)인지 표시"""
    return (x >= t_ox - 1e-15).astype(int)


def charge_and_deriv(phi, is_semi, Na):
    """
    노드별 net charge density (C/m^3)와 phi에 대한 미분.
    산화막 노드는 charge=0.
    """
    n = NI * np.exp(phi / VT)
    p = NI * np.exp(-phi / VT)
    charge = Q * (p - n - Na) * is_semi
    dcharge_dphi = -(Q / VT) * (n + p) * is_semi
    return charge, dcharge_dphi


def solve_poisson(Vg, t_ox=5e-9, t_semi=3e-7, Na_val=1e23, max_iter=100, tol=1e-10):
    """
    주어진 게이트 전압 Vg에서 비선형 Poisson 방정식을 Newton-Raphson으로 풀어
    전위 분포 phi(x)를 반환.

    Na_val: acceptor doping (m^-3), p-type 기판
    """
    x = build_mesh(t_ox, t_semi)
    N = len(x)
    is_semi = material_region(x, t_ox)
    Na = Na_val * is_semi

    eps = np.where(x < t_ox, EPS_OX, EPS_SI)
    # edge별 유효 permittivity (두 노드 중간 지점 재질 기준)
    eps_edge = np.where((x[:-1] + x[1:]) / 2 < t_ox, EPS_OX, EPS_SI)
    h = np.diff(x)  # edge 길이

    # 평형 bulk potential (p-type, 전하중성조건)
    phi_p = -VT * np.log(Na_val / NI)

    # 초기값: 선형 보간
    phi = np.linspace(Vg, phi_p, N)

    # control volume 크기 (각 노드 주변 절반씩)
    vol = np.zeros(N)
    vol[1:-1] = (h[:-1] + h[1:]) / 2
    vol[0] = h[0] / 2
    vol[-1] = h[-1] / 2

    for it in range(max_iter):
        charge, dcharge = charge_and_deriv(phi, is_semi, Na)

        R = np.zeros(N)
        J = np.zeros((N, N))

        # 내부 노드: box-integration flux balance
        flux_L = eps_edge[:-1] * (phi[1:-1] - phi[:-2]) / h[:-1]   # 왼쪽 edge flux (into node)
        flux_R = eps_edge[1:] * (phi[2:] - phi[1:-1]) / h[1:]      # 오른쪽 edge flux (out of node)
        R[1:-1] = (flux_R - flux_L) + charge[1:-1] * vol[1:-1]

        for i in range(1, N - 1):
            J[i, i - 1] = eps_edge[i - 1] / h[i - 1]
            J[i, i + 1] = eps_edge[i] / h[i]
            J[i, i] = -eps_edge[i - 1] / h[i - 1] - eps_edge[i] / h[i] + dcharge[i] * vol[i]

        # Dirichlet 경계조건
        R[0] = phi[0] - Vg
        J[0, 0] = 1.0
        R[-1] = phi[-1] - phi_p
        J[-1, -1] = 1.0

        try:
            dphi = np.linalg.solve(J, -R)
        except np.linalg.LinAlgError:
            break

        # damping (수렴 안정성을 위해 스텝 크기 제한)
        step_norm = np.max(np.abs(dphi))
        if step_norm > 1.0:
            dphi = dphi * (1.0 / step_norm)

        phi = phi + dphi

        if np.max(np.abs(dphi)) < tol:
            break

    return x, phi, is_semi, vol, phi_p, it


def total_semiconductor_charge(x, phi, is_semi, vol, Na_val, t_ox):
    """반도체 영역의 총 net charge (C/m^2 단위, 1D이므로 면전하밀도)"""
    Na = Na_val * is_semi
    n = NI * np.exp(phi / VT)
    p = NI * np.exp(-phi / VT)
    charge = Q * (p - n - Na) * is_semi
    return np.sum(charge * vol)


if __name__ == "__main__":
    print(f"Vt = {VT:.5f} V, ni = {NI:.2e} m^-3")
    x, phi, is_semi, vol, phi_p, iters = solve_poisson(Vg=1.0)
    print(f"Converged in {iters} iterations")
    print(f"phi_p (bulk potential) = {phi_p:.4f} V")
    print(f"Surface potential (phi at oxide/semi interface) = {phi[is_semi.argmax()]:.4f} V")
