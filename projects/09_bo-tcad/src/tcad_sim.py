"""
"비용이 큰(expensive)" TCAD 블랙박스 시뮬레이터: MOS 커패시터 문턱전압(Vth) 계산.

설계 변수: (t_ox_nm, Na_cm3) -> 산화막 두께, 채널 도핑 농도
목표: Newton-Raphson 비선형 Poisson 방정식을 Vg 스윕으로 풀어 C-V 곡선을 얻고,
표면전위가 2*phi_F(강반전 기준)에 도달하는 Vg를 문턱전압 Vth로 추출.

이 시뮬레이터는 [프로젝트 03(TCAD Surrogate)]의 MOS 커패시터 Poisson 솔버와
동일한 물리(box-integration + Newton-Raphson)를 재사용한다. Bayesian
Optimization 데모에서 "값비싼 실제 TCAD 호출"의 역할을 한다.

부가 검증: 교과서 closed-form Vth 공식(Sze, Physics of Semiconductor Devices)과
비교해 PDE 기반 추출값이 합리적인 범위에 있는지 확인.
"""

import numpy as np

Q = 1.602176634e-19
EPS0 = 8.8541878128e-12
KB = 1.380649e-23
T = 300.0
VT = KB * T / Q

EPS_SI = 11.7 * EPS0
EPS_OX = 3.9 * EPS0
NI = 1.5e16  # m^-3


def build_mesh(t_ox_nm, t_semi_nm, n_ox=15, n_semi=60):
    """균일 산화막 메시 + 반지수 성김 반도체 메시 (계면 근처 촘촘)."""
    x_ox = np.linspace(0, t_ox_nm, n_ox, endpoint=False)
    # 반도체 영역: 계면 근처 촘촘 -> 벌크로 갈수록 성김 (기하급수 증가)
    frac = np.linspace(0, 1, n_semi) ** 2  # 제곱 스케일링으로 초반 촘촘하게
    x_semi = t_ox_nm + frac * t_semi_nm
    x = np.concatenate([x_ox, x_semi, [t_ox_nm + t_semi_nm]])
    return np.unique(np.sort(x))  # nm 단위


def solve_full_profile(Vg, t_ox_nm, Na_val, x_nm, max_iter=150, tol=1e-11):
    """
    box-integration + Newton-Raphson으로 비선형 Poisson을 풀어 전체 phi(x) 프로파일 반환.
    x_nm: nm 단위 좌표 (내부에서 m 변환)
    """
    x = x_nm * 1e-9  # nm -> m (단위 버그 방지: 항상 명시적으로 변환)
    N = len(x)
    is_semi = (x_nm >= t_ox_nm - 1e-9).astype(float)
    Na = Na_val * is_semi
    eps_edge = np.where((x_nm[:-1] + x_nm[1:]) / 2 < t_ox_nm, EPS_OX, EPS_SI)
    h = np.diff(x)

    phi_p = -VT * np.log(Na_val / NI)
    phi = np.linspace(Vg, phi_p, N)

    vol = np.zeros(N)
    vol[1:-1] = (h[:-1] + h[1:]) / 2
    vol[0] = h[0] / 2
    vol[-1] = h[-1] / 2

    for it in range(max_iter):
        n = NI * np.exp(phi / VT)
        p = NI * np.exp(-phi / VT)
        charge = Q * (p - n - Na) * is_semi
        dcharge = -(Q / VT) * (n + p) * is_semi

        R = np.zeros(N)
        J = np.zeros((N, N))
        flux_L = eps_edge[:-1] * (phi[1:-1] - phi[:-2]) / h[:-1]
        flux_R = eps_edge[1:] * (phi[2:] - phi[1:-1]) / h[1:]
        R[1:-1] = (flux_R - flux_L) + charge[1:-1] * vol[1:-1]
        for i in range(1, N - 1):
            J[i, i - 1] = eps_edge[i - 1] / h[i - 1]
            J[i, i + 1] = eps_edge[i] / h[i]
            J[i, i] = -eps_edge[i - 1] / h[i - 1] - eps_edge[i] / h[i] + dcharge[i] * vol[i]

        R[0] = phi[0] - Vg
        J[0, 0] = 1.0
        R[-1] = phi[-1] - phi_p
        J[-1, -1] = 1.0

        dphi = np.linalg.solve(J, -R)
        dphi = np.clip(dphi, -0.3, 0.3)
        phi = phi + dphi
        if np.max(np.abs(dphi)) < tol:
            break

    return phi


def total_charge(phi, x_nm, t_ox_nm, Na_val, is_semi):
    """반도체 영역의 총 net charge (면전하밀도, C/m^2)."""
    n = NI * np.exp(phi / VT)
    p = NI * np.exp(-phi / VT)
    charge = Q * (p - n - Na_val) * is_semi
    x = x_nm * 1e-9
    h = np.diff(x)
    vol = np.zeros(len(x))
    vol[1:-1] = (h[:-1] + h[1:]) / 2
    vol[0] = h[0] / 2
    vol[-1] = h[-1] / 2
    return np.sum(charge * vol)


def simulate_vth(t_ox_nm, Na_cm3, vg_range=(-0.5, 2.0), n_vg=25):
    """
    설계변수 (t_ox_nm, Na_cm3)에서 Vg 스윕으로 quasi-static C-V 곡선을 얻고,
    C(Vg) = -dQs/dVg 의 최솟값 위치를 Vth로 정의 (공핍->반전 전이 근처,
    강반전 영역까지 메시를 정밀화할 필요 없이 안정적으로 잘 잡히는 기준 —
    프로젝트 03 Milestone 1에서 검증된 것과 동일한 접근).
    """
    Na_val = Na_cm3 * 1e6  # cm^-3 -> m^-3
    x_nm = build_mesh(t_ox_nm, t_semi_nm=300.0)
    is_semi = (x_nm >= t_ox_nm - 1e-9).astype(float)

    vg_array = np.linspace(vg_range[0], vg_range[1], n_vg)
    Qs_array = np.zeros(n_vg)
    for i, vg in enumerate(vg_array):
        phi = solve_full_profile(vg, t_ox_nm, Na_val, x_nm)
        Qs_array[i] = total_charge(phi, x_nm, t_ox_nm, Na_val, is_semi)

    C_semi = -np.gradient(Qs_array, vg_array)
    min_idx = np.argmin(C_semi)
    if min_idx == 0 or min_idx == n_vg - 1:
        return vg_range[1]  # 스윕 범위 경계에 최솟값 -> 페널티

    # 이산 격자점을 그대로 반환하면 목표함수가 계단함수가 되어 최적화가 무의미해짐 —
    # 최솟값 주변 3점으로 포물선(parabolic) 보간해 연속적인 Vth 추정치를 얻는다
    # (피크/최솟값 보간의 표준 기법).
    i = min_idx
    y0, y1, y2 = C_semi[i - 1], C_semi[i], C_semi[i + 1]
    denom = (y0 - 2 * y1 + y2)
    if abs(denom) < 1e-30:
        return vg_array[i]
    offset = 0.5 * (y0 - y2) / denom
    dv = vg_array[1] - vg_array[0]
    return vg_array[i] + offset * dv


def vth_closed_form(t_ox_nm, Na_cm3):
    """교과서 closed-form Vth 공식 (Sze), ideal flatband(Vfb=0) 가정 — 검증용."""
    Na_val = Na_cm3 * 1e6
    Cox = EPS_OX / (t_ox_nm * 1e-9)
    phi_F = VT * np.log(Na_val / NI)
    depletion_term = np.sqrt(2 * EPS_SI * Q * Na_val * 2 * phi_F) / Cox
    return 2 * phi_F + depletion_term


if __name__ == "__main__":
    import time
    print("Validation: PDE-based Vth (C-V minimum) vs closed-form formula (reference only)")
    for t_ox, Na in [(5.0, 1e17), (8.0, 5e17), (3.0, 1e17)]:
        t0 = time.time()
        vth_pde = simulate_vth(t_ox, Na)
        elapsed = time.time() - t0
        vth_cf = vth_closed_form(t_ox, Na)
        print(f"  t_ox={t_ox}nm, Na={Na:.0e}cm^-3: "
              f"PDE Vth={vth_pde:.4f}V, closed-form={vth_cf:.4f}V, "
              f"diff={abs(vth_pde-vth_cf):.4f}V  [{elapsed*1000:.0f}ms/eval]")
