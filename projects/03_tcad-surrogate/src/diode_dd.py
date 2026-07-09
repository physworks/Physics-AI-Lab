"""
1D PN Junction Diode: Poisson-Drift-Diffusion Self-Consistent Solver

Milestone 1(Poisson-only, 전류 없음)의 다음 단계 -- 실제 전류가 흐르는
PN 접합에서 I-V 특성을 계산한다.

수치 기법:
1. Scharfetter-Gummel (SG) discretization: 전자/정공 전류밀도를 노드 간
   지수함수적으로 변하는 준-페르미 준위를 가정해 안정적으로 이산화하는
   표준 기법 (거의 모든 상용 TCAD 솔버가 채택).
2. Gummel iteration: Poisson과 Drift-Diffusion(연속방정식)을 번갈아 풀며
   self-consistent 해로 수렴시키는 고전적 decoupled 반복법 (Gummel, 1964).
   - Poisson step: n(x), p(x)를 고정하고 phi(x)에 대해 (선형) Poisson 방정식을 풂
   - DD step: phi(x)를 고정하고 n(x), p(x)에 대해 (선형) 연속방정식을 SG로 풂
   - 수렴할 때까지 반복

지배 방정식:
  Poisson:     d/dx(eps dphi/dx) = -q(p - n + Cdop(x))
  전자 연속:    dJn/dx = q*R   (여기서는 R=0, 벌크 재결합 없는 이상적 단파장 다이오드)
  정공 연속:    dJp/dx = -q*R
  SG 전자전류:  Jn_{i+1/2} = (q*Dn/h)[B(delta) n_{i+1} - B(-delta) n_i], delta=(phi_{i+1}-phi_i)/Vt
  SG 정공전류:  Jp_{i+1/2} = (q*Dp/h)[B(delta) p_i - B(-delta) p_{i+1}]
  B(x) = x / (exp(x) - 1)  (Bernoulli function)
"""

import numpy as np
from scipy.optimize import brentq

Q = 1.602176634e-19
EPS0 = 8.8541878128e-12
KB = 1.380649e-23
T = 300.0
VT = KB * T / Q

EPS_SI = 11.7 * EPS0
NI = 1.5e16          # m^-3

MU_N = 0.135          # m^2/V/s (electron mobility, Si)
MU_P = 0.048           # m^2/V/s (hole mobility, Si)
DN = MU_N * VT         # Einstein relation
DP = MU_P * VT


def bernoulli(x):
    """수치적으로 안정적인 Bernoulli function B(x) = x/(e^x - 1)."""
    x = np.clip(x, -60, 60)
    small = np.abs(x) < 1e-6
    safe_x = np.where(small, 1.0, x)
    out = np.where(small, 1 - x / 2 + x**2 / 12, safe_x / np.expm1(safe_x))
    return out


def build_diode_mesh(L, n_near=70, n_far=35, near_half_width_frac=0.01, grade=1.1):
    """접합(x=L/2) 근처는 촘촘하게, 바깥으로 갈수록 기하급수적으로 성기게."""
    xj = L / 2
    near_hw = L * near_half_width_frac
    x_near = np.linspace(-near_hw, near_hw, n_near)

    remaining = L / 2 - near_hw
    h0 = remaining * (1 - grade) / (1 - grade**n_far) if grade != 1 else remaining / n_far
    steps, h = [], h0
    for _ in range(n_far):
        steps.append(h)
        h *= grade
    offsets = np.cumsum(steps)

    x_right = near_hw + offsets
    x_left = -near_hw - offsets[::-1]
    x = np.concatenate([x_left, x_near, x_right]) + xj
    x = np.clip(x, 0, L)
    x = np.unique(np.sort(x))
    if x[0] > 1e-15:
        x = np.insert(x, 0, 0.0)
    if x[-1] < L - 1e-15:
        x = np.append(x, L)
    return x


def doping_profile(x, L, Na_val, Nd_val):
    """x < L/2: p-side (Na), x >= L/2: n-side (Nd). net doping Cdop = Nd - Na"""
    is_n_side = (x >= L / 2).astype(float)
    Na = Na_val * (1 - is_n_side)
    Nd = Nd_val * is_n_side
    return Na, Nd


def solve_equilibrium(x, Na, Nd, max_iter=100, tol=1e-10):
    """
    Va=0 평형 상태 nonlinear Poisson 방정식을 Newton-Raphson으로 풂.
    (Milestone 1과 동일한 box-integration + Newton 방식을, 공간에 따라
    변하는 net doping Cdop(x) = Nd(x)-Na(x)에 대해 일반화.)
    """
    N = len(x)
    Cdop = Nd - Na
    eps = EPS_SI
    h = np.diff(x)
    vol = np.zeros(N)
    vol[1:-1] = (h[:-1] + h[1:]) / 2
    vol[0] = h[0] / 2
    vol[-1] = h[-1] / 2

    phi_p_eq = -VT * np.log(Na[0] / NI) if Na[0] > 0 else 0.0
    phi_n_eq = VT * np.log(Nd[-1] / NI) if Nd[-1] > 0 else 0.0

    phi = np.where(x < x[-1] / 2, phi_p_eq, phi_n_eq)

    for it in range(max_iter):
        n = NI * np.exp(phi / VT)
        p = NI * np.exp(-phi / VT)
        charge = Q * (p - n + Cdop)
        dcharge = -(Q / VT) * (n + p)

        R = np.zeros(N)
        J = np.zeros((N, N))
        R[1:-1] = (eps * (phi[2:] - phi[1:-1]) / h[1:] -
                    eps * (phi[1:-1] - phi[:-2]) / h[:-1]) + charge[1:-1] * vol[1:-1]
        for i in range(1, N - 1):
            J[i, i - 1] = eps / h[i - 1]
            J[i, i + 1] = eps / h[i]
            J[i, i] = -eps / h[i - 1] - eps / h[i] + dcharge[i] * vol[i]

        R[0] = phi[0] - phi_p_eq
        J[0, 0] = 1.0
        R[-1] = phi[-1] - phi_n_eq
        J[-1, -1] = 1.0

        dphi = np.linalg.solve(J, -R)
        step_norm = np.max(np.abs(dphi))
        if step_norm > 1.0:
            dphi = dphi / step_norm
        phi = phi + dphi
        if np.max(np.abs(dphi)) < tol:
            break

    n = NI * np.exp(phi / VT)
    p = NI * np.exp(-phi / VT)
    return phi, n, p, phi_p_eq, phi_n_eq


def solve_poisson_gummel_step(x, phi_prev, n_prev, p_prev, Cdop, phi_left, phi_right,
                                 max_iter=50, tol=1e-10):
    """
    Gummel iteration의 Poisson step.

    순수하게 전하를 고정하고 선형 Poisson을 푸는 방식(Picard iteration)은
    전하가 phi에 지수적으로 민감하다는 사실을 반영하지 못해 발산한다.
    대신 표준 Gummel map 방식대로, 이전 스텝의 n_prev, p_prev로부터
    준-페르미 준위(quasi-Fermi level) phi_n, phi_p를 계산해 "고정"한 뒤,
    n(phi) = ni*exp((phi-phi_n)/Vt), p(phi) = ni*exp(-(phi-phi_p)/Vt)
    관계식을 유지하며 비선형 Poisson을 Newton-Raphson으로 푼다.
    (Milestone 1의 Newton solver와 동일한 구조, 기준점만 phi_n/phi_p로 이동)
    """
    N = len(x)
    eps = EPS_SI
    h = np.diff(x)
    vol = np.zeros(N)
    vol[1:-1] = (h[:-1] + h[1:]) / 2
    vol[0] = h[0] / 2
    vol[-1] = h[-1] / 2

    # quasi-Fermi levels (고정)
    phi_n = phi_prev - VT * np.log(n_prev / NI)
    phi_p = phi_prev + VT * np.log(p_prev / NI)

    phi = phi_prev.copy()

    for it in range(max_iter):
        n = NI * np.exp((phi - phi_n) / VT)
        p = NI * np.exp(-(phi - phi_p) / VT)
        charge = Q * (p - n + Cdop)
        dcharge = -(Q / VT) * (n + p)

        R = np.zeros(N)
        Jm = np.zeros((N, N))
        R[1:-1] = (eps * (phi[2:] - phi[1:-1]) / h[1:] -
                    eps * (phi[1:-1] - phi[:-2]) / h[:-1]) + charge[1:-1] * vol[1:-1]
        for i in range(1, N - 1):
            Jm[i, i - 1] = eps / h[i - 1]
            Jm[i, i + 1] = eps / h[i]
            Jm[i, i] = -eps / h[i - 1] - eps / h[i] + dcharge[i] * vol[i]

        R[0] = phi[0] - phi_left
        Jm[0, 0] = 1.0
        R[-1] = phi[-1] - phi_right
        Jm[-1, -1] = 1.0

        dphi = np.linalg.solve(Jm, -R)
        step_norm = np.max(np.abs(dphi))
        if step_norm > 1.0:
            dphi = dphi / step_norm
        phi = phi + dphi
        if np.max(np.abs(dphi)) < tol:
            break

    return phi


def _propagate_electron(x, phi, n_left, J):
    """
    J(상수 전류)를 가정하고 왼쪽 경계 n_left로부터 SG 관계식을 재귀적으로
    전파시켜 n(x) 전체를 계산.
    SG: Jn_{i+1/2} = (qDn/h_i)[n_{i+1}*B(delta_i) - n_i*B(-delta_i)] = J
     => n_{i+1} = ( J*h_i/(qDn) + n_i*B(-delta_i) ) / B(delta_i)
    """
    h = np.diff(x)
    delta = np.diff(phi) / VT
    n = np.empty(len(x))
    n[0] = n_left
    for i in range(len(h)):
        Bp = bernoulli(delta[i])
        Bm = bernoulli(-delta[i])
        Bp = Bp if abs(Bp) > 1e-300 else 1e-300
        n[i + 1] = (J * h[i] / (Q * DN) + n[i] * Bm) / Bp
    return n


def solve_continuity_electron(x, phi, n_left, n_right):
    """
    전류가 일정하다는 물리적 사실(dJn/dx=0, R=0)을 직접 이용하는 shooting method.
    N차원 선형계 대신, 전류 J(스칼라) 하나에 대한 root-finding으로 문제를 축소해
    n 값이 19자리 가까이 차이 나는 상황에서도 dense 행렬 풀이의 병적인
    조건수(ill-conditioning) 문제를 피한다.
    """
    def residual(J):
        return _propagate_electron(x, phi, n_left, J)[-1] - n_right

    lo, hi = -1.0, 1.0
    r_lo, r_hi = residual(lo), residual(hi)
    expand = 0
    while r_lo * r_hi > 0 and expand < 60:
        lo *= 10
        hi *= 10
        r_lo, r_hi = residual(lo), residual(hi)
        expand += 1

    J = brentq(residual, lo, hi, xtol=1e-30, rtol=1e-14, maxiter=200)
    n = _propagate_electron(x, phi, n_left, J)
    return np.clip(n, 1e-6, None), J


def _propagate_hole(x, phi, p_left, J):
    """
    SG: Jp_{i+1/2} = (qDp/h_i)[p_i*B(delta_i) - p_{i+1}*B(-delta_i)] = J
     => p_{i+1} = ( p_i*B(delta_i) - J*h_i/(qDp) ) / B(-delta_i)
    """
    h = np.diff(x)
    delta = np.diff(phi) / VT
    p = np.empty(len(x))
    p[0] = p_left
    for i in range(len(h)):
        Bp = bernoulli(delta[i])
        Bm = bernoulli(-delta[i])
        Bm = Bm if abs(Bm) > 1e-300 else 1e-300
        p[i + 1] = (p[i] * Bp - J * h[i] / (Q * DP)) / Bm
    return p


def solve_continuity_hole(x, phi, p_left, p_right):
    """정공 버전 shooting method (전자와 동일한 원리)."""
    def residual(J):
        return _propagate_hole(x, phi, p_left, J)[-1] - p_right

    lo, hi = -1.0, 1.0
    r_lo, r_hi = residual(lo), residual(hi)
    expand = 0
    while r_lo * r_hi > 0 and expand < 60:
        lo *= 10
        hi *= 10
        r_lo, r_hi = residual(lo), residual(hi)
        expand += 1

    J = brentq(residual, lo, hi, xtol=1e-30, rtol=1e-14, maxiter=200)
    p = _propagate_hole(x, phi, p_left, J)
    return np.clip(p, 1e-6, None), J


def compute_current(x, phi, n, p):
    """각 edge에서 SG 전류밀도를 계산 (median으로 노이즈 완화)."""
    h = np.diff(x)
    delta = np.diff(phi) / VT
    Jn = (Q * DN / h) * (n[1:] * bernoulli(delta) - n[:-1] * bernoulli(-delta))
    Jp = (Q * DP / h) * (p[:-1] * bernoulli(delta) - p[1:] * bernoulli(-delta))
    J = Jn + Jp
    return np.median(J)


def solve_diode_iv(Va_array, L=2e-6, Na_val=1e23, Nd_val=1e23,
                     n_gummel=25, verbose=False):
    x = build_diode_mesh(L)
    Na, Nd = doping_profile(x, L, Na_val, Nd_val)
    Cdop = Nd - Na

    phi_eq, n_eq, p_eq, phi_p0, phi_n0 = solve_equilibrium(x, Na, Nd)
    J_eq = compute_current(x, phi_eq, n_eq, p_eq)

    results = []
    phi, n, p = phi_eq.copy(), n_eq.copy(), p_eq.copy()

    for Va in Va_array:
        n_p0 = NI**2 / Na_val
        p_n0 = NI**2 / Nd_val

        n_left, p_left = n_p0 * np.exp(Va / VT), Na_val
        n_right, p_right = Nd_val, p_n0 * np.exp(Va / VT)
        phi_left, phi_right = phi_p0, phi_n0 - Va

        Jn, Jp = 0.0, 0.0
        for gi in range(n_gummel):
            phi_new = solve_poisson_gummel_step(x, phi, n, p, Cdop, phi_left, phi_right)
            n_new, Jn = solve_continuity_electron(x, phi_new, n_left, n_right)
            p_new, Jp = solve_continuity_hole(x, phi_new, p_left, p_right)

            d_phi = np.max(np.abs(phi_new - phi))
            phi, n, p = phi_new, n_new, p_new
            if d_phi < 1e-9:
                break

        J = Jn + Jp
        results.append(J)
        if verbose:
            print(f"Va={Va:+.3f}V  Jn={Jn:.4e}  Jp={Jp:.4e}  J={J:.4e} A/m^2  "
                  f"(Gummel iters: {gi+1})")

    return x, np.array(results), J_eq


if __name__ == "__main__":
    Va_array = np.concatenate([
        np.linspace(-0.3, 0.0, 4),
        np.linspace(0.05, 0.7, 14),
    ])
    x, J, J_eq = solve_diode_iv(Va_array, verbose=True)
    print(f"\nEquilibrium (Va=0) current check: J_eq = {J_eq:.3e} A/m^2 (should be ~0)")
