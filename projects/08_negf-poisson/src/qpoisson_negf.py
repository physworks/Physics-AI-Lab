"""
Self-consistent Schrodinger(NEGF)-Poisson solver for a quantum MOS capacitor.

프로젝트 03(Poisson/Drift-Diffusion)과 프로젝트 05(NEGF 양자수송)의 결합.
반전층(inversion layer)의 소수캐리어(전자)를 classical Boltzmann 대신
NEGF로 양자역학적으로 계산해, 실제 device 물리의 유명한 현상인
"quantum confinement로 인한 반전층 전하의 계면으로부터의 setback"을 재현한다.

모델 구조 (03 Milestone 1과 동일 기하 + 05의 리드 self-energy):
- Poisson 경계조건: x=0(계면)에 게이트 전압 Vg(Dirichlet), x=L(bulk)에 평형 bulk potential
- 양자역학 경계조건: x=0은 hard-wall(반사 경계, 산화막 장벽이 충분히 높다는 표준 근사),
  x=L은 05에서 만든 반무한 리드(self-energy)로 열림 — Poisson과 NEGF가 서로 다른
  경계조건 근사를 쓰는 것은 실제 Schrodinger-Poisson MOS 솔버에서도 흔한 관행
  (전기적 경계조건과 파동함수 경계조건은 별개로 근사 가능)

전자는 effective-mass tight-binding으로 이산화 (hopping t = hbar^2/(2 m* a^2)),
정공은 여전히 classical Boltzmann (반전층 물리에서 핵심은 소수캐리어인 전자의
양자화이므로, 이 비대칭적 처리는 실제 문헌에서도 표준적인 단순화).

자기 일관성: NEGF 밀도는 phi에 대한 해석적 미분(dn/dphi)을 구하기 어려워
Newton 대신 damped fixed-point iteration(under-relaxation)으로 수렴시킨다.
"""

import numpy as np

# --- 물리 상수 ---
Q = 1.602176634e-19
EPS0 = 8.8541878128e-12
KB = 1.380649e-23
HBAR = 1.054571817e-34
M0 = 9.1093837015e-31
T = 300.0
VT = KB * T / Q  # eV 단위 열전압

EPS_SI = 11.7 * EPS0
NI = 1.5e16  # m^-3

M_STAR = 0.19 * M0  # Si 전도대 유효질량 (single-valley 근사, 실제는 6-valley)
A_GRID = 0.3e-9     # m, tight-binding 격자 간격
HBAR2_2M0_EV_ANG2 = 3.81  # eV*Angstrom^2, hbar^2/(2*m0)의 잘 알려진 상수

# tight-binding hopping (eV 단위), a를 Angstrom으로 변환해서 계산
A_ANG = A_GRID * 1e10
T_HOP = HBAR2_2M0_EV_ANG2 / (M_STAR / M0) / A_ANG ** 2  # eV

# 유효상태밀도 Nc (3D, Boltzmann 근사와의 calibration에 사용)
NC = 2 * (M_STAR * KB * T / (2 * np.pi * HBAR ** 2)) ** 1.5  # m^-3

NA_VAL = 1e23  # m^-3, p-type bulk doping


def bernoulli_like_greens_function(E, t, eta=1e-8):
    """반무한 1D tight-binding 리드의 표면 Green's function (analytic closed form).
    (프로젝트 05의 surface_greens_function과 동일한 물리, 여기서는 onsite 기준을
    호출부에서 에너지 시프트로 처리)"""
    Ec = E + 1j * eta
    disc = np.asarray(4 * t ** 2 - Ec ** 2, dtype=complex)
    sqrt_term = np.sqrt(disc)
    g = (Ec - 1j * sqrt_term) / (2 * t ** 2)
    if np.imag(g) > 0:
        g = (Ec + 1j * sqrt_term) / (2 * t ** 2)
    return g


def fermi(E, EF, kT):
    x = np.clip((E - EF) / kT, -60, 60)
    return 1.0 / (1.0 + np.exp(x))


def build_grid(L_nm, a_nm=A_ANG / 10):
    N = int(round(L_nm / a_nm))
    x = np.arange(1, N + 1) * a_nm  # 1..N (site 0은 hard-wall 밖의 가상 노드)
    return x, N


def classical_hole_density(phi):
    return NI * np.exp(-phi / VT)


def classical_electron_density_boltzmann(phi, Ec_offset, EF=0.0):
    """calibration/비교용: Nc 기반 classical Boltzmann 전자밀도."""
    Ec = -phi + Ec_offset  # eV 단위 (phi는 V, 여기서는 q=1 편의상 생략, 아래서 일괄 처리)
    return NC * np.exp(-(Ec - EF) / VT)


def negf_electron_density(phi, Ec_offset, EF, E_grid, a_nm, eta=1e-3):
    """
    phi: (N,) 전위 프로파일 (V)
    Ec_offset: bulk 기준 전도대 edge (calibration 상수, eV)
    EF: 전역 페르미 준위 (eV, =0 기준)
    E_grid: 적분할 에너지 격자 (eV)
    eta: retarded Green's function의 수치적 broadening (eV). 너무 작으면
         (~1e-6) 준속박상태 근처에서 행렬 조건수가 나빠져 인접 site 간
         비물리적인 진동(수치 노이즈)이 발생함 — 1e-3 eV(t_hop 대비 매우
         작지만 수치적으로 충분히 안정적인 값)로 조정.
    반환: n(x) (m^-3)
    """
    N = len(phi)
    Ec_site = -phi + Ec_offset  # 각 site의 전도대 edge (eV)

    n = np.zeros(N)
    dE = E_grid[1] - E_grid[0]

    Ec_bulk = Ec_site[-1]  # 오른쪽 끝 site 기준으로 리드 onsite 정렬

    for E in E_grid:
        H = np.diag(Ec_site).astype(complex)
        for i in range(N - 1):
            H[i, i + 1] = -T_HOP
            H[i + 1, i] = -T_HOP

        Sigma = np.zeros((N, N), dtype=complex)
        g_surf = bernoulli_like_greens_function(E - Ec_bulk, T_HOP, eta=eta)
        Sigma[-1, -1] = T_HOP ** 2 * g_surf

        G_R = np.linalg.inv((E + 1j * eta) * np.eye(N) - H - Sigma)
        A_spec = 1j * (G_R - G_R.conj().T)  # spectral function diag = local DOS
        dos_local = np.real(np.diag(A_spec)) / (2 * np.pi)  # states per energy per site

        f = fermi(E, EF, VT)
        n += dos_local * f * dE

    # tight-binding 1D 사이트당 값을 3D 밀도(m^-3)로 변환: 사이트 간격 a로 나눠
    # "단위 길이당 상태수 -> 3D 밀도"로 근사 변환 (단면적 1 m^2 가정, 단순화 명시)
    # 단위 주의: a_nm은 nm 단위 숫자이므로 m로 변환 후 나눔 (단위 버그 방지)
    a_m = a_nm * 1e-9
    n_3d = n / a_m

    # 공간 평활화: hard-wall 경계와 이산 격자가 만나 생기는 격자 스케일(주기 ~2 site)
    # 정상파 간섭 패턴(Friedel 진동과 유사한 현상, 단일 에너지에서도 나타남을 직접 확인)이
    # 존재하는데, 이는 Poisson 방정식이 다뤄야 할 물리적 스케일(격자 간격보다 훨씬 큼)보다
    # 미세하다. 격자 스케일 잔물결을 제거하기 위해 3-site moving average 적용 —
    # 연속체(effective-mass) 해석과 일관된 표준적 후처리.
    pad = 3
    n_padded = np.pad(n_3d, pad, mode='reflect')
    kernel = np.ones(7) / 7
    n_smooth = np.convolve(n_padded, kernel, mode='same')[pad:-pad]

    return n_smooth


if __name__ == "__main__":
    print(f"t_hop = {T_HOP:.4f} eV, Nc = {NC:.3e} m^-3, Vt = {VT:.5f} eV")
    x, N = build_grid(15.0, a_nm=A_ANG / 10)
    print(f"N sites = {N}, domain length = {x[-1]:.2f} nm")


EC_OFFSET = VT * np.log(NC / NI)  # eV, classical Boltzmann과 NI*exp(phi/Vt) 규약을 정확히 일치시키는 calibration


def classical_poisson_newton(Vg, Na_val, x_full_nm, max_iter=300, tol=1e-11):
    """
    프로젝트 03 Milestone 1과 동일한 box-integration + Newton-Raphson.
    x_full_nm: (N+2,) 경계 포함 전체 mesh, 단위 nm (0번=게이트 접점, -1번=bulk 접점)
    반환: phi (N+2,) 전체 프로파일, phi_p_eq (bulk 평형 potential)

    주의(단위 버그 수정 이력): 초기 구현에서 nm 단위 좌표를 SI 공식(EPS_SI는
    F/m 기준)에 변환 없이 그대로 넣어, 결합(flux)항이 실제보다 10^9배 작아지는
    단위 버그가 있었다 — 전하항이 부당하게 지배해 "전위 강하가 첫 셀에만
    집중되는" 비물리적 해가 나왔다. x_full_m = x_full_nm*1e-9로 명시적 변환.
    """
    x_full = x_full_nm * 1e-9  # nm -> m
    N_full = len(x_full)
    eps = EPS_SI
    h = np.diff(x_full)
    vol = np.zeros(N_full)
    vol[1:-1] = (h[:-1] + h[1:]) / 2
    vol[0] = h[0] / 2
    vol[-1] = h[-1] / 2

    phi_p_eq = -VT * np.log(Na_val / NI)
    phi = np.linspace(Vg, phi_p_eq, N_full)

    for it in range(max_iter):
        n = NI * np.exp(phi / VT)
        p = NI * np.exp(-phi / VT)
        charge = Q * (p - n - Na_val)
        dcharge = -(Q / VT) * (n + p)

        R = np.zeros(N_full)
        J = np.zeros((N_full, N_full))
        R[1:-1] = (eps * (phi[2:] - phi[1:-1]) / h[1:] -
                    eps * (phi[1:-1] - phi[:-2]) / h[:-1]) + charge[1:-1] * vol[1:-1]
        for i in range(1, N_full - 1):
            J[i, i - 1] = eps / h[i - 1]
            J[i, i + 1] = eps / h[i]
            J[i, i] = -eps / h[i - 1] - eps / h[i] + dcharge[i] * vol[i]

        R[0] = phi[0] - Vg
        J[0, 0] = 1.0
        R[-1] = phi[-1] - phi_p_eq
        J[-1, -1] = 1.0

        dphi = np.linalg.solve(J, -R)
        dphi = np.clip(dphi, -0.3, 0.3)  # element-wise damping
        phi = phi + dphi
        if np.max(np.abs(dphi)) < tol:
            break

    return phi, phi_p_eq


def solve_linear_poisson_fixed_charge(x_full_nm, charge_density, Vg, phi_bulk):
    """전하를 고정한 상태에서 (선형) Poisson 방정식을 풂 — self-consistent loop의 한 스텝.
    x_full_nm: nm 단위 좌표 (내부에서 m로 변환)"""
    x_full = x_full_nm * 1e-9
    N_full = len(x_full)
    eps = EPS_SI
    h = np.diff(x_full)
    vol = np.zeros(N_full)
    vol[1:-1] = (h[:-1] + h[1:]) / 2
    vol[0] = h[0] / 2
    vol[-1] = h[-1] / 2

    A = np.zeros((N_full, N_full))
    b = np.zeros(N_full)
    for i in range(1, N_full - 1):
        A[i, i - 1] = eps / h[i - 1]
        A[i, i + 1] = eps / h[i]
        A[i, i] = -eps / h[i - 1] - eps / h[i]
        b[i] = -charge_density[i] * vol[i]

    A[0, 0] = 1.0
    b[0] = Vg
    A[-1, -1] = 1.0
    b[-1] = phi_bulk

    return np.linalg.solve(A, b)
