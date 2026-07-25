"""
1D box-integration Poisson solver (project 03과 같은 방식) + NEGF와의
self-consistent 결합.

Dirichlet 경계조건 U=0을 양 끝(리드)에 부여 — 리드가 이상적인 금속성
저장소(reservoir)로서 전위를 고정한다는 표준 가정.

전하 불균형(charge imbalance) q*(N_D(x) - n(x))를 소스로 하는 Poisson
방정식을 풀어 전위 보정 delta_U를 얻고, 이를 damped linear mixing으로
매 iteration마다 조금씩만 반영해 self-consistent 루프를 안정화한다
(mixing 없이 그대로 업데이트하면 전하-전위 되먹임으로 발산하는 경우가 많음 —
project 03의 Gummel iteration에서 겪은 안정성 문제와 같은 종류의 도전과제).
"""

import numpy as np


def build_poisson_matrix(N, coupling):
    """
    -d2U/dx2 = coupling * charge_imbalance 를 box-integration(FD와 동일)으로
    이산화한 행렬. Dirichlet U=0 양 끝.
    """
    A = np.zeros((N, N))
    A[0, 0] = 1.0
    A[-1, -1] = 1.0
    for i in range(1, N - 1):
        A[i, i - 1] = -1.0
        A[i, i + 1] = -1.0
        A[i, i] = 2.0
    return A


def solve_poisson(charge_imbalance, coupling):
    """
    charge_imbalance: (N,) = N_D(x) - n(x)
    coupling: 무차원 결합 상수 (실제 단위에서 q^2/(eps*a^2)에 해당하는 스케일)
    """
    N = len(charge_imbalance)
    A = build_poisson_matrix(N, coupling)
    b = coupling * charge_imbalance
    b[0] = 0.0
    b[-1] = 0.0
    U = np.linalg.solve(A, b)
    return U
