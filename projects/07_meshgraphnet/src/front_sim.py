"""
2D 경계면(front) 곡률 흐름 시뮬레이터 — MeshGraphNet 학습용 ground truth 생성기.

물리적 배경: 식각(etching) 공정에서 경계면의 국소 속도는 종종 곡률에 의존한다
(볼록한 곳은 빠르게, 오목한 곳은 느리게 식각되는 경향) — 이는 SK하이닉스
TCAD Intelligence 팀 블로그에서 다룬 식각 프로파일 예측 문제의 단순화 버전이다.

경계면을 순서가 있는 점들의 열(open curve, front-tracking 방식)로 표현하고,
각 점을 국소 법선 방향으로 다음 속도로 이동시킨다:

    v_normal = etch_rate - alpha * kappa

kappa: 국소 곡률 (유한차분으로 근사)
alpha > 0: 곡률 흐름 계수 (양수면 볼록한 곳이 더 빨리 식각되어 매끄러워짐,
           이는 표준 mean curvature flow와 동일한 부호 관례)

검증: etch_rate=0, 원호(circular arc) 초기조건으로 두면 순수 curvature flow가
되어, 반지름이 시간에 따라 일정한 속도(dR/dt = -alpha/R * R = -alpha, 정확히는
2D 곡선의 curvature flow에서 원의 반지름 R은 dR/dt = -alpha/R 을 따름)로
줄어드는 해석해와 비교할 수 있다.
"""

import numpy as np


def curvature_and_normal(points):
    """
    points: (N, 2) open curve 점들.
    반환: normals (N,2) 외향 법선, kappa (N,) 국소 곡률 (유한차분 근사)
    양 끝점은 이웃이 하나뿐이므로 근사치 사용.
    """
    N = len(points)
    tangent = np.zeros((N, 2))
    tangent[1:-1] = points[2:] - points[:-2]
    tangent[0] = points[1] - points[0]
    tangent[-1] = points[-1] - points[-2]
    seg_len = np.linalg.norm(tangent, axis=1, keepdims=True)
    seg_len[seg_len < 1e-12] = 1e-12
    tangent_unit = tangent / seg_len

    # 법선: 접선을 90도 회전 (곡선이 위쪽을 향하도록: 표면이 아래에서 위로 식각된다고 가정)
    normal = np.stack([-tangent_unit[:, 1], tangent_unit[:, 0]], axis=1)

    # 곡률: 두 번째 차분 (내부점만 정확, 끝점은 0으로 근사)
    kappa = np.zeros(N)
    d2 = points[2:] - 2 * points[1:-1] + points[:-2]
    ds = np.linalg.norm(points[2:] - points[:-2], axis=1) / 2
    ds[ds < 1e-12] = 1e-12
    kappa_signed = np.einsum('ij,ij->i', d2, normal[1:-1]) / (ds ** 2)
    kappa[1:-1] = kappa_signed

    return normal, kappa


def redistribute(points, target_n=None):
    """호의 길이 기준으로 점을 균등 재배치 (front-tracking의 표준 기법,
    점들이 한쪽으로 쏠리는 것을 방지)."""
    if target_n is None:
        target_n = len(points)
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc = np.concatenate([[0], np.cumsum(seg)])
    total = arc[-1]
    if total < 1e-9:
        return points.copy()
    new_arc = np.linspace(0, total, target_n)
    new_x = np.interp(new_arc, arc, points[:, 0])
    new_y = np.interp(new_arc, arc, points[:, 1])
    return np.stack([new_x, new_y], axis=1)


def simulate_front(points0, n_steps, dt, etch_rate, alpha, redistribute_every=5):
    """
    points0: (N,2) 초기 경계면. 양 끝점은 고정(fixed boundary, x축 값 유지)한다.
    반환: trajectory, shape (n_steps+1, N, 2)
    """
    points = points0.copy()
    N = len(points)
    traj = [points.copy()]

    for step in range(n_steps):
        normal, kappa = curvature_and_normal(points)
        v_normal = etch_rate - alpha * kappa
        new_points = points + dt * v_normal[:, None] * normal

        # 양 끝점 고정 (경계조건)
        new_points[0] = points[0]
        new_points[-1] = points[-1]

        points = new_points
        if (step + 1) % redistribute_every == 0:
            fixed_left, fixed_right = points[0].copy(), points[-1].copy()
            points = redistribute(points, N)
            points[0], points[-1] = fixed_left, fixed_right

        traj.append(points.copy())

    return np.array(traj)


def make_bump_profile(n_points=40, width=10.0, bump_height=2.0, bump_center=0.5, bump_width=0.15):
    """평평한 시작선 위에 가우시안 범프가 있는 초기 프로파일 (오목/볼록 시작점 다양화용)."""
    x = np.linspace(0, width, n_points)
    xc = bump_center * width
    y = bump_height * np.exp(-((x - xc) ** 2) / (2 * (bump_width * width) ** 2))
    return np.stack([x, y], axis=1)


if __name__ == "__main__":
    # --- 물리 검증: 순수 curvature flow에서 반원의 반지름 감소 ---
    # 초기 형태: 반원 (etch_rate=0으로 두면 순수 mean curvature flow)
    R0 = 3.0
    n_pts = 60
    theta = np.linspace(0, np.pi, n_pts)
    x0 = R0 * np.cos(theta) + R0
    y0 = R0 * np.sin(theta)
    points0 = np.stack([x0, y0], axis=1)

    alpha = 0.05
    dt = 0.02
    n_steps = 100
    traj = simulate_front(points0, n_steps, dt, etch_rate=0.0, alpha=alpha, redistribute_every=5)

    # 각 시점에서 "반지름"을 정점(peak) 높이로 근사 추정 (반원이므로 peak height = R)
    heights = traj[:, :, 1].max(axis=1)
    times = np.arange(n_steps + 1) * dt

    # 해석해: 2D curve의 mean curvature flow에서 원의 반지름 R(t)는
    # dR/dt = -alpha/R  =>  R(t)^2 = R0^2 - 2*alpha*t
    R_analytic = np.sqrt(np.clip(R0 ** 2 - 2 * alpha * times, 0, None))

    rel_err = np.abs(heights - R_analytic) / R0
    print(f"Curvature flow validation (semicircle shrinkage):")
    print(f"  R0={R0}, alpha={alpha}")
    print(f"  Final simulated height: {heights[-1]:.4f}, analytic R(t_final): {R_analytic[-1]:.4f}")
    print(f"  Max relative error over trajectory: {rel_err.max():.4f}")
