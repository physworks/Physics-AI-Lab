"""
Milestone 3 데이터 생성: 여러 도핑 농도(Na)에 대한 Qs(Vg) 커브를
자체 Poisson 솔버로 생성한다.

Neural Operator 프로젝트(Vth 일반화)와 정확히 같은 패턴:
학습에 없던 Na 값에 대해서도 재학습 없이 전체 Qs(Vg) 커브를
예측할 수 있는 Operator를 목표로 한다.
"""

import numpy as np
import time
from poisson_1d import solve_poisson, total_semiconductor_charge

T_OX = 5e-9
T_SEMI = 3e-7
VG_GRID_N = 41
VG_RANGE = (-2.0, 2.0)


def generate_qv_curve(na_val, vg_grid, t_ox=T_OX, t_semi=T_SEMI):
    Qs = np.zeros(len(vg_grid))
    for i, vg in enumerate(vg_grid):
        x, phi, is_semi, vol, phi_p, iters = solve_poisson(
            Vg=vg, t_ox=t_ox, t_semi=t_semi, Na_val=na_val
        )
        Qs[i] = total_semiconductor_charge(x, phi, is_semi, vol, na_val, t_ox)
    return Qs


if __name__ == "__main__":
    vg_grid = np.linspace(VG_RANGE[0], VG_RANGE[1], VG_GRID_N).astype(np.float32)

    rng = np.random.default_rng(42)
    # 도핑 농도 범위: 5e16 ~ 5e17 cm^-3 (m^-3 단위로는 5e22~5e23), log-uniform 샘플링
    log_na_train = rng.uniform(np.log10(5e22), np.log10(5e23), 15)
    na_train = np.sort(10 ** log_na_train)
    na_test = np.array([7e22, 1.5e23, 2.5e23, 4e23])  # held-out 값

    print("Generating training curves (solver time measured)...")
    t0 = time.time()
    train_curves = np.stack([generate_qv_curve(na, vg_grid) for na in na_train])
    solver_time_total = time.time() - t0
    solver_time_per_curve = solver_time_total / len(na_train)
    print(f"  {len(na_train)} curves generated in {solver_time_total:.2f}s "
          f"({solver_time_per_curve*1000:.1f} ms/curve)")

    print("Generating held-out test curves...")
    test_curves = np.stack([generate_qv_curve(na, vg_grid) for na in na_test])

    np.savez(
        "../data/qv_operator_dataset.npz",
        vg_grid=vg_grid,
        na_train=na_train.astype(np.float32),
        na_test=na_test.astype(np.float32),
        train_curves=train_curves.astype(np.float32),
        test_curves=test_curves.astype(np.float32),
        solver_time_per_curve=solver_time_per_curve,
    )
    print(f"\nNa train range: {na_train.min():.2e} ~ {na_train.max():.2e} m^-3")
    print(f"Na test (held-out): {na_test}")
    print("Saved to ../data/qv_operator_dataset.npz")
