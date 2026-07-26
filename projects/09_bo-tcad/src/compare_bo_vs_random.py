"""
목표: 특정 target Vth를 만족하는 (t_ox_nm, Na_cm3)를 찾는 문제에서,
Bayesian Optimization과 Random Search를 같은 평가 횟수(예산) 조건에서 비교.

TCAD 시뮬레이터(tcad_sim.simulate_vth)는 호출 1회에 ~150-250ms가 걸리는
"비용이 큰" 블랙박스로 취급 — 실제 TCAD 워크플로우에서 평가 횟수를
최소화하는 것이 핵심 동기.

각 방법을 여러 random seed로 반복해 평균 수렴 곡선(best-so-far loss vs
평가 횟수)을 비교한다.
"""

import numpy as np
import time
from tcad_sim import simulate_vth
from gp_bo import bayesian_optimize

# 설계 변수 범위: t_ox [2, 10] nm, Na [1e17, 1e18] cm^-3 (log 스케일로 최적화)
BOUNDS = [(2.0, 10.0), (17.0, 18.0)]  # 두 번째 변수는 log10(Na_cm3)

TARGET_VTH = 0.6  # V, 목표 문턱전압


def objective(x):
    t_ox, log_na = x
    na = 10 ** log_na
    vth = simulate_vth(t_ox, na, n_vg=20)  # BO 중에는 속도를 위해 n_vg 축소
    return (vth - TARGET_VTH) ** 2


def random_search(bounds, n_evals, seed):
    rng = np.random.default_rng(seed)
    X = np.array([[rng.uniform(lo, hi) for lo, hi in bounds] for _ in range(n_evals)])
    y = np.array([objective(x) for x in X])
    return X, y


def best_so_far(y):
    return np.minimum.accumulate(y)


if __name__ == "__main__":
    N_INIT = 5
    N_ITER = 15
    N_TOTAL = N_INIT + N_ITER
    N_SEEDS = 5

    print(f"Comparing Bayesian Optimization vs Random Search")
    print(f"Target Vth = {TARGET_VTH}V, total evaluations per run = {N_TOTAL}, seeds = {N_SEEDS}\n")

    bo_curves, rand_curves = [], []
    bo_times, rand_times = [], []

    for seed in range(N_SEEDS):
        t0 = time.time()
        X_bo, y_bo = bayesian_optimize(objective, BOUNDS, n_init=N_INIT, n_iter=N_ITER, seed=seed)
        bo_times.append(time.time() - t0)
        bo_curves.append(best_so_far(y_bo))

        t0 = time.time()
        X_r, y_r = random_search(BOUNDS, N_TOTAL, seed=seed + 100)
        rand_times.append(time.time() - t0)
        rand_curves.append(best_so_far(y_r))

        print(f"  seed {seed}: BO final loss={y_bo.min():.6f}, "
              f"Random final loss={y_r.min():.6f}")

    bo_curves = np.array(bo_curves)
    rand_curves = np.array(rand_curves)

    bo_mean, bo_std = bo_curves.mean(0), bo_curves.std(0)
    rand_mean, rand_std = rand_curves.mean(0), rand_curves.std(0)

    print(f"\nFinal loss (mean +/- std over {N_SEEDS} seeds):")
    print(f"  Bayesian Optimization: {bo_mean[-1]:.6f} +/- {bo_std[-1]:.6f}")
    print(f"  Random Search:         {rand_mean[-1]:.6f} +/- {rand_std[-1]:.6f}")

    # 목표 loss threshold 도달까지 필요한 평가 횟수 비교
    threshold = 1e-4
    bo_reach = np.argmax(bo_mean < threshold) if (bo_mean < threshold).any() else -1
    rand_reach = np.argmax(rand_mean < threshold) if (rand_mean < threshold).any() else -1
    print(f"\nEvaluations to reach loss < {threshold}:")
    print(f"  BO: {bo_reach+1 if bo_reach>=0 else 'not reached'}")
    print(f"  Random: {rand_reach+1 if rand_reach>=0 else 'not reached'}")

    np.savez("../models/bo_comparison.npz",
             bo_mean=bo_mean, bo_std=bo_std, rand_mean=rand_mean, rand_std=rand_std,
             n_total=N_TOTAL, target_vth=TARGET_VTH)
    print("\nSaved results.")
