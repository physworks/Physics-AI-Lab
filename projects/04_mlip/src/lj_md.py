"""
Lennard-Jones (LJ) 포텐셜 기반 분자동역학(MD) 시뮬레이터.

실제 MLIP 파이프라인에서는 DFT/ab-initio 계산이 "정답 데이터"를 제공하지만,
여기서는 그 자리를 해석적으로 알려진 LJ 포텐셜이 대신한다 — GCA-PINN
프로젝트에서 GCA 방정식이 "정답"역할을 했던 것과 동일한 패턴.

LJ potential (reduced units, epsilon=1, sigma=1, mass=1 — Frenkel & Smit
"Understanding Molecular Simulation" 등 MD 교재의 표준 관행):
    U(r) = 4*eps*[(sigma/r)^12 - (sigma/r)^6]

적분: Velocity Verlet + 간단한 velocity-rescaling thermostat (canonical
sampling으로 다양한 온도/구성의 학습 데이터를 확보하기 위함)
"""

import numpy as np

EPS = 1.0
SIGMA = 1.0
MASS = 1.0
CUTOFF = 3.0 * SIGMA  # 계산 비용을 위해 cutoff 이후 상호작용 무시


def lj_energy_forces(pos, eps=EPS, sigma=SIGMA, cutoff=CUTOFF):
    """
    pos: (N, 3) 배열
    반환: total_energy (scalar), forces (N, 3), per_atom_energy (N,)
    (per_atom_energy는 쌍에너지를 두 원자에 절반씩 분배 — MLIP 학습 타겟용)
    """
    N = len(pos)
    diff = pos[:, None, :] - pos[None, :, :]        # (N, N, 3)
    r = np.linalg.norm(diff, axis=-1)                # (N, N)
    np.fill_diagonal(r, np.inf)                       # 자기 자신 제외

    mask = r < cutoff
    sr6 = (sigma / np.where(mask, r, np.inf)) ** 6
    sr12 = sr6 ** 2
    pair_energy = 4 * eps * (sr12 - sr6)               # (N, N), symmetric
    pair_energy = np.where(mask, pair_energy, 0.0)

    total_energy = 0.5 * np.sum(pair_energy)           # 이중 카운트 보정
    per_atom_energy = 0.5 * np.sum(pair_energy, axis=1)

    # Force: F_i = -dU/dr_i = sum_j [24*eps/r * (2*sr12 - sr6)] * (diff_ij / r)
    coef = np.where(mask, 24 * eps / np.where(mask, r, np.inf) * (2 * sr12 - sr6), 0.0)
    forces = np.sum((coef / np.where(mask, r, np.inf))[:, :, None] * diff, axis=1)

    return total_energy, forces, per_atom_energy


def init_cluster(n_atoms, box_size=4.0, seed=0):
    """무작위 초기 배치 (겹치지 않도록 최소 거리 확보), 초기 속도는 0."""
    rng = np.random.default_rng(seed)
    pos = np.zeros((n_atoms, 3))
    placed = 0
    while placed < n_atoms:
        cand = rng.uniform(-box_size / 2, box_size / 2, size=3)
        if placed == 0 or np.min(np.linalg.norm(pos[:placed] - cand, axis=1)) > 0.9 * SIGMA:
            pos[placed] = cand
            placed += 1
    vel = np.zeros((n_atoms, 3))
    return pos, vel


def velocity_verlet_md(pos, vel, n_steps, dt=0.002, target_temp=0.3,
                          thermostat_every=20, record_every=10, seed=0):
    """
    Velocity Verlet 적분 + 간단한 velocity-rescaling thermostat.
    record_every 스텝마다 (pos, energy, forces, per_atom_energy) 스냅샷 저장.
    """
    rng = np.random.default_rng(seed)
    N = len(pos)

    # 초기 속도를 목표 온도로 무작위 초기화 (Maxwell-Boltzmann 근사, reduced units: kT=target_temp)
    vel = rng.normal(0, np.sqrt(target_temp / MASS), size=(N, 3))
    vel -= vel.mean(axis=0)  # 전체 운동량 제거

    energy, forces, per_atom = lj_energy_forces(pos)

    trajectory = {"pos": [], "energy": [], "forces": [], "per_atom_energy": [], "temp": []}

    for step in range(n_steps):
        vel_half = vel + 0.5 * dt * forces / MASS
        pos = pos + dt * vel_half
        energy, forces, per_atom = lj_energy_forces(pos)
        vel = vel_half + 0.5 * dt * forces / MASS

        # 온도 계산 (등분배 정리, reduced units: kT = 2*KE/(3N))
        ke = 0.5 * MASS * np.sum(vel ** 2)
        current_temp = 2 * ke / (3 * N)

        if step % thermostat_every == 0 and current_temp > 1e-8:
            scale = np.sqrt(target_temp / current_temp)
            vel *= scale

        if step % record_every == 0:
            trajectory["pos"].append(pos.copy())
            trajectory["energy"].append(energy)
            trajectory["forces"].append(forces.copy())
            trajectory["per_atom_energy"].append(per_atom.copy())
            trajectory["temp"].append(current_temp)

    for k in trajectory:
        trajectory[k] = np.array(trajectory[k])
    return trajectory


if __name__ == "__main__":
    N_ATOMS = 20
    pos, vel = init_cluster(N_ATOMS, box_size=4.5, seed=1)

    print("Running MD trajectories at multiple temperatures...")
    all_traj = {}
    BURN_IN = 30  # 초반 non-equilibrium 완화(collapse) 구간 제외
    for temp, seed in [(0.2, 10), (0.4, 11), (0.6, 12), (0.8, 13)]:
        pos0, vel0 = init_cluster(N_ATOMS, box_size=4.5, seed=seed)
        traj = velocity_verlet_md(pos0, vel0, n_steps=6000, target_temp=temp,
                                     record_every=15, seed=seed)
        for k in traj:
            traj[k] = traj[k][BURN_IN:]
        all_traj[temp] = traj
        print(f"  T={temp}: {len(traj['energy'])} snapshots (burn-in {BURN_IN} removed), "
              f"E range [{traj['energy'].min():.2f}, {traj['energy'].max():.2f}]")

    # 학습용: T=0.2, 0.4, 0.6 / 검증(held-out)용: T=0.8
    train_pos = np.concatenate([all_traj[t]["pos"] for t in [0.2, 0.4, 0.6]])
    train_E = np.concatenate([all_traj[t]["energy"] for t in [0.2, 0.4, 0.6]])
    train_F = np.concatenate([all_traj[t]["forces"] for t in [0.2, 0.4, 0.6]])

    test_pos = all_traj[0.8]["pos"]
    test_E = all_traj[0.8]["energy"]
    test_F = all_traj[0.8]["forces"]

    np.savez("../data/lj_md_dataset.npz",
             train_pos=train_pos, train_E=train_E, train_F=train_F,
             test_pos=test_pos, test_E=test_E, test_F=test_F,
             n_atoms=N_ATOMS)
    print(f"\nSaved: train {len(train_E)} snapshots, test (T=0.8, held-out) {len(test_E)} snapshots")
    print("-> ../data/lj_md_dataset.npz")
