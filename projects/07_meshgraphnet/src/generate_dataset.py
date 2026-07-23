"""
front_sim.py의 시뮬레이터로 여러 초기 형상(범프 위치/높이가 다른)에 대한
궤적을 생성하고, MeshGraphNet 학습에 필요한 그래프 형식(node feature,
edge feature, senders/receivers)으로 변환한다.

그래프 구조: 경계면의 점들을 체인(chain) 그래프로 연결 (각 점 - 이웃 점 양방향 edge).
Node feature: 현재 위치, 곡률, boundary 여부(양 끝점 고정 마스크)
Edge feature: 이웃 노드 간 상대 위치 (dx, dy) — MeshGraphNet의 표준 관행
              (절대 좌표 대신 상대 좌표를 edge feature로 사용하면 평행이동에 불변)
Target: 다음 스텝의 변위 (dx, dy)
"""

import numpy as np
from front_sim import simulate_front, make_bump_profile, curvature_and_normal

N_POINTS = 40
DT = 0.02
N_STEPS = 80
ETCH_RATE = 0.3
ALPHA = 0.05


def build_chain_edges(n_nodes):
    """체인 그래프의 양방향 edge (senders, receivers)."""
    senders = np.concatenate([np.arange(n_nodes - 1), np.arange(1, n_nodes)])
    receivers = np.concatenate([np.arange(1, n_nodes), np.arange(n_nodes - 1)])
    return senders, receivers


def make_graph_snapshot(points):
    """단일 시점의 (node_features, edge_features, senders, receivers) 생성."""
    N = len(points)
    _, kappa = curvature_and_normal(points)
    is_boundary = np.zeros(N)
    is_boundary[[0, -1]] = 1.0

    node_features = np.stack([points[:, 0], points[:, 1], kappa, is_boundary], axis=1)

    senders, receivers = build_chain_edges(N)
    edge_features = points[receivers] - points[senders]  # 상대 위치

    return node_features.astype(np.float32), edge_features.astype(np.float32), senders, receivers


if __name__ == "__main__":
    # 학습용: 범프 위치/높이가 다른 5개 궤적 / 검증용: 학습에 없던 조합 2개
    train_configs = [
        {"bump_height": 1.5, "bump_center": 0.3, "bump_width": 0.12},
        {"bump_height": 2.0, "bump_center": 0.5, "bump_width": 0.15},
        {"bump_height": 1.0, "bump_center": 0.7, "bump_width": 0.10},
        {"bump_height": 2.5, "bump_center": 0.4, "bump_width": 0.18},
        {"bump_height": 1.8, "bump_center": 0.6, "bump_width": 0.13},
    ]
    test_configs = [
        {"bump_height": 2.2, "bump_center": 0.35, "bump_width": 0.14},  # 학습에 없던 조합
    ]

    def build_dataset(configs, seed_offset=0):
        all_node_f, all_edge_f, all_senders, all_receivers, all_targets = [], [], [], [], []
        trajectories = []
        for cfg in configs:
            p0 = make_bump_profile(n_points=N_POINTS, **cfg)
            traj = simulate_front(p0, N_STEPS, DT, ETCH_RATE, ALPHA, redistribute_every=1000)
            # redistribute_every를 크게 주어 재배치가 학습 그래프 연결성을 흔들지 않게 함
            trajectories.append(traj)
            for t in range(len(traj) - 1):
                nf, ef, sd, rc = make_graph_snapshot(traj[t])
                target = (traj[t + 1] - traj[t]).astype(np.float32)
                all_node_f.append(nf)
                all_edge_f.append(ef)
                all_senders.append(sd)
                all_receivers.append(rc)
                all_targets.append(target)
        return all_node_f, all_edge_f, all_senders, all_receivers, all_targets, trajectories

    train_data = build_dataset(train_configs)
    test_data = build_dataset(test_configs)

    # test_configs가 1개뿐이면 trajectories[0]을 그대로 사용 (차원 축소 방지)
    test_traj_to_save = test_data[5][0] if len(test_data[5]) == 1 else np.array(test_data[5])

    np.savez("../data/mgn_dataset.npz",
             train_node=np.array(train_data[0]), train_edge=np.array(train_data[1]),
             train_target=np.array(train_data[4]),
             test_node=np.array(test_data[0]), test_edge=np.array(test_data[1]),
             test_target=np.array(test_data[4]),
             test_traj=test_traj_to_save,
             senders=train_data[2][0], receivers=train_data[3][0])

    print(f"Train snapshots: {len(train_data[0])} (from {len(train_configs)} trajectories x {N_STEPS} steps)")
    print(f"Test snapshots: {len(test_data[0])} (held-out bump configuration)")
    print(f"Nodes per snapshot: {N_POINTS}, edges: {len(train_data[2][0])}")
