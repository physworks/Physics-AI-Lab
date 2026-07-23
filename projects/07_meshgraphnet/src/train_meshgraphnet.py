"""
MeshGraphNet 학습: 다음 스텝 변위(displacement) 예측.

평가는 두 단계로 나눈다:
1. One-step error: 실제 이전 스텝에서 한 스텝만 예측했을 때의 오차 (학습 loss와 동일 조건)
2. Rollout error: held-out 궤적에서 초기 프로파일만 주고, 예측된 변위를 반복 적용해
   여러 스텝을 자기회귀적으로(autoregressive) 진행시켰을 때 실제 궤적과의 오차.
   이는 실제 MeshGraphNet 문헌에서도 잘 알려진 핵심 난제(rollout 시 오차 누적)를
   드러내는 평가이며, SK하이닉스 블로그에서 "매 iteration마다 재메싱, material feature
   업데이트가 필요했다"고 언급한 것과 같은 이유(누적 오차 문제)로 연결된다.
"""

import numpy as np
import torch
import torch.nn as nn
from meshgraphnet_model import MeshGraphNet
from generate_dataset import build_chain_edges, make_graph_snapshot, N_POINTS

torch.manual_seed(0)

d = np.load("../data/mgn_dataset.npz")
train_node = torch.tensor(d["train_node"], dtype=torch.float32)   # (n_snap, N, 4)
train_edge = torch.tensor(d["train_edge"], dtype=torch.float32)   # (n_snap, E, 2)
train_target = torch.tensor(d["train_target"], dtype=torch.float32)  # (n_snap, N, 2)
test_node = torch.tensor(d["test_node"], dtype=torch.float32)
test_edge = torch.tensor(d["test_edge"], dtype=torch.float32)
test_target = torch.tensor(d["test_target"], dtype=torch.float32)
test_traj = d["test_traj"]  # (n_steps+1, N, 2)
senders = torch.tensor(d["senders"], dtype=torch.long)
receivers = torch.tensor(d["receivers"], dtype=torch.long)

n_train = len(train_node)
print(f"Train snapshots: {n_train}, Test snapshots: {len(test_node)}")

# 정규화 (node/edge feature, target 각각 스케일이 다르므로)
node_mean, node_std = train_node.mean((0, 1)), train_node.std((0, 1)) + 1e-8
edge_mean, edge_std = train_edge.mean((0, 1)), train_edge.std((0, 1)) + 1e-8
target_mean, target_std = train_target.mean((0, 1)), train_target.std((0, 1)) + 1e-8


def normalize(x, mean, std):
    return (x - mean) / std


model = MeshGraphNet(node_in_dim=4, edge_in_dim=2, out_dim=2,
                      latent_dim=32, hidden_dim=48, n_message_passing=6)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
mse = nn.MSELoss()

BATCH = 16
n_epochs = 100
history = []

for epoch in range(n_epochs):
    perm = torch.randperm(n_train)
    epoch_loss = 0.0
    n_batches = 0
    for start in range(0, n_train, BATCH):
        idx = perm[start:start + BATCH]
        opt.zero_grad()
        batch_loss = 0.0
        for i in idx:
            nf = normalize(train_node[i], node_mean, node_std)
            ef = normalize(train_edge[i], edge_mean, edge_std)
            pred = model(nf, ef, senders, receivers)
            target_n = normalize(train_target[i], target_mean, target_std)
            batch_loss = batch_loss + mse(pred, target_n)
        batch_loss = batch_loss / len(idx)
        batch_loss.backward()
        opt.step()
        epoch_loss += batch_loss.item()
        n_batches += 1
    history.append(epoch_loss / n_batches)
    if epoch % 10 == 0 or epoch == n_epochs - 1:
        print(f"epoch {epoch:4d} | train loss {history[-1]:.5f}")

# --- One-step 평가 (held-out bump 궤적, 실제 이전 스텝을 그대로 입력으로 사용) ---
model.eval()
with torch.no_grad():
    onestep_errs = []
    for i in range(len(test_node)):
        nf = normalize(test_node[i], node_mean, node_std)
        ef = normalize(test_edge[i], edge_mean, edge_std)
        pred_n = model(nf, ef, senders, receivers)
        pred = pred_n * target_std + target_mean
        err = torch.mean((pred - test_target[i]) ** 2).item()
        onestep_errs.append(err)
    onestep_mse = np.mean(onestep_errs)

# --- Rollout 평가 (초기 프로파일만 주고 자기회귀적으로 전체 궤적 재구성) ---
with torch.no_grad():
    current = torch.tensor(test_traj[0], dtype=torch.float32)
    rollout_positions = [current.numpy()]
    for t in range(len(test_traj) - 1):
        kappa_dummy = torch.zeros(len(current))  # placeholder, make_graph_snapshot이 numpy 기반이라 아래서 재계산
        cur_np = current.numpy()
        nf_np, ef_np, sd, rc = make_graph_snapshot(cur_np)
        nf = normalize(torch.tensor(nf_np), node_mean, node_std)
        ef = normalize(torch.tensor(ef_np), edge_mean, edge_std)
        pred_n = model(nf, ef, senders, receivers)
        pred_disp = (pred_n * target_std + target_mean).numpy()

        new_pos = cur_np + pred_disp
        new_pos[0] = cur_np[0]   # 경계조건: 양 끝점 고정
        new_pos[-1] = cur_np[-1]
        current = torch.tensor(new_pos, dtype=torch.float32)
        rollout_positions.append(new_pos)

    rollout_positions = np.array(rollout_positions)
    rollout_mse = np.mean((rollout_positions - test_traj) ** 2)

print(f"\nHeld-out bump configuration evaluation:")
print(f"  One-step MSE:  {onestep_mse:.6f}")
print(f"  Rollout MSE:   {rollout_mse:.6f}  (over {len(test_traj)-1} autoregressive steps)")
print(f"  Rollout/One-step ratio: {rollout_mse/onestep_mse:.1f}x "
      f"(error accumulation over autoregressive rollout — a known MeshGraphNet challenge)")

torch.save(model.state_dict(), "../models/model_mgn.pt")
np.savez("../models/results_mgn.npz",
         history=np.array(history), onestep_mse=onestep_mse, rollout_mse=rollout_mse,
         rollout_positions=rollout_positions, test_traj=test_traj)
print("\nSaved model and results.")
