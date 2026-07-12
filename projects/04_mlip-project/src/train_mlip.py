"""
MLIP 학습: 에너지 손실 + 힘(force) 손실을 함께 사용하는 dual loss.

GCA-PINN에서 data loss + physics loss를 함께 쓴 것과 개념적으로 유사하다:
에너지만 맞추도록 학습하면 힘의 방향이 부정확해도 손실에 안 잡힐 수 있으므로,
force loss를 추가로 걸어 원자 간 상호작용의 미분 구조(기울기)까지 학습하도록
강제한다. 실제 MLIP 학습에서도 force loss가 안정성에 핵심적이라고 알려져 있다.
"""

import numpy as np
import torch
import torch.nn as nn
from mlip_model import MLIP

torch.manual_seed(42)

d = np.load("../data/lj_md_dataset.npz")
SUBSAMPLE = 3  # 학습 시간 단축을 위해 궤적에서 매 3번째 스냅샷만 사용
train_pos = torch.tensor(d["train_pos"][::SUBSAMPLE], dtype=torch.float32)   # (n_samples, N, 3)
train_E = torch.tensor(d["train_E"][::SUBSAMPLE], dtype=torch.float32)
train_F = torch.tensor(d["train_F"][::SUBSAMPLE], dtype=torch.float32)
test_pos = torch.tensor(d["test_pos"][::SUBSAMPLE], dtype=torch.float32)
test_E = torch.tensor(d["test_E"][::SUBSAMPLE], dtype=torch.float32)
test_F = torch.tensor(d["test_F"][::SUBSAMPLE], dtype=torch.float32)
N_ATOMS = int(d["n_atoms"])

print(f"Train samples: {len(train_E)}, Test (held-out T=0.8) samples: {len(test_E)}")

model = MLIP()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
mse = nn.MSELoss()

FORCE_WEIGHT = 3.0
BATCH = 32
n_epochs = 80
n_train = len(train_E)

history = {"total": [], "energy": [], "force": []}

for epoch in range(n_epochs):
    model.train()
    perm = torch.randperm(n_train)
    epoch_loss, epoch_eloss, epoch_floss = 0.0, 0.0, 0.0
    n_batches = 0

    for start in range(0, n_train, BATCH):
        idx = perm[start:start + BATCH]
        opt.zero_grad()

        batch_e_loss, batch_f_loss = 0.0, 0.0
        for i in idx:
            pos_i = train_pos[i].clone().requires_grad_(True)
            E_pred, F_pred = model(pos_i)
            batch_e_loss = batch_e_loss + (E_pred - train_E[i]) ** 2
            batch_f_loss = batch_f_loss + torch.mean((F_pred - train_F[i]) ** 2)

        n = len(idx)
        e_loss = batch_e_loss / n
        f_loss = batch_f_loss / n
        loss = e_loss + FORCE_WEIGHT * f_loss

        loss.backward()
        opt.step()

        epoch_loss += loss.item()
        epoch_eloss += e_loss.item()
        epoch_floss += f_loss.item()
        n_batches += 1

    history["total"].append(epoch_loss / n_batches)
    history["energy"].append(epoch_eloss / n_batches)
    history["force"].append(epoch_floss / n_batches)

    if epoch % 15 == 0 or epoch == n_epochs - 1:
        print(f"epoch {epoch:4d} | total {history['total'][-1]:.4f} | "
              f"E_loss {history['energy'][-1]:.4f} | F_loss {history['force'][-1]:.4f}")

# --- 평가 (held-out T=0.8) ---
model.eval()
pred_E_list, pred_F_list = [], []
for i in range(len(test_E)):
    pos_i = test_pos[i].clone().requires_grad_(True)
    E_pred, F_pred = model(pos_i)
    pred_E_list.append(E_pred.item())
    pred_F_list.append(F_pred.detach().numpy())

pred_E = np.array(pred_E_list)
pred_F = np.array(pred_F_list)
true_E = test_E.numpy()
true_F = test_F.numpy()

e_rmse = np.sqrt(np.mean((pred_E - true_E) ** 2))
f_rmse = np.sqrt(np.mean((pred_F - true_F) ** 2))
e_mae_per_atom = np.mean(np.abs(pred_E - true_E)) / N_ATOMS

print(f"\nHeld-out (T=0.8, unseen temperature) evaluation:")
print(f"  Energy RMSE: {e_rmse:.4f}  (MAE/atom: {e_mae_per_atom:.4f})")
print(f"  Force RMSE:  {f_rmse:.4f}")

torch.save(model.state_dict(), "../models/mlip.pt")
np.savez("../models/mlip_results.npz",
         history_total=history["total"], history_energy=history["energy"],
         history_force=history["force"],
         pred_E=pred_E, true_E=true_E, pred_F=pred_F, true_F=true_F,
         e_rmse=e_rmse, f_rmse=f_rmse)
print("Saved model and results.")
