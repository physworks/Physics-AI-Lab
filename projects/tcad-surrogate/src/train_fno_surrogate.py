"""
Milestone 3: 자체 Poisson 솔버로 생성한 데이터로 PhysicsNeMo FNO 서로게이트를 학습.

Neural Operator 프로젝트(Vth 일반화)와 동일한 구조: Na(도핑 농도) 값을
constant field로 broadcast하여 입력하고, 전체 Qs(Vg) 커브를 출력으로 예측.
학습에 없던 Na 값에 대한 일반화 + 솔버 대비 추론 속도를 함께 측정한다.
"""

import time
import numpy as np
import torch
import torch.nn as nn
from physicsnemo.models.fno import FNO

torch.manual_seed(42)

d = np.load("../data/qv_operator_dataset.npz")
vg_grid = d["vg_grid"]
na_train, na_test = d["na_train"], d["na_test"]
train_curves, test_curves = d["train_curves"], d["test_curves"]
solver_time_per_curve = float(d["solver_time_per_curve"])

# 정규화: Na는 로그 스케일, 값 자체가 크므로 log10 후 스케일 조정
def norm_na(na):
    return (np.log10(na) - 22.5) / 1.0  # 대략 -0.5 ~ 1.0 범위로

# Qs 값도 매우 작은 물리 단위(C/m^2)이므로 스케일 조정
Q_SCALE = 1e5


def make_tensors(na_array, curves):
    n, N = curves.shape
    x = np.zeros((n, 1, N), dtype=np.float32)
    for i, na in enumerate(na_array):
        x[i, 0, :] = norm_na(na)
    y = (curves * Q_SCALE)[:, None, :].astype(np.float32)
    return torch.tensor(x), torch.tensor(y)


X_train, Y_train = make_tensors(na_train, train_curves)
X_test, Y_test = make_tensors(na_test, test_curves)

model = FNO(
    in_channels=1,
    out_channels=1,
    dimension=1,
    latent_channels=32,
    num_fno_layers=4,
    num_fno_modes=16,
    padding=8,
    decoder_layers=1,
    decoder_layer_size=32,
)

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
mse = nn.MSELoss()

n_epochs = 800
history = []
for epoch in range(n_epochs):
    opt.zero_grad()
    pred = model(X_train)
    loss = mse(pred, Y_train)
    loss.backward()
    opt.step()
    history.append(loss.item())
    if epoch % 100 == 0 or epoch == n_epochs - 1:
        print(f"epoch {epoch:4d} | train MSE {loss.item():.6f}")

model.eval()
with torch.no_grad():
    pred_test = model(X_test)
    test_mse = mse(pred_test, Y_test).item()
print(f"\nHeld-out Na generalization MSE: {test_mse:.6f}")

# --- 속도 비교: FNO 순전파 vs Newton-Raphson 솔버 ---
n_speed_trials = 20
with torch.no_grad():
    t0 = time.time()
    for _ in range(n_speed_trials):
        _ = model(X_test)
    fno_time_per_batch = (time.time() - t0) / n_speed_trials
fno_time_per_curve = fno_time_per_batch / len(na_test)

speedup = solver_time_per_curve / fno_time_per_curve

print(f"\nSolver time per curve:  {solver_time_per_curve*1000:.2f} ms  (41 Vg points, Newton-Raphson)")
print(f"FNO time per curve:     {fno_time_per_curve*1000:.4f} ms  (single forward pass, CPU)")
print(f"Speedup:                {speedup:.0f}x")

torch.save(model.state_dict(), "../models/fno_tcad_surrogate.pt")
np.savez(
    "../models/fno_surrogate_results.npz",
    history=np.array(history),
    test_mse=test_mse,
    vg_grid=vg_grid,
    na_test=na_test,
    pred_test=(pred_test.numpy() / Q_SCALE),
    true_test=(Y_test.numpy() / Q_SCALE),
    solver_time_per_curve=solver_time_per_curve,
    fno_time_per_curve=fno_time_per_curve,
    speedup=speedup,
)
print("Saved model and results.")
