"""
NVIDIA PhysicsNeMo의 공식 FNO(Fourier Neural Operator) 구현을 사용해,
서로 다른 Vth(문턱전압)를 가진 소자들의 Id-Vg 곡선을 한 번에 학습하는
Neural Operator를 만든다.

입력: Vth 값을 Vgs grid 전체에 constant field로 broadcast한 1채널 함수
출력: 해당 Vth에서의 Id-Vg 곡선 (Vgs grid 전체에 대한 함수)

기존 GCA-PINN(점 단위 회귀, 하나의 고정 소자)과 달리, 여기서는
"소자 파라미터가 바뀌어도 재학습 없이 대응하는 mapping"을 학습한다 —
이것이 PINN과 Neural Operator의 핵심 차이.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from physicsnemo.models.fno import FNO

HERE = os.path.dirname(__file__)
DATA_PATH = os.path.join(HERE, "..", "data", "fno_vth_dataset.npz")
MODELS_DIR = os.path.join(HERE, "..", "models")
ASSETS_DIR = os.path.join(HERE, "..", "assets")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

torch.manual_seed(42)


def load_data():
    d = np.load(DATA_PATH)
    return d


def make_tensors(vth_array, curves, id_scale=1e4):
    """
    FNO 입력 형식: (batch, in_channels=1, N_GRID)
    Vth 값을 grid 전체에 걸쳐 constant broadcast.
    """
    n, N = curves.shape
    x = np.zeros((n, 1, N), dtype=np.float32)
    for i, vth in enumerate(vth_array):
        x[i, 0, :] = vth
    y = (curves * id_scale)[:, None, :].astype(np.float32)  # (n, 1, N)
    return torch.tensor(x), torch.tensor(y)


if __name__ == "__main__":
    d = load_data()
    vgs_grid = d["vgs_grid"]
    vth_train, vth_test = d["vth_train"], d["vth_test"]
    train_curves, test_curves = d["train_curves"], d["test_curves"]

    id_scale = 1e4
    X_train, Y_train = make_tensors(vth_train, train_curves, id_scale)
    X_test, Y_test = make_tensors(vth_test, test_curves, id_scale)

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

    n_epochs = 500
    history = []
    for epoch in range(n_epochs):
        opt.zero_grad()
        pred = model(X_train)
        loss = mse(pred, Y_train)
        loss.backward()
        opt.step()
        history.append(loss.item())
        if epoch % 50 == 0 or epoch == n_epochs - 1:
            print(f"epoch {epoch:4d} | train MSE {loss.item():.6f}")

    # held-out Vth 값들에 대한 일반화 성능 평가
    model.eval()
    with torch.no_grad():
        pred_test = model(X_test)
        test_mse = mse(pred_test, Y_test).item()
    print(f"\nHeld-out Vth generalization MSE: {test_mse:.6f}")
    print(f"Held-out Vth values tested: {vth_test.tolist()}")

    torch.save(model.state_dict(), os.path.join(MODELS_DIR, "fno_vth.pt"))
    np.savez(
        os.path.join(MODELS_DIR, "fno_results.npz"),
        history=np.array(history),
        test_mse=test_mse,
        vgs_grid=vgs_grid,
        vth_test=vth_test,
        pred_test=pred_test.numpy() / id_scale,
        true_test=Y_test.numpy() / id_scale,
    )
    print(f"Saved model and results to {MODELS_DIR}/")
