"""
GCA-based TFT Id-Vg 모델 학습: PINN vs Data-only Baseline 비교

실험 설계:
- 전체 5000개 샘플 중 소량(n_label)만 "라벨 있는 실험 데이터"로 취급 (노이즈 포함)
- 나머지(라벨 없는 영역)는 collocation point로만 사용 (물리 방정식 제약 적용)
- Baseline: 라벨 데이터만으로 학습 (data loss만)
- PINN: 라벨 데이터(data loss) + 전체 영역 물리 제약(physics loss)를 함께 학습
- 라벨이 부족한 영역에서 두 모델의 예측 정확도를 비교하여 physics loss의 효과를 정량화
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn

from model import TFTNet
from physics import gca_current_torch, load_device_params

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

HERE = os.path.dirname(__file__)
DATA_PATH = os.path.join(HERE, "..", "data", "tft_idvg_dataset.npz")
MODELS_DIR = os.path.join(HERE, "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def normalize(x, lo, hi):
    return 2 * (x - lo) / (hi - lo) - 1


def load_and_prepare(n_label=150, id_scale=1e4):
    data = np.load(DATA_PATH)
    Vgs, Vds = data["Vgs"], data["Vds"]
    Id_clean, Id_noisy = data["Id_clean"], data["Id_noisy"]

    n_total = len(Vgs)
    rng = np.random.default_rng(SEED)
    label_idx = rng.choice(n_total, size=n_label, replace=False)

    Vgs_n = normalize(Vgs, 0, 20)
    Vds_n = normalize(Vds, 0, 20)

    X_all = np.stack([Vgs_n, Vds_n], axis=1).astype(np.float32)
    y_all_clean = (Id_clean * id_scale).astype(np.float32)

    X_label = X_all[label_idx]
    y_label = (Id_noisy[label_idx] * id_scale).astype(np.float32)

    # collocation points = 전체 (Vgs,Vds) 격자, 라벨 없이 물리 방정식만 적용
    Vgs_colloc_raw = torch.tensor(Vgs, dtype=torch.float32)
    Vds_colloc_raw = torch.tensor(Vds, dtype=torch.float32)

    return {
        "X_all": torch.tensor(X_all),
        "y_all_clean": torch.tensor(y_all_clean),
        "X_label": torch.tensor(X_label),
        "y_label": torch.tensor(y_label),
        "Vgs_colloc_raw": Vgs_colloc_raw,
        "Vds_colloc_raw": Vds_colloc_raw,
        "label_idx": label_idx,
        "id_scale": id_scale,
    }


def train_model(data, use_physics, n_epochs=3000, lr=1e-3, physics_weight=0.5, log_every=500):
    model = TFTNet()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    device_params = load_device_params()
    history = {"total": [], "data": [], "physics": []}

    for epoch in range(n_epochs):
        opt.zero_grad()

        pred_label = model(data["X_label"])
        data_loss = mse(pred_label, data["y_label"])

        if use_physics:
            pred_colloc = model(data["X_all"])
            id_gca_scaled = gca_current_torch(
                data["Vgs_colloc_raw"], data["Vds_colloc_raw"], device_params
            ) * data["id_scale"]
            physics_loss = mse(pred_colloc, id_gca_scaled)
            loss = data_loss + physics_weight * physics_loss
        else:
            physics_loss = torch.tensor(0.0)
            loss = data_loss

        loss.backward()
        opt.step()

        history["total"].append(loss.item())
        history["data"].append(data_loss.item())
        history["physics"].append(physics_loss.item())

        if epoch % log_every == 0 or epoch == n_epochs - 1:
            tag = "PINN " if use_physics else "Base "
            print(f"[{tag}] epoch {epoch:5d} | total {loss.item():.6f} | "
                  f"data {data_loss.item():.6f} | physics {physics_loss.item():.6f}")

    return model, history


def evaluate(model, data):
    with torch.no_grad():
        pred_all = model(data["X_all"])  # scaled units
    true_all = data["y_all_clean"]  # scaled units
    mse_all = torch.mean((pred_all - true_all) ** 2).item()

    # 라벨이 없던 영역만 따로 평가 (일반화 성능이 핵심 관심사)
    mask = np.ones(len(true_all), dtype=bool)
    mask[data["label_idx"]] = False
    mse_unlabeled = torch.mean((pred_all[mask] - true_all[mask]) ** 2).item()

    return {"mse_all": mse_all, "mse_unlabeled_region": mse_unlabeled}


if __name__ == "__main__":
    print("Loading data (sparse label regime: n_label=150 / 5000)...\n")
    data = load_and_prepare(n_label=150)

    print("=== Training Baseline (data loss only) ===")
    baseline_model, baseline_history = train_model(data, use_physics=False)

    print("\n=== Training PINN (data loss + physics loss) ===")
    pinn_model, pinn_history = train_model(data, use_physics=True)

    print("\n=== Evaluation (full dataset, clean ground truth) ===")
    baseline_metrics = evaluate(baseline_model, data)
    pinn_metrics = evaluate(pinn_model, data)

    print(f"Baseline - MSE(all): {baseline_metrics['mse_all']:.6f} | "
          f"MSE(unlabeled region): {baseline_metrics['mse_unlabeled_region']:.6f}")
    print(f"PINN     - MSE(all): {pinn_metrics['mse_all']:.6f} | "
          f"MSE(unlabeled region): {pinn_metrics['mse_unlabeled_region']:.6f}")

    improvement = (
        (baseline_metrics["mse_unlabeled_region"] - pinn_metrics["mse_unlabeled_region"])
        / baseline_metrics["mse_unlabeled_region"] * 100
    )
    print(f"\nPINN improvement over baseline (unlabeled region MSE): {improvement:.1f}%")

    torch.save(baseline_model.state_dict(), os.path.join(MODELS_DIR, "baseline.pt"))
    torch.save(pinn_model.state_dict(), os.path.join(MODELS_DIR, "pinn.pt"))

    results = {
        "baseline_metrics": baseline_metrics,
        "pinn_metrics": pinn_metrics,
        "improvement_pct": improvement,
        "n_label": 150,
        "n_total": len(data["X_all"]),
    }
    with open(os.path.join(MODELS_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    np.savez(
        os.path.join(MODELS_DIR, "loss_history.npz"),
        baseline_total=baseline_history["total"],
        baseline_data=baseline_history["data"],
        pinn_total=pinn_history["total"],
        pinn_data=pinn_history["data"],
        pinn_physics=pinn_history["physics"],
    )

    print(f"\nSaved models and results to {MODELS_DIR}/")
