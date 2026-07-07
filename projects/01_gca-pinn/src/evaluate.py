"""
학습 결과 시각화:
1. Loss curve (baseline vs PINN, data loss vs physics loss)
2. Id-Vg 예측 곡선 비교 (fixed Vds에서 Vgs sweep) — ground truth vs baseline vs PINN
   특히 라벨 데이터가 희소한 영역에서 두 모델의 예측 품질 차이를 시각적으로 강조
"""

import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

from model import TFTNet
from physics import gca_current_torch, load_device_params
from train import normalize, load_and_prepare

HERE = os.path.dirname(__file__)
MODELS_DIR = os.path.join(HERE, "..", "models")
ASSETS_DIR = os.path.join(HERE, "..", "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)


def load_trained_models():
    baseline = TFTNet()
    baseline.load_state_dict(torch.load(os.path.join(MODELS_DIR, "baseline.pt")))
    baseline.eval()

    pinn = TFTNet()
    pinn.load_state_dict(torch.load(os.path.join(MODELS_DIR, "pinn.pt")))
    pinn.eval()

    return baseline, pinn


def plot_loss_curves():
    hist = np.load(os.path.join(MODELS_DIR, "loss_history.npz"))

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.plot(hist["baseline_total"], label="Baseline (data loss only)", color="#d62728")
    ax.plot(hist["pinn_total"], label="PINN total loss", color="#1f77b4")
    ax.plot(hist["pinn_data"], label="PINN data loss", color="#1f77b4", linestyle="--", alpha=0.6)
    ax.plot(hist["pinn_physics"], label="PINN physics loss", color="#2ca02c", linestyle="--", alpha=0.6)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("Training Loss: Baseline vs PINN")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS_DIR, "loss_curves.png"), dpi=150)
    print(f"Saved {ASSETS_DIR}/loss_curves.png")


def plot_idvg_comparison(data, baseline, pinn, id_scale=1e4, fixed_vds=15.0):
    device_params = load_device_params()

    vgs_sweep = np.linspace(0, 20, 200)
    vds_fixed = np.full_like(vgs_sweep, fixed_vds)

    vgs_n = normalize(vgs_sweep, 0, 20)
    vds_n = normalize(vds_fixed, 0, 20)
    X = torch.tensor(np.stack([vgs_n, vds_n], axis=1), dtype=torch.float32)

    with torch.no_grad():
        pred_baseline = baseline(X).numpy() / id_scale
        pred_pinn = pinn(X).numpy() / id_scale

    Id_true = gca_current_torch(
        torch.tensor(vgs_sweep, dtype=torch.float32),
        torch.tensor(vds_fixed, dtype=torch.float32),
        device_params,
    ).numpy()

    # 이 Vds 근처에 라벨 데이터가 얼마나 있었는지 표시 (희소성 시각화)
    label_idx = data["label_idx"]
    Vgs_all = data["X_all"][:, 0].numpy()
    Vds_all = data["X_all"][:, 1].numpy()
    vds_fixed_n = normalize(fixed_vds, 0, 20)
    near_mask = np.abs(Vds_all[label_idx] - vds_fixed_n) < 0.1
    n_nearby_labels = near_mask.sum()

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.plot(vgs_sweep, Id_true * 1e3, label="Ground truth (GCA)", color="black", linewidth=2)
    ax.plot(vgs_sweep, pred_baseline * 1e3, label="Baseline (data-only)", color="#d62728", linestyle="--")
    ax.plot(vgs_sweep, pred_pinn * 1e3, label="PINN (data + physics)", color="#1f77b4", linestyle="--")
    ax.set_xlabel("Vgs (V)")
    ax.set_ylabel("Id (mA)")
    ax.set_title(f"Id-Vg at Vds={fixed_vds}V  ({n_nearby_labels} nearby labeled points)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fname = f"idvg_comparison_vds{int(fixed_vds)}.png"
    fig.savefig(os.path.join(ASSETS_DIR, fname), dpi=150)
    print(f"Saved {ASSETS_DIR}/{fname}")


if __name__ == "__main__":
    data = load_and_prepare(n_label=150)
    baseline, pinn = load_trained_models()

    plot_loss_curves()
    plot_idvg_comparison(data, baseline, pinn, fixed_vds=15.0)
    plot_idvg_comparison(data, baseline, pinn, fixed_vds=5.0)

    with open(os.path.join(MODELS_DIR, "results.json")) as f:
        results = json.load(f)
    print("\nSummary:")
    print(json.dumps(results, indent=2))
