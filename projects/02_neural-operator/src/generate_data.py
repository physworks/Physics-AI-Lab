"""
Neural Operator 학습을 위한 데이터 생성.

기존 GCA-PINN 프로젝트는 "고정된 하나의 소자"에 대해 (Vgs, Vds) -> Id를
점 단위로 예측하는 PINN이었다. 여기서는 한 걸음 나아가, Vth(문턱전압)가
서로 다른 여러 소자에 대해 "Vgs grid 전체에 대한 Id-Vg 곡선 함수"를
한 번에 매핑하는 Neural Operator(FNO)를 학습한다.

Operator learning의 핵심: 학습에 없던 새로운 Vth 값에 대해서도
재학습 없이 전체 Id-Vg 곡선을 예측할 수 있어야 한다 (구조/파라미터 일반화).
"""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "gca-pinn", "src"))
from generate_data import gca_current, DEVICE_PARAMS

HERE = os.path.dirname(__file__)
N_GRID = 128          # Vgs grid resolution (fixed Vds sweep 대신 Vgs sweep 곡선)
VDS_FIXED = 10.0       # 이번 실험은 고정 Vds에서의 Id-Vg 곡선에 대해 Vth 일반화를 본다
VGS_RANGE = (0, 20)


def generate_operator_dataset(vth_train, vth_test, seed=42):
    """
    vth_train, vth_test: Vth 값의 배열 (서로 겹치지 않음, test는 학습에 없던 값)
    각 Vth에 대해 Vgs grid 전체에서의 Id 곡선(clean)을 생성.
    """
    vgs_grid = np.linspace(VGS_RANGE[0], VGS_RANGE[1], N_GRID).astype(np.float32)
    vds_grid = np.full_like(vgs_grid, VDS_FIXED)

    def curves_for(vth_array):
        curves = []
        for vth in vth_array:
            params = dict(DEVICE_PARAMS)
            params["Vth"] = float(vth)
            Id = gca_current(vgs_grid, vds_grid, params=params, noise_std=0.0)
            curves.append(Id.astype(np.float32))
        return np.stack(curves)  # (n_samples, N_GRID)

    train_curves = curves_for(vth_train)
    test_curves = curves_for(vth_test)

    return {
        "vgs_grid": vgs_grid,
        "vth_train": np.array(vth_train, dtype=np.float32),
        "vth_test": np.array(vth_test, dtype=np.float32),
        "train_curves": train_curves,
        "test_curves": test_curves,
    }


if __name__ == "__main__":
    # 학습: Vth = 0.5 ~ 3.5V 중 15개 값 / 테스트: 학습에 없던 4개 값으로 일반화 검증
    rng = np.random.default_rng(42)
    vth_train = np.sort(rng.uniform(0.5, 3.5, 15))
    vth_test = np.array([0.75, 1.9, 2.6, 3.3])  # 학습 구간 내부의 held-out 값들

    data = generate_operator_dataset(vth_train, vth_test)

    out_dir = os.path.join(HERE, "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    np.savez(
        os.path.join(out_dir, "fno_vth_dataset.npz"),
        **data
    )
    print(f"Train curves: {data['train_curves'].shape}, Vth range {vth_train.min():.2f}~{vth_train.max():.2f}")
    print(f"Test curves (held-out Vth): {data['test_curves'].shape}, Vth values {vth_test}")
    print(f"Saved to {out_dir}/fno_vth_dataset.npz")
