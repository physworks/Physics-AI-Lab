"""
GCA (Gradual Channel Approximation) 기반 TFT Id-Vg synthetic 데이터 생성

물리 모델:
- 선형 영역 (Vds < Vgs - Vth):
    Id = mu * Cox * (W/L) * [(Vgs - Vth) * Vds - Vds^2 / 2]
- 포화 영역 (Vds >= Vgs - Vth):
    Id = 0.5 * mu * Cox * (W/L) * (Vgs - Vth)^2 * (1 + lambda * Vds)

TFT는 이상적인 MOSFET 대비 비선형성이 커서(subthreshold swing, mobility
degradation 등), 이후 Neural Operator 확장 단계에서 이 비선형 항을
데이터 기반으로 보정하는 것을 목표로 함.
"""

import numpy as np
import json
import os

# ---------------------------------------------------------------
# 디바이스 파라미터 (a-Si / oxide TFT 대표값 기반 가정치)
# ---------------------------------------------------------------
DEVICE_PARAMS = {
    "mu": 10.0,      # cm^2/V*s, mobility (oxide TFT 대표값)
    "Cox": 3.45e-8,  # F/cm^2, gate oxide capacitance (SiO2 100nm 기준)
    "W": 50.0,       # um, channel width
    "L": 10.0,       # um, channel length
    "Vth": 1.5,      # V, threshold voltage
    "lambda_": 0.02, # 1/V, channel length modulation
}


def gca_current(Vgs, Vds, params=DEVICE_PARAMS, noise_std=0.0):
    """
    GCA 방정식에 따라 Id를 계산.
    Vgs, Vds: numpy array (동일 shape)
    returns: Id in Amperes
    """
    mu = params["mu"] * 1e-4  # cm^2/Vs -> m^2/Vs
    Cox = params["Cox"] * 1e4  # F/cm^2 -> F/m^2
    W = params["W"] * 1e-6  # um -> m
    L = params["L"] * 1e-6  # um -> m
    Vth = params["Vth"]
    lam = params["lambda_"]

    Vov = Vgs - Vth  # overdrive voltage
    Vov = np.clip(Vov, 0, None)  # cutoff 영역 (Vgs < Vth) -> Id = 0

    Id = np.zeros_like(Vgs)

    linear_mask = Vds < Vov
    sat_mask = ~linear_mask

    # 선형 영역
    Id[linear_mask] = (
        mu * Cox * (W / L) *
        (Vov[linear_mask] * Vds[linear_mask] - Vds[linear_mask] ** 2 / 2)
    )

    # 포화 영역
    # continuity correction: lambda 보정을 Vds가 아닌 (Vds - Vov)에 적용하여
    # 경계(Vds = Vov)에서 선형식과 정확히 연속되도록 함 (SPICE Level-1 방식)
    Id[sat_mask] = (
        mu * Cox * (W / L) *
        Vov[sat_mask] ** 2 / 2 * (1 + lam * (Vds[sat_mask] - Vov[sat_mask]))
    )

    # cutoff: Vov == 0인 지점은 자동으로 Id = 0 처리됨

    if noise_std > 0:
        # 측정 노이즈를 모사 (실제 실험 데이터와의 유사성을 위해)
        Id = Id * (1 + np.random.normal(0, noise_std, size=Id.shape))

    return Id


def generate_dataset(n_samples=5000, noise_std=0.02, seed=42):
    """
    Vgs in [0, 20V], Vds in [0, 20V] 범위에서 랜덤 샘플링하여
    (Vgs, Vds, Id) 데이터셋 생성.
    """
    rng = np.random.default_rng(seed)

    Vgs = rng.uniform(0, 20, n_samples)
    Vds = rng.uniform(0, 20, n_samples)

    Id_clean = gca_current(Vgs, Vds, noise_std=0.0)
    Id_noisy = gca_current(Vgs, Vds, noise_std=noise_std)

    dataset = {
        "Vgs": Vgs,
        "Vds": Vds,
        "Id_clean": Id_clean,
        "Id_noisy": Id_noisy,
    }
    return dataset


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    dataset = generate_dataset(n_samples=5000, noise_std=0.02)

    np.savez(
        os.path.join(out_dir, "tft_idvg_dataset.npz"),
        Vgs=dataset["Vgs"],
        Vds=dataset["Vds"],
        Id_clean=dataset["Id_clean"],
        Id_noisy=dataset["Id_noisy"],
    )

    with open(os.path.join(out_dir, "device_params.json"), "w") as f:
        json.dump(DEVICE_PARAMS, f, indent=2)

    print(f"Generated {len(dataset['Vgs'])} samples")
    print(f"Id range: [{dataset['Id_clean'].min():.3e}, {dataset['Id_clean'].max():.3e}] A")
    print(f"Saved to {out_dir}/tft_idvg_dataset.npz")