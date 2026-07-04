"""
GCA (Gradual Channel Approximation) 방정식의 PyTorch 미분가능 버전.
physics loss 계산 및 collocation point 평가에 사용.

generate_data.py의 numpy 버전과 동일한 물리 모델을 사용하되,
autograd 그래프 안에서 사용 가능하도록 torch 연산으로 재구현.
"""

import torch
import json
import os


def load_device_params(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "..", "data", "device_params.json")
    with open(path, "r") as f:
        return json.load(f)


def gca_current_torch(Vgs, Vds, params):
    """
    Vgs, Vds: torch.Tensor (same shape)
    params: dict (device parameters, SI 변환 전 원본 단위)
    returns: Id (Amperes), torch.Tensor
    """
    mu = params["mu"] * 1e-4       # cm^2/Vs -> m^2/Vs
    Cox = params["Cox"] * 1e4      # F/cm^2 -> F/m^2
    W = params["W"] * 1e-6         # um -> m
    L = params["L"] * 1e-6         # um -> m
    Vth = params["Vth"]
    lam = params["lambda_"]

    Vov = torch.clamp(Vgs - Vth, min=0.0)

    linear_mask = (Vds < Vov).float()
    sat_mask = 1.0 - linear_mask

    Id_linear = mu * Cox * (W / L) * (Vov * Vds - Vds ** 2 / 2)
    Id_sat = 0.5 * mu * Cox * (W / L) * Vov ** 2 * (1 + lam * Vds)

    Id = linear_mask * Id_linear + sat_mask * Id_sat
    return Id
