"""
TFT Id-Vg 예측을 위한 간단한 MLP.
Input: (Vgs_norm, Vds_norm) -> Output: Id_scaled

PINN과 baseline(data-only) 모두 동일한 구조를 사용하여
학습 방식(loss 구성)의 차이만으로 성능을 비교할 수 있게 함.
"""

import torch
import torch.nn as nn


class TFTNet(nn.Module):
    def __init__(self, hidden_dim=64, n_layers=4):
        super().__init__()
        layers = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers += [nn.Linear(hidden_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (N, 2) -> [Vgs_norm, Vds_norm]
        return self.net(x).squeeze(-1)
