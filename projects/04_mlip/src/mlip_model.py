"""
Behler-Parrinello 스타일 신경망 포텐셜(NNP).

핵심 아이디어: 모든 원자에 "동일한" 작은 신경망(shared weights)을 적용해
각 원자의 descriptor -> 원자당 에너지를 예측하고, 전체 에너지는 단순 합.
Force는 전체 에너지를 원자 좌표에 대해 미분(autograd)해서 얻는다
(-dE/dx), 이는 물리적으로 에너지 보존을 자동으로 만족시키는 방식이다.

같은 신경망을 모든 원자에 재사용하는 것이 순서(permutation) 불변성과
임의 개수 원자로의 일반화를 동시에 보장하는 핵심 설계.
"""

import torch
import torch.nn as nn
from descriptors import compute_descriptors, R_CENTERS


class AtomicNN(nn.Module):
    """단일 원자의 descriptor -> 에너지. 모든 원자가 이 네트워크를 공유."""

    def __init__(self, n_desc=len(R_CENTERS), hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_desc, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, desc):
        return self.net(desc).squeeze(-1)  # (N,)


class MLIP(nn.Module):
    def __init__(self, n_desc=len(R_CENTERS), hidden=32):
        super().__init__()
        self.atomic_nn = AtomicNN(n_desc, hidden)

    def forward(self, pos):
        """
        pos: (N, 3), requires_grad=True
        반환: total_energy (scalar), forces (N, 3)
        """
        desc = compute_descriptors(pos)
        atomic_E = self.atomic_nn(desc)         # (N,)
        total_E = atomic_E.sum()

        forces = -torch.autograd.grad(
            total_E, pos, create_graph=self.training, retain_graph=True
        )[0]
        return total_E, forces


if __name__ == "__main__":
    torch.manual_seed(0)
    model = MLIP()
    pos = (torch.randn(10, 3) * 1.5).requires_grad_(True)
    E, F = model(pos)
    print("Total energy:", E.item())
    print("Forces shape:", F.shape)
