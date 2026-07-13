"""
Behler-Parrinello 스타일 radial symmetry function descriptor (간략화 버전).

각 원자 i에 대해, 주변 이웃과의 거리 분포를 여러 개의 Gaussian 기준점(r_s)에서
평가한 값들의 합으로 표현한다:

    G_i(r_s) = sum_{j != i, r_ij < cutoff} exp(-eta * (r_ij - r_s)^2) * f_cutoff(r_ij)

이 표현은 순서(permutation)와 회전(rotation)에 불변하며, 원자 개수와 무관하게
고정된 길이의 벡터를 만들어낸다 — 신경망 입력으로 쓰기 위한 핵심 조건.

실제 Behler & Parrinello(2007) 논문은 각도 항(G4/G5, 3체 상호작용)까지 포함하지만,
여기서는 학습·구현 목적상 radial(2체) 항만 사용하는 단순화 버전임을 명시.
"""

import torch

CUTOFF = 3.0
R_CENTERS = torch.tensor([0.9, 1.1, 1.3, 1.6, 2.0, 2.5], dtype=torch.float32)
ETA = 4.0


def cutoff_fn(r, rc=CUTOFF):
    """부드럽게 0으로 수렴하는 cosine cutoff function (BP 논문 표준)."""
    return torch.where(
        r < rc,
        0.5 * (torch.cos(torch.pi * r / rc) + 1.0),
        torch.zeros_like(r),
    )


def compute_descriptors(pos, r_centers=R_CENTERS, eta=ETA, cutoff=CUTOFF):
    """
    pos: (N, 3) torch tensor, requires_grad=True 가능 (force 계산을 위해)
    반환: (N, len(r_centers)) descriptor 행렬

    주의: 자기 자신과의 거리(diagonal)를 inf로 마스킹하면 torch.where의
    backward pass에서 cos(inf) 등이 계산되며 NaN gradient가 발생할 수 있다
    (선택되지 않은 branch도 backward 시 미분됨 — PyTorch의 잘 알려진 함정).
    대신 항상 유한한 값을 유지한 채, 명시적 (1-eye) 마스크로 self-term을
    안전하게 0으로 만든다.
    """
    N = pos.shape[0]
    diff = pos.unsqueeze(1) - pos.unsqueeze(0)            # (N, N, 3), diagonal = 0
    r2 = (diff ** 2).sum(-1)                                # (N, N), diagonal = 0
    r = torch.sqrt(r2 + 1e-12)                               # eps로 sqrt(0) gradient 특이점 방지

    eye = torch.eye(N, dtype=torch.bool, device=pos.device)
    off_diag = (~eye).float()                                # self-term을 명시적으로 0 처리

    fc = cutoff_fn(r, cutoff)                                # r이 항상 유한하므로 안전

    rs = r_centers.to(pos.device).view(1, 1, -1)
    diff_r = r.unsqueeze(-1) - rs
    gauss = torch.exp(-eta * diff_r ** 2)
    contrib = gauss * fc.unsqueeze(-1) * off_diag.unsqueeze(-1)

    descriptors = contrib.sum(dim=1)                         # (N, n_centers), 이웃에 대해 합산
    return descriptors


if __name__ == "__main__":
    torch.manual_seed(0)
    pos = (torch.randn(10, 3) * 1.5).requires_grad_(True)
    desc = compute_descriptors(pos)
    print("Descriptor shape:", desc.shape)
    print("Sample descriptor (atom 0):", desc[0].detach().numpy())

    # 미분 가능성 확인 (force 계산에 필수)
    loss = desc.sum()
    loss.backward()
    print("Gradient w.r.t. positions works:", pos.grad is not None, pos.grad.shape)
