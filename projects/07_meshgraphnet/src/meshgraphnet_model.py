"""
MeshGraphNet-style Encode-Process-Decode 아키텍처 (Pfaff et al., 2020,
"Learning Mesh-Based Simulation with Graph Networks"의 핵심 구조를 직접 구현).

설계 노트: NVIDIA PhysicsNeMo에도 공식 MeshGraphNet 모듈이 있지만, 내부적으로
DGL(Deep Graph Library)에 의존할 가능성이 높아 이 환경에서 추가 설치 리스크가
있다. 대신 핵심 알고리즘(node/edge encoder, message-passing processor,
decoder)을 순수 PyTorch로 직접 구현했다 — MLIP 프로젝트에서 Behler-Parrinello
신경망을 직접 구현한 것과 같은 접근.

구조:
1. Encoder: 원시 node feature, edge feature를 각각 MLP로 latent space에 투영
2. Processor: 여러 라운드의 message passing.
   - 각 edge: (source node, target node, edge) latent를 합쳐 MLP로 edge 업데이트
   - 각 node: 자신의 latent + 인접 edge latent 합산을 MLP로 node 업데이트
   - residual connection으로 안정적 학습
3. Decoder: 최종 node latent를 MLP로 물리량(여기서는 다음 스텝 변위)으로 디코딩
"""

import torch
import torch.nn as nn


def mlp(in_dim, hidden_dim, out_dim, n_hidden=2, layer_norm=True):
    layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for _ in range(n_hidden - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    layers += [nn.Linear(hidden_dim, out_dim)]
    if layer_norm:
        layers += [nn.LayerNorm(out_dim)]
    return nn.Sequential(*layers)


class MessagePassingBlock(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        self.edge_mlp = mlp(3 * latent_dim, hidden_dim, latent_dim)
        self.node_mlp = mlp(2 * latent_dim, hidden_dim, latent_dim)

    def forward(self, node_latent, edge_latent, senders, receivers):
        # edge update: [edge, sender_node, receiver_node] -> new edge latent
        edge_input = torch.cat([
            edge_latent, node_latent[senders], node_latent[receivers]
        ], dim=-1)
        new_edge_latent = edge_latent + self.edge_mlp(edge_input)

        # node update: 인접 edge latent를 노드별로 합산 (scatter-add) 후 결합
        agg = torch.zeros_like(node_latent)
        agg.index_add_(0, receivers, new_edge_latent)
        node_input = torch.cat([node_latent, agg], dim=-1)
        new_node_latent = node_latent + self.node_mlp(node_input)

        return new_node_latent, new_edge_latent


class MeshGraphNet(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, out_dim,
                 latent_dim=32, hidden_dim=48, n_message_passing=6):
        super().__init__()
        self.node_encoder = mlp(node_in_dim, hidden_dim, latent_dim)
        self.edge_encoder = mlp(edge_in_dim, hidden_dim, latent_dim)
        self.blocks = nn.ModuleList([
            MessagePassingBlock(latent_dim, hidden_dim) for _ in range(n_message_passing)
        ])
        self.decoder = mlp(latent_dim, hidden_dim, out_dim, layer_norm=False)

    def forward(self, node_features, edge_features, senders, receivers):
        node_latent = self.node_encoder(node_features)
        edge_latent = self.edge_encoder(edge_features)

        for block in self.blocks:
            node_latent, edge_latent = block(node_latent, edge_latent, senders, receivers)

        return self.decoder(node_latent)


if __name__ == "__main__":
    torch.manual_seed(0)
    N, E = 10, 18  # 10 노드, 방향 포함 18개 edge (양방향 체인)
    node_features = torch.randn(N, 4)
    edge_features = torch.randn(E, 3)
    senders = torch.randint(0, N, (E,))
    receivers = torch.randint(0, N, (E,))

    model = MeshGraphNet(node_in_dim=4, edge_in_dim=3, out_dim=2)
    out = model(node_features, edge_features, senders, receivers)
    print("Output shape:", out.shape, "(should be (N, 2))")
