from __future__ import annotations
import torch
import torch.nn as nn


class EndpointEmbedding(nn.Module):
    """
    每个 endpoint 一个 embedding 向量。
    num_endpoints: 数据集中 endpoint 的总数（这里就是 len(EndpointDataset)）
    d_ep: 端点 embedding 的维度，对应 config 里的 model.endpoint_dim
    """
    def __init__(self, num_endpoints: int, d_ep: int = 16):
        super().__init__()
        self.emb = nn.Embedding(num_endpoints, d_ep)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        idx: [B] 的 LongTensor，每个元素是一个 endpoint 的索引（0 ~ num_endpoints-1）
        返回: [B, d_ep]
        """
        return self.emb(idx)


class FusionRegressorCond(nn.Module):
    """
    把 gnn_emb, cnn_emb 和 ep_emb 融合在一起，输出一个标量预测。
    """
    def __init__(self, d_gnn: int, d_cnn: int, d_ep: int, hidden: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_gnn + d_cnn + d_ep, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        gnn_emb: torch.Tensor,  # [B, d_gnn]
        cnn_emb: torch.Tensor,  # [B, d_cnn]
        ep_emb: torch.Tensor,   # [B, d_ep]
    ) -> torch.Tensor:
        z = torch.cat([gnn_emb, cnn_emb, ep_emb], dim=-1)  # [B, d_gnn + d_cnn + d_ep]
        return self.mlp(z).squeeze(-1)  # [B]