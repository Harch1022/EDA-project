from __future__ import annotations
import torch
import torch.nn as nn
from typing import List

class CplLoss(nn.Module):
    """
    L = mse_weight * MSE(y_hat, y) + cpl_weight * penalty
    当前基线没有 per-node 重要性输出，用 gnn_emb 范数作为近关键存在的代理增益项（可后续替换）。
    """
    def __init__(self, mse_weight: float = 1.0, cpl_weight: float = 0.1):
        super().__init__()
        self.mse = nn.MSELoss()
        self.mse_weight = mse_weight
        self.cpl_weight = cpl_weight

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor,
                gnn_emb: torch.Tensor,
                cpl_indices: List[List[int]] | None = None) -> torch.Tensor:
        loss = self.mse_weight * self.mse(y_pred, y_true)
        if self.cpl_weight > 0 and cpl_indices is not None:
            norms = torch.linalg.norm(gnn_emb, dim=-1)
            mask = torch.tensor([1.0 if len(idx)>0 else 0.0 for idx in cpl_indices],
                                dtype=norms.dtype, device=norms.device)
            penalty = -torch.mean(norms * mask)  # encourage larger norm when CPL exists
            loss = loss + self.cpl_weight * penalty
        return loss