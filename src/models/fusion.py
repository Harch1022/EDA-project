from __future__ import annotations
import torch
import torch.nn as nn

class FusionRegressor(nn.Module):
    """Fuse GNN and CNN embeddings -> scalar arrival time"""
    def __init__(self, d_gnn: int = 64, d_cnn: int = 64, hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_gnn + d_cnn, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, gnn_emb: torch.Tensor, cnn_emb: torch.Tensor) -> torch.Tensor:
        z = torch.cat([gnn_emb, cnn_emb], dim=-1)
        return self.mlp(z).squeeze(-1)