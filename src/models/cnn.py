from __future__ import annotations
import torch
import torch.nn as nn

class PhysCNN(nn.Module):
    """CNN over 3-channel physical maps [B, 3, H, W] -> embedding"""
    def __init__(self, channels=(16, 32), out_dim=64):
        super().__init__()
        c1, c2 = channels
        self.net = nn.Sequential(
            nn.Conv2d(3, c1, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.proj = nn.Linear(c2, out_dim)

    def forward(self, maps: torch.Tensor) -> torch.Tensor:
        h = self.net(maps)
        h = h.flatten(1)
        return self.proj(h)