from __future__ import annotations
import torch
import torch.nn as nn
import dgl
from dgl.nn import GraphConv

class BPN(nn.Module):
    """
    Simple bidirectional propagation using reverse edges.
    Input: node_feats [N, d_in]
    Output: graph embedding [d_hid]
    """
    def __init__(self, d_in: int, hidden: int = 64, layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList()
        last = d_in
        for _ in range(layers):
            self.layers.append(GraphConv(last, hidden, norm='both'))
            last = hidden
        self.readout = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )

    def forward(self, g: dgl.DGLGraph, x: torch.Tensor) -> torch.Tensor:
        g_bid = dgl.add_reverse_edges(g, copy_edata=True)
        h = x
        for conv in self.layers:
            h = conv(g_bid, h)
            h = torch.relu(h)
        # graph-level mean pool
        hg = h.mean(dim=0)  # [hidden]
        return self.readout(hg)