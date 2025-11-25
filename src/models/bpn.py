from __future__ import annotations
import torch
import torch.nn as nn
import dgl
from dgl.nn import GraphConv


class BPN(nn.Module):
    """
    Simple bidirectional propagation GNN using reverse edges inclusion.
    Input: node_feats [N, d_in]
    Output:
      - default: graph embedding [hidden]
      - if return_node_emb=True: (graph embedding [hidden], node embeddings [N, hidden])
    """
    def __init__(self, d_in: int, hidden: int = 64, layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList()
        last = d_in
        for _ in range(layers):
            # 关键改动：allow_zero_in_degree=True，避免 0 入度节点报错
            self.layers.append(
                GraphConv(
                    last,
                    hidden,
                    norm='both',
                    allow_zero_in_degree=True
                )
            )
            last = hidden
        self.readout = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )

    def node_embeddings(self, g: dgl.DGLGraph, x: torch.Tensor) -> torch.Tensor:
        """
        返回节点级 embedding [N, hidden]，内部会在图上加入反向边。
        """
        # 为增强信息流，使用双向图（add_reverse_edges）
        g_bid = dgl.add_reverse_edges(g, copy_edata=True)
        h = x
        for conv in self.layers:
            h = conv(g_bid, h)
            h = torch.relu(h)
        return h

    def forward(self, g: dgl.DGLGraph, x: torch.Tensor, return_node_emb: bool = False):
        """
        默认返回图级 embedding；当 return_node_emb=True 时，额外返回节点 embedding。
        """
        h = self.node_embeddings(g, x)         # [N, hidden]
        hg = h.mean(dim=0)                     # [hidden]
        g_emb = self.readout(hg)               # [hidden]
        if return_node_emb:
            return g_emb, h
        return g_emb