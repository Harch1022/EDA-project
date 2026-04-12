from __future__ import annotations

import torch
import torch.nn as nn
import dgl
from dgl.nn import GraphConv, SAGEConv


class VanillaGNN(nn.Module):
    """
    Standard GCN / GraphSAGE encoder with the same interface as BPN.
    This lets train_cpl.py reuse the existing training loop unchanged.
    """

    def __init__(
        self,
        d_in: int,
        hidden: int = 64,
        layers: int = 3,
        conv_type: str = "gcn",
        use_edge_weight: bool = True,
    ):
        super().__init__()
        self.conv_type = conv_type.lower()
        self.use_edge_weight = bool(use_edge_weight)

        self.layers = nn.ModuleList()
        last = d_in
        for _ in range(layers):
            if self.conv_type == "gcn":
                layer = GraphConv(
                    in_feats=last,
                    out_feats=hidden,
                    norm="both",
                    allow_zero_in_degree=True,
                )
            elif self.conv_type == "sage":
                layer = SAGEConv(
                    in_feats=last,
                    out_feats=hidden,
                    aggregator_type="mean",
                )
            else:
                raise ValueError(
                    f"Unknown conv_type={conv_type}, expected 'gcn' or 'sage'."
                )
            self.layers.append(layer)
            last = hidden

        self.readout = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

    def _get_edge_weight(self, g: dgl.DGLGraph, x: torch.Tensor):
        if (not self.use_edge_weight) or ("weight" not in g.edata):
            return None
        ew = g.edata["weight"].to(device=x.device, dtype=x.dtype)
        if ew.dim() == 2 and ew.shape[-1] == 1:
            ew = ew.squeeze(-1)
        return ew

    def node_embeddings(self, g: dgl.DGLGraph, x: torch.Tensor) -> torch.Tensor:
        if g.device != x.device:
            g = g.to(x.device)

        # keep behavior aligned with your current BPN
        g_bid = dgl.add_reverse_edges(g, copy_ndata=False, copy_edata=True)
        edge_w = self._get_edge_weight(g_bid, x)

        h = x
        for conv in self.layers:
            if self.conv_type == "gcn":
                h = conv(g_bid, h, edge_weight=edge_w)
            else:  # sage
                h = conv(g_bid, h)
            h = torch.relu(h)
        return h

    def forward(
        self,
        g: dgl.DGLGraph,
        x: torch.Tensor,
        return_node_emb: bool = False,
    ):
        if g.device != x.device:
            g = g.to(x.device)

        h = self.node_embeddings(g, x)

        with g.local_scope():
            g.ndata["_h"] = h
            hg = dgl.mean_nodes(g, "_h")

        if hg.dim() == 2 and hg.shape[0] == 1:
            hg = hg.squeeze(0)

        g_emb = self.readout(hg)

        if return_node_emb:
            return g_emb, h
        return g_emb