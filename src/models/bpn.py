from __future__ import annotations
import torch
import torch.nn as nn
import dgl
from dgl.nn import GATConv


class BPN(nn.Module):
    """
    Bidirectional propagation GNN with GATConv and edge weights.

    Input:
      - g.edata['weight']: [E], normalized Manhattan distance
      - node_feats x: [N, d_in]

    Output:
      - default:
          * single graph: [hidden]
          * batched graph: [B, hidden]
      - if return_node_emb=True:
          (graph embedding, node embeddings [N, hidden])
    """

    def __init__(
        self,
        d_in: int,
        hidden: int = 64,
        layers: int = 3,
        heads: int = 1,
    ):
        super().__init__()

        if hidden % heads != 0:
            raise ValueError(
                f"hidden ({hidden}) must be divisible by heads ({heads})."
            )

        self.hidden = hidden
        self.heads = heads
        self.layers = nn.ModuleList()

        last = d_in
        out_per_head = hidden // heads

        for _ in range(layers):
            self.layers.append(
                GATConv(
                    in_feats=last,
                    out_feats=out_per_head,
                    num_heads=heads,
                    feat_drop=0.0,
                    attn_drop=0.0,
                    residual=(last == hidden),
                    activation=None,
                    allow_zero_in_degree=True,
                )
            )
            last = hidden

        self.readout = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )

    def _get_edge_weight(
        self,
        g: dgl.DGLGraph,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        读取图上的边权。若不存在，则退化为全 1。
        """
        if "weight" in g.edata:
            ew = g.edata["weight"]
        else:
            ew = torch.ones(g.num_edges(), dtype=x.dtype, device=x.device)

        ew = ew.to(device=x.device, dtype=x.dtype)

        # 兼容 [E, 1] / [E] 两种形式
        if ew.dim() == 2 and ew.shape[-1] == 1:
            ew = ew.squeeze(-1)

        return ew

    def node_embeddings(self, g: dgl.DGLGraph, x: torch.Tensor) -> torch.Tensor:
        """
        返回节点级 embedding [N, hidden]。
        内部会在图上加入反向边，并复制 edge weight。
        """
        if g.device != x.device:
            g = g.to(x.device)

        # 关键点：copy_edata=True，会把 weight 一起复制到反向边
        g_bid = dgl.add_reverse_edges(
            g,
            copy_ndata=False,
            copy_edata=True,
        )

        edge_w = self._get_edge_weight(g_bid, x)

        h = x
        for conv in self.layers:
            # DGL 新版 GATConv 支持 edge_weight 参数
            h = conv(g_bid, h, edge_weight=edge_w)   # [N, heads, out_per_head]
            h = h.flatten(1)                         # [N, hidden]
            h = torch.relu(h)

        return h

    def forward(
        self,
        g: dgl.DGLGraph,
        x: torch.Tensor,
        return_node_emb: bool = False,
    ):
        """
        默认返回图级 embedding；
        当 return_node_emb=True 时，额外返回节点 embedding。
        """
        if g.device != x.device:
            g = g.to(x.device)

        h = self.node_embeddings(g, x)  # [N, hidden]

        # 比 h.mean(dim=0) 更稳：既支持单图，也支持 batched graph
        with g.local_scope():
            g.ndata["_h"] = h
            hg = dgl.mean_nodes(g, "_h")   # single graph: [1, hidden], batch: [B, hidden]

        if hg.dim() == 2 and hg.shape[0] == 1:
            hg = hg.squeeze(0)            # 保持和你原先单图接口一致 -> [hidden]

        g_emb = self.readout(hg)

        if return_node_emb:
            return g_emb, h
        return g_emb