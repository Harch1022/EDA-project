from __future__ import annotations
import torch
import dgl
import numpy as np

from src.models.bpn import BPN
from src.models.node_importance import NodeImportanceHead


def build_chain_graph(n: int) -> dgl.DGLGraph:
    # chain 0->1->...->(n-1)
    src = torch.arange(0, n - 1, dtype=torch.int64)
    dst = torch.arange(1, n, dtype=torch.int64)
    g = dgl.graph((src, dst), num_nodes=n)
    return g


def kl_loss(p_model: torch.Tensor, p_teacher: torch.Tensor) -> torch.Tensor:
    eps = 1e-12
    p_m = torch.clamp(p_model, min=eps)
    p_t = torch.clamp(p_teacher, min=eps)
    return torch.sum(p_t * (torch.log(p_t) - torch.log(p_m)))


def test_bpn_aux_loss_backward():
    torch.manual_seed(0)
    n = 6
    d_in = 8
    hidden = 16
    layers = 2

    # 图与特征
    g = build_chain_graph(n)
    x = torch.randn(n, d_in)

    # 模型与 head
    bpn = BPN(d_in=d_in, hidden=hidden, layers=layers)
    head = NodeImportanceHead(d_in=hidden, temperature=1.0, normalize="softmax")

    # 前向获取节点 embedding 与分布
    g_emb, node_emb = bpn(g, x, return_node_emb=True)
    assert g_emb.shape == (hidden,)
    assert node_emb.shape == (n, hidden)

    p_model = head(node_emb)
    assert p_model.shape == (n,)
    assert torch.isfinite(p_model).all()
    assert abs(p_model.sum().item() - 1.0) < 1e-5

    # 构造老师分布（随机关联且归一化）
    t = torch.rand(n)
    p_teacher = t / t.sum()

    # 计算 KL 并反向传播
    loss = kl_loss(p_model, p_teacher)
    loss.backward()

    # 检查梯度
    # BPN 第一层权重与 head 线性层应有梯度
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in bpn.parameters())
    assert head.lin.weight.grad is not None
    assert torch.isfinite(head.lin.weight.grad).all()