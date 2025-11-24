from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class NodeImportanceHead(nn.Module):
    """
    将节点级 embedding [N, D] 投影为节点重要性分布 p_model [N]。
    实现：线性层生成 logits -> softmax(temperature) 归一为概率分布。
    """
    def __init__(self, d_in: int, temperature: float = 1.0, normalize: str = "softmax"):
        super().__init__()
        self.lin = nn.Linear(d_in, 1)
        self.temperature = float(temperature)
        self.normalize = (normalize or "softmax").lower()

    def forward(self, node_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
          node_emb: [N, D]
        Returns:
          probs: [N], sum=1 (when normalize=softmax)
        """
        logits = self.lin(node_emb).squeeze(-1)  # [N]
        if self.normalize == "softmax":
            temp = max(self.temperature, 1e-6)
            probs = F.softmax(logits / temp, dim=0)  # over nodes
            return probs
        elif self.normalize == "l1":
            x = torch.relu(logits)
            denom = torch.clamp(x.sum(), min=1e-12)
            return x / denom
        else:
            # fallback: softmax
            probs = torch.softmax(logits, dim=0)
            return probs