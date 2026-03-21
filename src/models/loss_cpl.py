from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CplLoss(nn.Module):
    """
    复合损失：
        total = mse_weight * weighted_mse
              + rank_weight * pairwise_ranking
              + cpl_weight * legacy_cpl_penalty(optional)

    说明：
    1) weighted_mse:
       对关键端点做 CP-aware 加权，体现 EDA 的 "fix-first" 思想。

    2) pairwise_ranking:
       对端点相对顺序建模，是 Kendall's Tau / top-k recall 的平滑 surrogate。

    3) group_ids:
       多设计训练时，只在同一 design 内构造 pairwise 比较，避免跨设计排序污染。

    4) sample_weight:
       如果外部已经按 design / split 算好了关键端点权重，可直接传入；
       其优先级高于内部 critical_* 逻辑。
    """

    def __init__(
        self,
        mse_weight: float = 1.0,
        rank_weight: float = 0.2,
        cpl_weight: float = 0.1,
        critical_fraction: float = 0.10,
        critical_weight: float = 2.0,
        critical_mode: str = "high",   # "high" for CPL / arrival; "low" for slack
        pairwise_margin: float = 0.0,
        pairwise_loss: str = "logistic",   # "logistic" or "hinge"
        pair_gap_power: float = 0.0,       # >0 时更强调 |y_i-y_j| 大的 pair
        tie_epsilon: float = 1.0e-8,
        max_pairs: int = 0,                # 0 => use all valid pairs
    ):
        super().__init__()
        self.mse_weight = float(mse_weight)
        self.rank_weight = float(rank_weight)
        self.cpl_weight = float(cpl_weight)

        self.critical_fraction = float(critical_fraction)
        self.critical_weight = float(critical_weight)
        self.critical_mode = str(critical_mode).lower()
        if self.critical_mode not in ("high", "low"):
            self.critical_mode = "high"

        self.pairwise_margin = float(pairwise_margin)
        self.pairwise_loss = str(pairwise_loss).lower()
        if self.pairwise_loss not in ("logistic", "hinge"):
            raise ValueError(f"Unsupported pairwise_loss={pairwise_loss}")

        self.pair_gap_power = float(pair_gap_power)
        self.tie_epsilon = float(tie_epsilon)
        self.max_pairs = int(max_pairs)

    @staticmethod
    def _flatten_1d(x: torch.Tensor) -> torch.Tensor:
        return x.reshape(-1)

    def _build_sample_weights(
        self,
        y_true: torch.Tensor,
        group_ids: Optional[torch.Tensor] = None,
        critical_threshold: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        若外部没有传 sample_weight，则内部按照 critical_fraction 自动构造。

        - 若给了 critical_threshold：按统一阈值做 critical mask
        - 否则：
            * 如果给了 group_ids，则按 group 内 top-k% 做 critical
            * 如果没给 group_ids，则按整个 batch 内 top-k% 做 critical
        """
        y_true = self._flatten_1d(y_true)
        weights = torch.ones_like(y_true)
        critical_mask = torch.zeros_like(y_true, dtype=torch.bool)

        if y_true.numel() == 0:
            return weights, critical_mask

        if self.critical_fraction <= 0.0 or self.critical_weight <= 1.0:
            return weights, critical_mask

        # 路径 1：外部给统一阈值
        if critical_threshold is not None:
            thr = torch.as_tensor(
                critical_threshold, dtype=y_true.dtype, device=y_true.device
            )
            if self.critical_mode == "low":
                critical_mask = y_true <= thr
            else:
                critical_mask = y_true >= thr

            weights = 1.0 + (self.critical_weight - 1.0) * critical_mask.to(y_true.dtype)
            return weights, critical_mask

        # 路径 2：batch 内 / group 内 top-k%
        if group_ids is None:
            group_ids = torch.zeros_like(y_true, dtype=torch.long)
        else:
            group_ids = torch.as_tensor(group_ids, device=y_true.device).reshape(-1).long()
            if group_ids.numel() != y_true.numel():
                raise ValueError("group_ids length must match y_true length")

        unique_groups = torch.unique(group_ids)
        for gid in unique_groups:
            gmask = group_ids == gid
            y_g = y_true[gmask]
            if y_g.numel() == 0:
                continue

            k = max(1, int(round(y_g.numel() * self.critical_fraction)))
            k = min(k, int(y_g.numel()))

            if self.critical_mode == "low":
                thr_g = torch.topk(y_g, k=k, largest=False).values.max()
                critical_g = y_g <= thr_g
            else:
                thr_g = torch.topk(y_g, k=k, largest=True).values.min()
                critical_g = y_g >= thr_g

            critical_mask[gmask] = critical_g

        weights = 1.0 + (self.critical_weight - 1.0) * critical_mask.to(y_true.dtype)
        return weights, critical_mask

    def _weighted_mse(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        y_pred = self._flatten_1d(y_pred)
        y_true = self._flatten_1d(y_true)

        if sample_weight is None:
            sample_weight = torch.ones_like(y_true)
        else:
            sample_weight = torch.as_tensor(
                sample_weight, dtype=y_true.dtype, device=y_true.device
            ).reshape(-1)
            if sample_weight.numel() != y_true.numel():
                raise ValueError("sample_weight length must match y_true length")

        err2 = (y_pred - y_true) ** 2
        denom = torch.clamp(sample_weight.sum(), min=1.0e-12)
        return torch.sum(sample_weight * err2) / denom

    def _pairwise_ranking_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None,
        group_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, int]:
        """
        Pairwise ranking:
            若 y_i > y_j，则希望 pred_i - pred_j > margin
        """
        y_pred = self._flatten_1d(y_pred)
        y_true = self._flatten_1d(y_true)

        n = int(y_true.numel())
        zero = y_pred.new_zeros(())

        if self.rank_weight <= 0.0 or n < 2:
            return zero, 0

        if sample_weight is None:
            sample_weight = torch.ones_like(y_true)
        else:
            sample_weight = torch.as_tensor(
                sample_weight, dtype=y_true.dtype, device=y_true.device
            ).reshape(-1)
            if sample_weight.numel() != y_true.numel():
                raise ValueError("sample_weight length must match y_true length")

        idx_i, idx_j = torch.triu_indices(n, n, offset=1, device=y_true.device)
        if idx_i.numel() == 0:
            return zero, 0

        if group_ids is not None:
            group_ids = torch.as_tensor(group_ids, device=y_true.device).reshape(-1).long()
            if group_ids.numel() != y_true.numel():
                raise ValueError("group_ids length must match y_true length")
            same_group = group_ids[idx_i] == group_ids[idx_j]
            idx_i = idx_i[same_group]
            idx_j = idx_j[same_group]
            if idx_i.numel() == 0:
                return zero, 0

        diff_true = y_true[idx_i] - y_true[idx_j]
        valid = diff_true.abs() > self.tie_epsilon
        idx_i = idx_i[valid]
        idx_j = idx_j[valid]
        diff_true = diff_true[valid]

        if idx_i.numel() == 0:
            return zero, 0

        if self.max_pairs > 0 and idx_i.numel() > self.max_pairs:
            perm = torch.randperm(idx_i.numel(), device=y_true.device)[: self.max_pairs]
            idx_i = idx_i[perm]
            idx_j = idx_j[perm]
            diff_true = diff_true[perm]

        sign_true = torch.sign(diff_true)
        diff_pred = y_pred[idx_i] - y_pred[idx_j]
        score = sign_true * diff_pred  # 正确排序 => score 大

        if self.pairwise_loss == "hinge":
            pair_loss = F.relu(self.pairwise_margin - score)
        else:
            # logistic
            pair_loss = F.softplus(self.pairwise_margin - score)

        pair_weight = torch.maximum(sample_weight[idx_i], sample_weight[idx_j])

        if self.pair_gap_power > 0.0:
            gap = diff_true.abs()
            gap = gap / torch.clamp(gap.mean(), min=1.0e-12)
            pair_weight = pair_weight * torch.pow(gap, self.pair_gap_power)

        denom = torch.clamp(pair_weight.sum(), min=1.0e-12)
        rank_loss = torch.sum(pair_weight * pair_loss) / denom
        return rank_loss, int(idx_i.numel())

    def _legacy_cpl_penalty(
        self,
        gnn_emb: Optional[torch.Tensor],
        cpl_indices: Optional[List[List[int]]],
    ) -> torch.Tensor:
        """
        保留你原先的接口：
        若 cpl_indices 非空，则鼓励 graph embedding norm 更大。
        """
        if self.cpl_weight <= 0.0 or gnn_emb is None or cpl_indices is None:
            if gnn_emb is not None:
                return gnn_emb.new_zeros(())
            return torch.tensor(0.0)

        if gnn_emb.ndim == 1:
            gnn_emb = gnn_emb.unsqueeze(0)

        norms = torch.linalg.norm(gnn_emb, dim=-1)
        batch_n = int(norms.numel())

        cpl_list = list(cpl_indices)
        if len(cpl_list) < batch_n:
            cpl_list = cpl_list + [[] for _ in range(batch_n - len(cpl_list))]
        elif len(cpl_list) > batch_n:
            cpl_list = cpl_list[:batch_n]

        mask_vals = [1.0 if len(idx) > 0 else 0.0 for idx in cpl_list]
        mask = torch.tensor(mask_vals, dtype=norms.dtype, device=norms.device)
        return -torch.mean(norms * mask)

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        gnn_emb: Optional[torch.Tensor] = None,
        cpl_indices: Optional[List[List[int]]] = None,
        sample_weight: Optional[torch.Tensor] = None,
        group_ids: Optional[torch.Tensor] = None,
        critical_threshold: Optional[float] = None,
        return_details: bool = False,
    ):
        y_pred = self._flatten_1d(y_pred)
        y_true = self._flatten_1d(y_true)

        if y_pred.numel() != y_true.numel():
            raise ValueError("y_pred and y_true must have the same number of elements")

        if sample_weight is None:
            sample_weight, critical_mask = self._build_sample_weights(
                y_true=y_true,
                group_ids=group_ids,
                critical_threshold=critical_threshold,
            )
        else:
            sample_weight = torch.as_tensor(
                sample_weight, dtype=y_true.dtype, device=y_true.device
            ).reshape(-1)
            if sample_weight.numel() != y_true.numel():
                raise ValueError("sample_weight length must match y_true length")
            critical_mask = sample_weight > 1.0

        mse_loss = self._weighted_mse(y_pred, y_true, sample_weight=sample_weight)
        rank_loss, num_pairs = self._pairwise_ranking_loss(
            y_pred=y_pred,
            y_true=y_true,
            sample_weight=sample_weight,
            group_ids=group_ids,
        )
        cpl_penalty = self._legacy_cpl_penalty(gnn_emb=gnn_emb, cpl_indices=cpl_indices)

        mse_term = self.mse_weight * mse_loss
        rank_term = self.rank_weight * rank_loss
        cpl_term = self.cpl_weight * cpl_penalty

        total_loss = mse_term + rank_term + cpl_term

        if not return_details:
            return total_loss

        details: Dict[str, object] = {
            "total": total_loss,
            "mse": mse_loss,
            "rank": rank_loss,
            "cpl": cpl_penalty,
            "mse_term": mse_term,
            "rank_term": rank_term,
            "cpl_term": cpl_term,
            "num_pairs": num_pairs,
            "num_critical": int(critical_mask.sum().item()),
        }
        return total_loss, details