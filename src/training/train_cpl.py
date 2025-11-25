from __future__ import annotations
import argparse
import json
import logging
import os
import random
from typing import Dict, Tuple, List, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import yaml
from torch.utils.data import Dataset

from src.models.bpn import BPN
from src.models.cnn import PhysCNN
from src.models.fusion import FusionRegressor
from src.models.loss_cpl import CplLoss
from src.training.metrics import mape
from src.models.node_importance import NodeImportanceHead

from math import sqrt
from src.training.metrics import r2 as r2_score

from src.models.endpoint_cond import EndpointEmbedding, FusionRegressorCond

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_cpl")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EndpointDataset(Dataset):
    def __init__(self, npz_path: str):
        super().__init__()
        self.data = np.load(npz_path, allow_pickle=True)

        # 图结构与物理特征
        self.node_features = self.data["node_features"]  # [N, d]
        self.edges = self.data["edges"]                  # [2, E]
        self.maps = np.stack(
            [
                self.data["cell_density_map"],
                self.data["rudy_map"],
                self.data["macro_mask_map"],
            ],
            axis=0,
        )  # [3, H, W]

        # endpoint 标签
        self.endpoints = self.data["endpoints"]          # [E] (object array of names)
        self.y_arrival = self.data["y_arrival"]          # [E]
        self.cpl_indices = self.data["cpl_indices"]      # list[list[int]]

        # 构建 DGL 图与张量
        import dgl
        import torch as T

        src = T.from_numpy(self.edges[0].astype(np.int64))
        dst = T.from_numpy(self.edges[1].astype(np.int64))
        self.g = dgl.graph((src, dst), num_nodes=self.node_features.shape[0])

        self.x = T.from_numpy(self.node_features.astype(np.float32))
        # 保持 batch 维度为 1： [1, 3, H, W]
        self.maps_t = T.from_numpy(self.maps.astype(np.float32)).unsqueeze(0)

    def __len__(self) -> int:
        return len(self.y_arrival)

    def __getitem__(self, idx: int):
        """
        返回:
          ep_id:  端点在数据集中的索引（0..len(self)-1）
          y:      延迟标签
          ci:     CPL 索引列表
          ep_name:端点名字（字符串），用于 BPN 老师分布的查表
        """
        y = float(self.y_arrival[idx])
        ci = list(self.cpl_indices[idx])
        ep_name = str(self.endpoints[idx])
        ep_id = idx  # 这里直接用 idx 作为 endpoint 的 id
        return ep_id, y, ci, ep_name


def _infer_bpn_importance_path(dataset_npz: str) -> str:
    base = os.path.splitext(os.path.basename(dataset_npz))[0]
    out_dir = os.path.dirname(dataset_npz)
    return os.path.join(out_dir, f"{base}_bpn_importance.npz")


def _load_bpn_importance(importance_npz: str, num_nodes: int) -> Optional[Dict[str, Any]]:
    """
    返回:
      {
        'endpoints': List[str],
        'importance': np.ndarray [E, N],
        'map_by_name': Dict[str, np.ndarray [N]]
      }
    若文件不存在或形状不匹配，返回 None。
    """
    if not os.path.exists(importance_npz):
        logger.info("BPN importance file not found: %s (will skip BPN loss / teacher_pool)", importance_npz)
        return None
    arr = np.load(importance_npz, allow_pickle=True)
    endpoints = [str(e) for e in arr["endpoints"]]
    importance = arr["importance"].astype(np.float32)
    if importance.ndim != 2:
        logger.warning("BPN importance has invalid ndim=%d, expect 2. Skip.", importance.ndim)
        return None
    if importance.shape[1] != num_nodes:
        logger.warning(
            "BPN importance num_nodes mismatch: file=%d, graph=%d. Skip.",
            importance.shape[1],
            num_nodes,
        )
        return None
    # 归一化为概率分布
    imp = importance.copy()
    sums = np.clip(imp.sum(axis=1, keepdims=True), 1e-12, None)
    imp = imp / sums
    mp = {endpoints[i]: imp[i] for i in range(len(endpoints))}
    return {"endpoints": endpoints, "importance": imp, "map_by_name": mp}


def _compute_bpn_loss(
    p_model: torch.Tensor,           # [N], sum=1
    p_teacher_np: np.ndarray,        # [N], sum=1
    loss_type: str = "kl",
) -> torch.Tensor:
    """
    计算节点重要性分布的辅助损失。返回标量张量。
    """
    device = p_model.device
    p_teacher = torch.from_numpy(p_teacher_np).to(device=device, dtype=p_model.dtype)  # [N]
    eps = 1e-12

    if loss_type == "mse":
        return F.mse_loss(p_model, p_teacher)
    if loss_type == "l1":
        return torch.mean(torch.abs(p_model - p_teacher))

    # default: KL(p_teacher || p_model)
    p_t = torch.clamp(p_teacher, min=eps)
    p_m = torch.clamp(p_model, min=eps)
    kl = torch.sum(p_t * (torch.log(p_t) - torch.log(p_m)))
    return kl


def train_loop(cfg: Dict[str, object]):
    device = torch.device(cfg.get("device", "cpu"))
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    # ------------------------------------------------------------------
    # 加载数据集
    # ------------------------------------------------------------------
    data_cfg = cfg.get("data", {}) or {}
    npz_path = data_cfg.get("dataset_npz")
    if not npz_path or not os.path.exists(npz_path):
        raise FileNotFoundError(f"Dataset not found: {npz_path}")

    ds = EndpointDataset(npz_path=npz_path)
    n = len(ds)
    if n == 0:
        raise RuntimeError("Loaded dataset is empty (no endpoints with labels).")

    N_nodes, d_node = ds.node_features.shape
    logger.info("Loaded dataset from %s with %d endpoints.", npz_path, n)

    # ------------------------------------------------------------------
    # 端点条件化相关配置（支持 mode: none/id/teacher_pool/hybrid）
    # ------------------------------------------------------------------
    model_cfg = cfg.get("model", {}) or {}
    # 旧版开关（向后兼容）
    legacy_use_ep = bool(model_cfg.get("use_endpoint_condition", False))
    legacy_d_ep = int(model_cfg.get("endpoint_dim", 16))

    ep_cond_cfg = model_cfg.get("endpoint_conditioning", {}) or {}
    ep_mode = str(ep_cond_cfg.get("mode", "id" if legacy_use_ep else "none")).lower()
    d_ep_id = int(ep_cond_cfg.get("d_ep", legacy_d_ep))
    drop_ep = float(ep_cond_cfg.get("dropout", 0.0))

    use_ep_condition = ep_mode != "none"
    if use_ep_condition:
        logger.info("Endpoint conditioning mode: %s", ep_mode)
    else:
        logger.info("Endpoint conditioning disabled (mode=none).")

    # ------------------------------------------------------------------
    # 读取 BPN loss 配置与老师分布（同时供 teacher_pool 使用）
    # ------------------------------------------------------------------
    bpn_cfg = cfg.get("bpn", {}) or {}
    use_bpn_in_loss = bool(bpn_cfg.get("use_bpn_in_loss", False))
    bpn_loss_type = str(bpn_cfg.get("bpn_loss_type", "kl")).lower()
    base_bpn_loss_weight = float(bpn_cfg.get("bpn_loss_weight", 0.1))
    temperature = float(bpn_cfg.get("temperature", 1.0))
    importance_npz = bpn_cfg.get("importance_npz", None)
    if importance_npz is None:
        importance_npz = _infer_bpn_importance_path(npz_path)

    ablation = str(bpn_cfg.get("ablation", "none")).lower()
    ablation_seed = int(bpn_cfg.get("ablation_seed", seed + 2024))

    # 是否需要加载老师分布（BPN loss 或 teacher_pool/hybrid）
    need_teacher = use_bpn_in_loss or (use_ep_condition and ep_mode in ("teacher_pool", "hybrid"))
    teacher: Optional[Dict[str, Any]] = None
    p_teacher_t: Dict[str, torch.Tensor] = {}  # 用于 teacher_pool 的 Torch 版分布

    if need_teacher:
        teacher = _load_bpn_importance(importance_npz, num_nodes=N_nodes)
        if teacher is None:
            logger.warning(
                "Requested teacher importance (for %s), but failed to load. "
                "BPN loss / teacher_pool will be disabled.",
                importance_npz,
            )
            use_bpn_in_loss = False
        else:
            logger.info("Loaded BPN importance from %s.", importance_npz)
            rng = np.random.default_rng(ablation_seed)
            base_map = teacher["map_by_name"]  # {ep_name: np.ndarray[N]}

            # 先在 teacher 端点集合上做 ablation
            if ablation == "shuffle":
                names = list(base_map.keys())
                imp_vecs = [base_map[n] for n in names]
                perm = rng.permutation(len(names))
                ablated_map = {names[i]: imp_vecs[perm[i]] for i in range(len(names))}
                logger.info("BPN ablation=shuffle is ON.")
            elif ablation == "random":
                ablated_map = {}
                for name in base_map.keys():
                    v = rng.random(N_nodes).astype(np.float32)
                    s = v.sum()
                    if s <= 0:
                        v[:] = 1.0 / float(N_nodes)
                    else:
                        v /= s
                    ablated_map[name] = v
                logger.info("BPN ablation=random is ON.")
            else:
                if ablation != "none":
                    logger.warning("Unknown ablation='%s', fallback to 'none'.", ablation)
                ablated_map = base_map
                logger.info("BPN ablation=none.")

            # 再扩展到所有 dataset 中的 endpoints（缺失的用均匀分布）
            map_by_name_full: Dict[str, np.ndarray] = {}
            for e in ds.endpoints:
                name = str(e)
                vec = ablated_map.get(name, None)
                if vec is None:
                    # 若老师分布缺失该端点，退化为均匀分布
                    vec = np.ones(N_nodes, dtype=np.float32)
                    vec /= float(N_nodes)
                map_by_name_full[name] = vec
                p_teacher_t[name] = torch.from_numpy(vec.astype(np.float32))

            teacher["map_by_name"] = map_by_name_full

    # 如果模式需要 teacher_pool，但最终没拿到老师分布，则报错提示
    if use_ep_condition and ep_mode in ("teacher_pool", "hybrid") and (teacher is None or not p_teacher_t):
        raise RuntimeError(
            f"endpoint_conditioning.mode='{ep_mode}' 需要 BPN importance 文件，"
            f"但未能成功加载: {importance_npz}"
        )

    if use_bpn_in_loss and teacher is None:
        logger.info("BPN loss is enabled but teacher maps unavailable. Will skip BPN loss.")
        use_bpn_in_loss = False

    # ------------------------------------------------------------------
    # 构建模型
    # ------------------------------------------------------------------
    gnn_hidden = int(model_cfg.get("gnn_hidden", 64))
    gnn_layers = int(model_cfg.get("gnn_layers", 3))
    cnn_channels = model_cfg.get("cnn_channels", [16, 32])
    fusion_hidden = int(model_cfg.get("fusion_hidden", 64))

    bpn = BPN(d_in=d_node, hidden=gnn_hidden, layers=gnn_layers).to(device)
    cnn = PhysCNN(channels=tuple(cnn_channels), out_dim=gnn_hidden).to(device)

    # 端点 embedding（仅在 id / hybrid 下使用）
    ep_emb: Optional[EndpointEmbedding] = None
    ep_dropout: Optional[nn.Dropout] = None
    num_endpoints = len(ds)

    if use_ep_condition and ep_mode in ("id", "hybrid"):
        ep_emb = EndpointEmbedding(num_endpoints=num_endpoints, d_ep=d_ep_id).to(device)
        ep_dropout = nn.Dropout(p=drop_ep).to(device)
        logger.info(
            "Using endpoint ID embedding: num_endpoints=%d, d_ep=%d, dropout=%.2f",
            num_endpoints,
            d_ep_id,
            drop_ep,
        )

    # 回归 head：根据 ep_mode 决定 d_ep_in
    if use_ep_condition:
        d_node_emb = gnn_hidden  # BPN 节点 embedding 维度
        if ep_mode == "id":
            d_ep_in = d_ep_id
        elif ep_mode == "teacher_pool":
            d_ep_in = d_node_emb
        elif ep_mode == "hybrid":
            d_ep_in = d_ep_id + d_node_emb
        else:
            raise ValueError(f"Unknown endpoint_conditioning.mode={ep_mode}")
        head = FusionRegressorCond(
            d_gnn=gnn_hidden,
            d_cnn=gnn_hidden,
            d_ep=d_ep_in,
            hidden=fusion_hidden,
        ).to(device)
        logger.info(
            "Using endpoint-conditioned head (mode=%s, d_ep_in=%d).",
            ep_mode,
            d_ep_in,
        )
    else:
        head = FusionRegressor(d_gnn=gnn_hidden, d_cnn=gnn_hidden, hidden=fusion_hidden).to(device)
        logger.info("Using original FusionRegressor (no endpoint conditioning).")

    # 节点重要性 head（仅在开启 BPN loss 时创建，并加入优化器）
    imp_head: Optional[NodeImportanceHead] = None
    if use_bpn_in_loss and teacher is not None:
        imp_head = NodeImportanceHead(
            d_in=gnn_hidden,
            temperature=temperature,
            normalize="softmax",
        ).to(device)
        logger.info("NodeImportanceHead created for BPN auxiliary loss.")

    loss_fn = CplLoss(
        mse_weight=float(cfg["loss"].get("mse_weight", 1.0)),
        cpl_weight=float(cfg["loss"].get("cpl_weight", 0.1)),
    )

    # 优化器：主体参数 +（可选）ID embedding 参数单独 weight_decay
    param_groups: List[Dict[str, Any]] = [
        {
            "params": list(bpn.parameters()) + list(cnn.parameters()) + list(head.parameters()),
            "weight_decay": float(cfg["train"].get("weight_decay", 1.0e-4)),
        }
    ]
    if imp_head is not None:
        param_groups[0]["params"] += list(imp_head.parameters())
    if ep_emb is not None:
        # 对 ID embedding 稍微强一点的 weight decay，避免强记忆
        param_groups.append(
            {
                "params": list(ep_emb.parameters()),
                "weight_decay": 1.0e-3,
            }
        )

    opt = torch.optim.Adam(
        param_groups,
        lr=float(cfg["train"].get("lr", 1e-3)),
    )

    # ------------------------------------------------------------------
    # 训练 / 验证 划分（对小数据集更鲁棒）
    # ------------------------------------------------------------------
    train_split = float(data_cfg.get("train_split", 0.8))
    if not (0.0 < train_split <= 1.0):
        raise ValueError(f"train_split must be in (0, 1], got {train_split}")

    indices = list(range(n))
    random.shuffle(indices)

    # 使用 round 而不是直接 int，避免 n=1 时 0.8 → 0 的问题
    n_train = int(round(n * train_split))
    if n_train == 0 and n > 0:
        n_train = 1
    if n_train > n:
        n_train = n

    tr_idx = indices[:n_train]
    va_idx = indices[n_train:]  # 剩余样本作为验证集（可能为空）

    # ------------------------------------------------------------------
    # Debug：小数据过拟合检查（limit_train_samples）
    # ------------------------------------------------------------------
    debug_cfg = cfg.get("debug", {}) or {}
    limit_train = int(debug_cfg.get("limit_train_samples", 0))
    if limit_train > 0:
        old_n = len(tr_idx)
        tr_idx = tr_idx[:min(limit_train, len(tr_idx))]
        logger.info("Debug: limit_train_samples=%d (from %d)", len(tr_idx), old_n)

    logger.info("Split: %d train endpoints, %d val endpoints.", len(tr_idx), len(va_idx))

    # ------------------------------------------------------------------
    # BPN loss warmup + ramp-up 配置
    # ------------------------------------------------------------------
    warmup_ep = int(bpn_cfg.get("warmup_epochs", 0))
    ramp_ep = int(bpn_cfg.get("ramp_epochs", 0))

    def get_bpn_weight(epoch: int) -> float:
        if not (use_bpn_in_loss and teacher is not None):
            return 0.0
        if warmup_ep > 0 and epoch <= warmup_ep:
            return 0.0
        if ramp_ep > 0 and epoch <= warmup_ep + ramp_ep:
            step = epoch - warmup_ep
            return base_bpn_loss_weight * step / max(ramp_ep, 1)
        return base_bpn_loss_weight

    # ------------------------------------------------------------------
    # 训练 / 验证 epoch 循环
    # ------------------------------------------------------------------
    def run_epoch(idxs: List[int], train: bool = True, bpn_weight: float = 0.0) -> Tuple[float, float]:
        # 如果没有样本，避免对空数组算 mape 产生 warning
        if not idxs:
            return 0.0, float("nan")

        y_true_all: List[float] = []
        y_pred_all: List[float] = []
        total_loss_sum = 0.0
        main_loss_sum = 0.0
        bpn_loss_sum = 0.0
        count_bpn = 0

        bpn.train(train)
        cnn.train(train)
        head.train(train)
        if imp_head is not None:
            imp_head.train(train)
        if ep_emb is not None:
            ep_emb.train(train)

        # 图和物理 map 是设计级别的，共享一份即可
        g = ds.g.to(device)
        x = ds.x.to(device)
        maps = ds.maps_t.to(device)

        for idx in idxs:
            # ep_id 就是 endpoint 的索引 (0..len(ds)-1)，__getitem__ 里返回的第一个值
            ep_id, y, ci, ep_name = ds[idx]
            ep_name_str = str(ep_name)

            # 是否需要节点级 embedding（teacher_pool / hybrid / BPN loss）
            need_node_emb = (
                (use_ep_condition and ep_mode in ("teacher_pool", "hybrid"))
                or (imp_head is not None and teacher is not None and bpn_weight > 0.0)
            )

            # 前向：根据 need_node_emb 决定是否返回 node_emb
            if need_node_emb:
                g_emb, node_emb = bpn(g, x, return_node_emb=True)  # [d], [N, d]
            else:
                g_emb = bpn(g, x)                                 # [d]
                node_emb = None

            gnn_emb = g_emb.unsqueeze(0)      # [1, d]
            cnn_emb = cnn(maps)               # [1, d]

            # ====== 端点条件化：根据 ep_mode 构造 z_ep ======
            if use_ep_condition:
                if ep_mode == "teacher_pool":
                    assert node_emb is not None, "teacher_pool mode requires node_emb"
                    pt = p_teacher_t[ep_name_str].to(device)          # [N]
                    z_ep = torch.matmul(pt.unsqueeze(0), node_emb)    # [1, d_node]

                elif ep_mode == "id":
                    assert ep_emb is not None and ep_dropout is not None
                    ep_idx_t = torch.tensor([int(ep_id)], dtype=torch.long, device=device)  # [1]
                    z_ep = ep_dropout(ep_emb(ep_idx_t))                                    # [1, d_ep_id]

                elif ep_mode == "hybrid":
                    assert node_emb is not None
                    assert ep_emb is not None and ep_dropout is not None
                    pt = p_teacher_t[ep_name_str].to(device)          # [N]
                    z_struct = torch.matmul(pt.unsqueeze(0), node_emb)  # [1, d_node]
                    ep_idx_t = torch.tensor([int(ep_id)], dtype=torch.long, device=device)  # [1]
                    z_id = ep_dropout(ep_emb(ep_idx_t))                # [1, d_ep_id]
                    z_ep = torch.cat([z_id, z_struct], dim=-1)         # [1, d_ep_id + d_node]
                else:
                    raise ValueError(f"Unknown ep_mode={ep_mode}")

                y_pred = head(gnn_emb, cnn_emb, z_ep)   # [1]
            else:
                y_pred = head(gnn_emb, cnn_emb)         # [1]
            # ============================================================

            y_t = torch.tensor([y], dtype=torch.float32, device=device)
            main_loss = loss_fn(y_pred, y_t, gnn_emb, cpl_indices=[ci])

            loss = main_loss
            bpn_this = 0.0

            # 计算 BPN 辅助 loss（若可用且当前权重>0）
            if (
                bpn_weight > 0.0
                and imp_head is not None
                and node_emb is not None
                and teacher is not None
            ):
                # 需要该 endpoint 的老师分布
                p_teacher_np = teacher["map_by_name"].get(ep_name_str, None)
                if p_teacher_np is not None:
                    p_model = imp_head(node_emb)  # [N]
                    bpn_loss = _compute_bpn_loss(
                        p_model,
                        p_teacher_np,
                        loss_type=bpn_loss_type,
                    )
                    loss = loss + bpn_weight * bpn_loss
                    bpn_this = float(bpn_loss.detach().cpu())
                    bpn_loss_sum += bpn_this
                    count_bpn += 1

            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()

            total_loss_sum += float(loss.detach().cpu())
            main_loss_sum += float(main_loss.detach().cpu())
            y_true_all.append(y)
            y_pred_all.append(float(y_pred.detach().cpu()))

        # 将收集到的真值与预测值转为 numpy 数组
        y_true_np = np.array(y_true_all, dtype=np.float32)
        y_pred_np = np.array(y_pred_all, dtype=np.float32)

        # ====== 预测方差 / 范围 / R2 / MAE / RMSE 统计并打印 ======
        if y_true_np.size > 0:
            pred_std = float(np.std(y_pred_np))
            pred_min = float(np.min(y_pred_np))
            pred_max = float(np.max(y_pred_np))

            # R2、MAE、RMSE
            r2_val = float(r2_score(y_true_np, y_pred_np))
            mae = float(np.mean(np.abs(y_true_np - y_pred_np)))
            rmse = float(sqrt(np.mean((y_true_np - y_pred_np) ** 2)))

            avg_total = total_loss_sum / max(len(idxs), 1)
            avg_main = main_loss_sum / max(len(idxs), 1)
            avg_bpn = (bpn_loss_sum / max(count_bpn, 1)) if count_bpn > 0 else 0.0

            split_name = "train" if train else "val"
            logger.info(
                "Split=%s: total=%.4f main=%.4f bpn=%.4f (count=%d) "
                "pred_std=%.4f range=[%.4f, %.4f] R2=%.4f MAE=%.4f RMSE=%.4f",
                split_name,
                avg_total,
                avg_main,
                avg_bpn,
                count_bpn,
                pred_std,
                pred_min,
                pred_max,
                r2_val,
                mae,
                rmse,
            )
        # ============================================================

        avg_loss = total_loss_sum / max(len(idxs), 1)
        avg_mape = float(mape(y_true_np, y_pred_np))
        return avg_loss, avg_mape

    epochs = int(cfg["train"].get("epochs", 3))
    best_val = float("inf")
    save_dir = os.path.dirname(npz_path)
    os.makedirs(save_dir, exist_ok=True)

    for ep in range(1, epochs + 1):
        w_bpn = get_bpn_weight(ep)
        if w_bpn > 0.0:
            logger.info("Epoch %d: using BPN loss weight=%.4f", ep, w_bpn)

        tr_loss, tr_mape = run_epoch(tr_idx, train=True, bpn_weight=w_bpn)

        if va_idx:
            va_loss, va_mape = run_epoch(va_idx, train=False, bpn_weight=w_bpn)
            logger.info(
                "[Epoch %d/%d] train_loss=%.4f mape=%.4f | val_loss=%.4f mape=%.4f",
                ep,
                epochs,
                tr_loss,
                tr_mape,
                va_loss,
                va_mape,
            )
            if va_loss < best_val:
                best_val = va_loss
                state = {
                    "bpn": bpn.state_dict(),
                    "cnn": cnn.state_dict(),
                    "head": head.state_dict(),
                }
                if imp_head is not None:
                    state["imp_head"] = imp_head.state_dict()
                if ep_emb is not None:
                    state["ep_emb"] = ep_emb.state_dict()
                torch.save(
                    state,
                    os.path.join(save_dir, "best.pt"),
                )
        else:
            # 没有验证集时只打印训练指标
            logger.info(
                "[Epoch %d/%d] train_loss=%.4f mape=%.4f",
                ep,
                epochs,
                tr_loss,
                tr_mape,
            )

    metrics = {
        "train_samples": len(tr_idx),
        "val_samples": len(va_idx),
        "epochs": epochs,
        "use_bpn_in_loss": use_bpn_in_loss and (teacher is not None),
        "bpn_loss_type": bpn_loss_type if (use_bpn_in_loss and teacher is not None) else "disabled",
        "bpn_loss_weight_base": base_bpn_loss_weight if (use_bpn_in_loss and teacher is not None) else 0.0,
        "warmup_epochs": warmup_ep,
        "ramp_epochs": ramp_ep,
        "use_endpoint_condition": use_ep_condition,
        "endpoint_dim_id": d_ep_id if use_ep_condition and ep_mode in ("id", "hybrid") else 0,
        "endpoint_mode": ep_mode,
    }
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Training done. Metrics saved to %s", os.path.join(save_dir, "metrics.json"))


def main():
    parser = argparse.ArgumentParser(description="Train CPL-based model with optional BPN auxiliary loss.")
    parser.add_argument("--config", type=str, default="configs/model.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    train_loop(cfg)


if __name__ == "__main__":
    main()