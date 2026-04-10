from __future__ import annotations
import argparse
import json
import logging
import os
import random
import glob
from typing import Dict, Tuple, List, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import yaml
from torch.utils.data import Dataset, ConcatDataset

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
        self.npz_path = npz_path
        self.teacher_map = None
        self.p_teacher_t = {}

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
        self.endpoints = self.data["endpoints"]          # [E]
        
        # ===== 极度稳健的 Min-Max 缩放到 [0, 1] =====
        raw_y = self.data["y_arrival"].astype(np.float32)
        if raw_y.size > 0:
            y_min = float(np.min(raw_y))
            y_max = float(np.max(raw_y))
            y_range = y_max - y_min
            
            # 如果所有的值都一样，强制拉平到 0.5
            if y_range < 1e-6:
                self.y_arrival = np.full_like(raw_y, 0.5)
            else:
                self.y_arrival = (raw_y - y_min) / y_range
        else:
            self.y_arrival = raw_y # 空数组原样返回
        # ========================================================
        
        self.cpl_indices = self.data["cpl_indices"]      # list[list[int]]

        # 可视化需要的 name_to_idx / idx_to_name
        self.name_to_idx: Optional[Dict[str, int]] = None
        self.idx_to_name: Optional[Dict[int, str]] = None
        if "name_to_idx" in self.data.files:
            try:
                pairs = self.data["name_to_idx"]
                mapping: Dict[str, int] = {}
                for name, idx in pairs:
                    mapping[str(name)] = int(idx)
                self.name_to_idx = mapping
                self.idx_to_name = {idx: name for name, idx in mapping.items()}
            except Exception as e:
                logger.warning("Failed to parse name_to_idx from dataset npz: %s", e)

        import dgl
        import torch as T

        src = T.from_numpy(self.edges[0].astype(np.int64))
        dst = T.from_numpy(self.edges[1].astype(np.int64))
        self.g = dgl.graph((src, dst), num_nodes=self.node_features.shape[0])

        if "edge_weight" in self.data.files:
            self.g.edata["weight"] = T.from_numpy(self.data["edge_weight"]).float()
        else:
            self.g.edata["weight"] = T.ones(self.g.num_edges(), dtype=T.float32)

        self.x = T.from_numpy(self.node_features.astype(np.float32))
        self.maps_t = T.from_numpy(self.maps.astype(np.float32)).unsqueeze(0)

    def __len__(self) -> int:
        return len(self.y_arrival)

    def __getitem__(self, idx: int):
        y = float(self.y_arrival[idx])
        ci = list(self.cpl_indices[idx])
        ep_name = str(self.endpoints[idx])
        ep_id = idx
        return ep_id, y, ci, ep_name, self


def _infer_bpn_importance_path(dataset_npz: str) -> str:
    base = os.path.splitext(os.path.basename(dataset_npz))[0]
    out_dir = os.path.dirname(dataset_npz)
    return os.path.join(out_dir, f"{base}_bpn_importance.npz")


def _load_bpn_importance(importance_npz: str, num_nodes: int) -> Optional[Dict[str, Any]]:
    if not os.path.exists(importance_npz):
        return None
    arr = np.load(importance_npz, allow_pickle=True)
    endpoints = [str(e) for e in arr["endpoints"]]
    importance = arr["importance"].astype(np.float32)
    if importance.ndim != 2 or importance.shape[1] != num_nodes:
        return None
    imp = importance.copy()
    sums = np.clip(imp.sum(axis=1, keepdims=True), 1e-12, None)
    imp = imp / sums
    mp = {endpoints[i]: imp[i] for i in range(len(endpoints))}
    return {"endpoints": endpoints, "importance": imp, "map_by_name": mp}


def _compute_bpn_loss(
    p_model: torch.Tensor,
    p_teacher_ref: np.ndarray | torch.Tensor,
    loss_type: str = "kl",
) -> torch.Tensor:
    device = p_model.device
    if isinstance(p_teacher_ref, torch.Tensor):
        p_teacher = p_teacher_ref.to(device=device, dtype=p_model.dtype)
    else:
        p_teacher = torch.from_numpy(np.asarray(p_teacher_ref)).to(device=device, dtype=p_model.dtype)

    eps = 1e-12
    if loss_type == "mse":
        return F.mse_loss(p_model, p_teacher)
    if loss_type == "l1":
        return torch.mean(torch.abs(p_model - p_teacher))

    p_t = torch.clamp(p_teacher, min=eps)
    p_m = torch.clamp(p_model, min=eps)
    kl = torch.sum(p_t * (torch.log(p_t) - torch.log(p_m)))
    return kl


def _topk_recall_single(y_true: np.ndarray, y_pred: np.ndarray, frac: float = 0.10, mode: str = "high") -> float:
    if y_true.size == 0:
        return float("nan")
    frac = 0.10 if frac <= 0.0 else min(float(frac), 1.0)
    n = int(y_true.shape[0])
    k = max(1, int(round(n * frac)))
    k = min(k, n)

    if mode == "low":
        true_idx = set(np.argpartition(y_true, k - 1)[:k].tolist())
        pred_idx = set(np.argpartition(y_pred, k - 1)[:k].tolist())
    else:
        true_idx = set(np.argpartition(y_true, n - k)[-k:].tolist())
        pred_idx = set(np.argpartition(y_pred, n - k)[-k:].tolist())

    return float(len(true_idx & pred_idx) / max(k, 1))


def _groupwise_topk_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    frac: float = 0.10,
    mode: str = "high",
) -> float:
    if y_true.size == 0:
        return float("nan")

    vals: List[float] = []
    for gid in np.unique(groups):
        mask = groups == gid
        yt = y_true[mask]
        yp = y_pred[mask]
        if yt.size == 0:
            continue
        vals.append(_topk_recall_single(yt, yp, frac=frac, mode=mode))

    return float(np.mean(vals)) if vals else float("nan")


# 【核心修复 1】：剔除 tie 当成错排的 Bug
def _kendall_tau_np(y_true: np.ndarray, y_pred: np.ndarray, mode: str = "high", tie_epsilon: float = 1e-12) -> float:
    n = int(y_true.shape[0])
    if n < 2:
        return float("nan")

    idx_i, idx_j = np.triu_indices(n, k=1)
    
    if mode == "low":
        y_true = -y_true
        y_pred = -y_pred
        
    diff_t = y_true[idx_i] - y_true[idx_j]
    diff_p = y_pred[idx_i] - y_pred[idx_j]

    # 只统计 y_true 和 y_pred 都真正有差别的 pair
    valid = (np.abs(diff_t) > tie_epsilon) & (np.abs(diff_p) > tie_epsilon)
    if not np.any(valid):
        return 0.0

    sign_t = np.sign(diff_t[valid])
    sign_p = np.sign(diff_p[valid])

    concord = np.sum(sign_t == sign_p)
    discord = np.sum(sign_t != sign_p)
    denom = max(int(concord + discord), 1)
    return float((concord - discord) / denom)


def _groupwise_kendall_tau(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    mode: str = "high",
    max_samples_per_group: int = 1024,
    seed: int = 42,
) -> float:
    if y_true.size == 0:
        return float("nan")

    rng = np.random.default_rng(seed)
    vals: List[float] = []

    for gid in np.unique(groups):
        mask = groups == gid
        yt = y_true[mask]
        yp = y_pred[mask]
        if yt.size < 2:
            continue

        if max_samples_per_group > 0 and yt.size > max_samples_per_group:
            sel = rng.choice(yt.size, size=max_samples_per_group, replace=False)
            yt = yt[sel]
            yp = yp[sel]

        tau = _kendall_tau_np(yt, yp, mode=mode)
        if not np.isnan(tau):
            vals.append(float(tau))

    return float(np.mean(vals)) if vals else float("nan")


def dump_node_importance_for_dataset(
    npz_path: str,
    ds: EndpointDataset,
    bpn: BPN,
    imp_head: Optional[NodeImportanceHead],
    teacher: Optional[Dict[str, Any]],
    device: torch.device,
    vis_cfg: Dict[str, Any],
) -> None:
    dump_flag = bool(vis_cfg.get("dump_node_importance", False))
    if not dump_flag:
        return

    source = str(vis_cfg.get("source", "model")).lower()
    if source not in ("model", "teacher"):
        source = "model"

    eps_cfg = vis_cfg.get("endpoints", [])
    target_eps = {str(e) for e in eps_cfg} if eps_cfg else None

    outdir = vis_cfg.get("outdir", None)
    if not outdir:
        outdir = os.path.join(os.path.dirname(npz_path), "vis_node_importance")
    os.makedirs(outdir, exist_ok=True)

    if source == "model" and imp_head is None:
        return
    if source == "teacher" and (teacher is None or "map_by_name" not in teacher):
        return

    N_nodes = ds.node_features.shape[0]
    node_names: List[str] = [f"node_{i}" for i in range(N_nodes)]
    if getattr(ds, "idx_to_name", None):
        for idx, name in ds.idx_to_name.items():
            if 0 <= idx < N_nodes:
                node_names[idx] = str(name)

    p_model_np: Optional[np.ndarray] = None
    if source == "model":
        bpn.eval()
        imp_head.eval()
        with torch.no_grad():
            g = ds.g.to(device)
            x = ds.x.to(device)
            g_emb, node_emb = bpn(g, x, return_node_emb=True)
            p = imp_head(node_emb)
            p = torch.clamp(p, min=0.0)
            s = p.sum()
            if s <= 0:
                p = torch.ones_like(p) / float(p.numel())
            else:
                p = p / s
            p_model_np = p.detach().cpu().numpy().astype(np.float32)

    teacher_map: Dict[str, np.ndarray] = {}
    if source == "teacher":
        teacher_map = teacher["map_by_name"]

    sel_endpoints: List[str] = []
    importance_rows: List[np.ndarray] = []

    for ep in ds.endpoints:
        ep_name = str(ep)
        if target_eps is not None and ep_name not in target_eps:
            continue
        if source == "model":
            sel_endpoints.append(ep_name)
            importance_rows.append(p_model_np.copy())
        else:
            vec = teacher_map.get(ep_name, None)
            if vec is None:
                vec = np.ones(N_nodes, dtype=np.float32) / float(N_nodes)
            sel_endpoints.append(ep_name)
            importance_rows.append(vec.astype(np.float32))

    if not sel_endpoints:
        return

    endpoints_np = np.array(sel_endpoints, dtype=object)
    importance_np = np.stack(importance_rows, axis=0)
    node_names_np = np.array(node_names, dtype=object)

    base = os.path.splitext(os.path.basename(npz_path))[0]
    tag = source
    out_path = os.path.join(outdir, f"{base}_node_importance_{tag}.npz")
    np.savez_compressed(out_path, endpoints=endpoints_np, importance=importance_np, node_names=node_names_np)
    logger.info("Exported node importance to %s", out_path)


def train_loop(cfg: Dict[str, object]):
    device = torch.device(cfg.get("device", "cpu"))
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    data_cfg = cfg.get("data", {}) or {}
    model_cfg = cfg.get("model", {}) or {}
    train_cfg = cfg.get("train", {}) or {}
    loss_cfg = cfg.get("loss", {}) or {}
    debug_cfg = cfg.get("debug", {}) or {}

    # ------------------------------------------------------------------
    # 多设计加载
    # ------------------------------------------------------------------
    base_npz_path = data_cfg.get("dataset_npz")
    if not base_npz_path:
        raise ValueError("Please provide dataset_npz in config to infer directory.")

    data_dir = os.path.dirname(base_npz_path)
    all_files = glob.glob(os.path.join(data_dir, "*.npz"))
    valid_files = [f for f in all_files if "importance" not in os.path.basename(f)]
    valid_files.sort()

    datasets = []
    for f in valid_files:
        ds = EndpointDataset(npz_path=f)
        if len(ds) > 0:
            datasets.append(ds)
            logger.info("Loaded sub-dataset: %s with %d endpoints", f, len(ds))

    if not datasets:
        raise RuntimeError(f"No valid datasets found in {data_dir}")

    concat_ds = ConcatDataset(datasets)
    n = len(concat_ds)
    d_node = datasets[0].node_features.shape[1]
    logger.info("Concatenated %d endpoints from %d designs.", n, len(datasets))

    global_idx_to_design = np.empty(n, dtype=np.int64)
    all_targets_global = np.empty(n, dtype=np.float32)
    cursor = 0
    for design_id, ds in enumerate(datasets):
        m = len(ds)
        global_idx_to_design[cursor: cursor + m] = design_id
        all_targets_global[cursor: cursor + m] = np.asarray(ds.y_arrival, dtype=np.float32)
        cursor += m

    # ------------------------------------------------------------------
    # 端点条件化配置
    # ------------------------------------------------------------------
    legacy_use_ep = bool(model_cfg.get("use_endpoint_condition", False))
    legacy_d_ep = int(model_cfg.get("endpoint_dim", 16))
    ep_cond_cfg = model_cfg.get("endpoint_conditioning", {}) or {}
    ep_mode = str(ep_cond_cfg.get("mode", "id" if legacy_use_ep else "none")).lower()
    d_ep_id = int(ep_cond_cfg.get("d_ep", legacy_d_ep))
    drop_ep = float(ep_cond_cfg.get("dropout", 0.0))
    use_ep_condition = ep_mode != "none"

    # ------------------------------------------------------------------
    # 为每个设计分别加载 BPN 老师分布
    # ------------------------------------------------------------------
    bpn_cfg = cfg.get("bpn", {}) or {}
    use_bpn_in_loss = bool(bpn_cfg.get("use_bpn_in_loss", False))
    bpn_loss_type = str(bpn_cfg.get("bpn_loss_type", "kl")).lower()
    base_bpn_loss_weight = float(bpn_cfg.get("bpn_loss_weight", 0.1))
    temperature = float(bpn_cfg.get("temperature", 1.0))
    ablation = str(bpn_cfg.get("ablation", "none")).lower()
    ablation_seed = int(bpn_cfg.get("ablation_seed", seed + 2024))

    need_teacher = use_bpn_in_loss or (use_ep_condition and ep_mode in ("teacher_pool", "hybrid"))

    has_any_teacher = False
    if need_teacher:
        for ds in datasets:
            imp_path = _infer_bpn_importance_path(ds.npz_path)
            N_ds_nodes = ds.node_features.shape[0]
            teacher = _load_bpn_importance(imp_path, num_nodes=N_ds_nodes)

            # 【核心修复 2】：不允许 teacher_pool 静默退化为 uniform
            if teacher is None:
                if use_ep_condition and ep_mode in ("teacher_pool", "hybrid"):
                    raise RuntimeError(f"Teacher not found for {ds.npz_path}. teacher_pool mode requires strict BPN maps.")
                else:
                    logger.warning("Teacher not found for %s. Fallback to uniform.", ds.npz_path)
                    ds.teacher_map = {
                        str(ep): np.ones(N_ds_nodes, dtype=np.float32) / N_ds_nodes
                        for ep in ds.endpoints
                    }
            else:
                has_any_teacher = True
                
                # 检查覆盖率
                missing = [str(e) for e in ds.endpoints if str(e) not in teacher["map_by_name"]]
                if missing and use_ep_condition and ep_mode in ("teacher_pool", "hybrid"):
                    raise RuntimeError(f"Missing teacher maps for {len(missing)} endpoints in {ds.npz_path}. Example: {missing[0]}")
                
                rng = np.random.default_rng(ablation_seed)
                base_map = teacher["map_by_name"]

                if ablation == "shuffle":
                    names = list(base_map.keys())
                    imp_vecs = [base_map[n] for n in names]
                    perm = rng.permutation(len(names))
                    ablated_map = {names[i]: imp_vecs[perm[i]] for i in range(len(names))}
                elif ablation == "random":
                    ablated_map = {}
                    for name in base_map.keys():
                        v = rng.random(N_ds_nodes).astype(np.float32)
                        s = v.sum()
                        v = v / s if s > 0 else np.ones(N_ds_nodes, dtype=np.float32) / N_ds_nodes
                        ablated_map[name] = v
                else:
                    ablated_map = base_map

                map_by_name_full = {}
                for e in ds.endpoints:
                    name = str(e)
                    vec = ablated_map.get(name, None)
                    if vec is None:
                        vec = np.ones(N_ds_nodes, dtype=np.float32) / N_ds_nodes
                    map_by_name_full[name] = vec
                ds.teacher_map = map_by_name_full

            ds.p_teacher_t = {name: torch.from_numpy(vec) for name, vec in ds.teacher_map.items()}

    if need_teacher and not has_any_teacher:
        logger.warning("Need teacher but NONE found across all designs. Disabling BPN loss.")
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

    ep_emb: Optional[EndpointEmbedding] = None
    ep_dropout: Optional[nn.Dropout] = None

    num_endpoints_global = n
    if use_ep_condition and ep_mode in ("id", "hybrid"):
        ep_emb = EndpointEmbedding(num_endpoints=num_endpoints_global, d_ep=d_ep_id).to(device)
        ep_dropout = nn.Dropout(p=drop_ep).to(device)

    if use_ep_condition:
        d_node_emb = gnn_hidden
        if ep_mode == "id":
            d_ep_in = d_ep_id
        elif ep_mode in ("teacher_pool", "node"):
            d_ep_in = d_node_emb
        elif ep_mode == "hybrid":
            d_ep_in = d_ep_id + d_node_emb
        else:
            raise ValueError(f"Unknown mode={ep_mode}")
        head = FusionRegressorCond(d_gnn=gnn_hidden, d_cnn=gnn_hidden, d_ep=d_ep_in, hidden=fusion_hidden).to(device)
    else:
        head = FusionRegressor(d_gnn=gnn_hidden, d_cnn=gnn_hidden, hidden=fusion_hidden).to(device)

    imp_head: Optional[NodeImportanceHead] = None
    if use_bpn_in_loss and has_any_teacher:
        imp_head = NodeImportanceHead(d_in=gnn_hidden, temperature=temperature, normalize="softmax").to(device)

    # ------------------------------------------------------------------
    # 新复合损失
    # ------------------------------------------------------------------
    loss_fn = CplLoss(
        mse_weight=float(loss_cfg.get("mse_weight", 1.0)),
        rank_weight=float(
            loss_cfg.get("rank_weight", loss_cfg.get("ranking_weight", loss_cfg.get("pairwise_weight", 0.2)))
        ),
        cpl_weight=float(loss_cfg.get("cpl_weight", 0.1)),
        critical_fraction=float(loss_cfg.get("critical_fraction", loss_cfg.get("topk_fraction", 0.10))),
        critical_weight=float(loss_cfg.get("critical_weight", loss_cfg.get("cp_aware_weight", 2.0))),
        critical_mode=str(loss_cfg.get("critical_mode", "high")).lower(),
        pairwise_margin=float(loss_cfg.get("pairwise_margin", 0.0)),
        pairwise_loss=str(loss_cfg.get("pairwise_loss", "logistic")).lower(),
        pair_gap_power=float(loss_cfg.get("pair_gap_power", 0.0)),
        tie_epsilon=float(loss_cfg.get("tie_epsilon", 1e-8)),
        max_pairs=int(loss_cfg.get("max_pairs", 0)),
    )

    loss_batch_size = int(train_cfg.get("batch_size", loss_cfg.get("ranking_batch_size", 16)))
    loss_batch_size = max(loss_batch_size, 1)

    recall_top_fraction = float(loss_cfg.get("recall_top_fraction", loss_fn.critical_fraction if loss_fn.critical_fraction > 0 else 0.10))
    if recall_top_fraction <= 0.0:
        recall_top_fraction = 0.10
    tau_max_samples_per_group = int(loss_cfg.get("tau_max_samples_per_group", 1024))

    logger.info(
        "Loss setup: mse_w=%.3f rank_w=%.3f cpl_w=%.3f critical_frac=%.3f critical_w=%.3f mode=%s micro_batch=%d",
        loss_fn.mse_weight,
        loss_fn.rank_weight,
        loss_fn.cpl_weight,
        loss_fn.critical_fraction,
        loss_fn.critical_weight,
        loss_fn.critical_mode,
        loss_batch_size,
    )
    if loss_fn.rank_weight > 0.0 and loss_batch_size < 2:
        logger.warning("batch_size < 2, pairwise ranking will be ineffective.")

    param_groups: List[Dict[str, Any]] = [{
        "params": list(bpn.parameters()) + list(cnn.parameters()) + list(head.parameters()),
        "weight_decay": float(train_cfg.get("weight_decay", 1.0e-4)),
    }]
    if imp_head is not None:
        param_groups[0]["params"] += list(imp_head.parameters())
    if ep_emb is not None:
        param_groups.append({"params": list(ep_emb.parameters()), "weight_decay": 1.0e-3})

    opt = torch.optim.Adam(param_groups, lr=float(train_cfg.get("lr", 1e-3)))

    train_split = float(data_cfg.get("train_split", 0.8))
    indices = list(range(n))
    random.shuffle(indices)
    n_train = int(round(n * train_split))
    if n_train == 0 and n > 0:
        n_train = 1
    if n_train > n:
        n_train = n

    tr_idx = indices[:n_train]
    va_idx = indices[n_train:]

    limit_train = int(debug_cfg.get("limit_train_samples", 0))
    if limit_train > 0:
        tr_idx = tr_idx[: min(limit_train, len(tr_idx))]

    def _compute_design_thresholds(split_indices: List[int]) -> Dict[int, float]:
        crit_frac = float(loss_fn.critical_fraction)
        crit_weight = float(loss_fn.critical_weight)
        crit_mode = str(loss_fn.critical_mode).lower()

        if crit_frac <= 0.0 or crit_weight <= 1.0 or not split_indices:
            return {}

        q = crit_frac if crit_mode == "low" else 1.0 - crit_frac
        q = float(np.clip(q, 0.0, 1.0))

        buckets: Dict[int, List[float]] = {}
        for gi in split_indices:
            did = int(global_idx_to_design[gi])
            buckets.setdefault(did, []).append(float(all_targets_global[gi]))

        thresholds: Dict[int, float] = {}
        for did, vals in buckets.items():
            arr = np.asarray(vals, dtype=np.float32)
            if arr.size > 0:
                thresholds[did] = float(np.quantile(arr, q))
        return thresholds

    train_design_thresholds = _compute_design_thresholds(tr_idx)
    val_design_thresholds = _compute_design_thresholds(va_idx)

    def _build_epoch_order(split_indices: List[int], train: bool = True) -> List[int]:
        if not split_indices:
            return []

        buckets: Dict[int, List[int]] = {}
        for gi in split_indices:
            did = int(global_idx_to_design[gi])
            buckets.setdefault(did, []).append(gi)

        design_ids = list(buckets.keys())
        if train:
            random.shuffle(design_ids)
            for did in design_ids:
                random.shuffle(buckets[did])
        else:
            design_ids.sort()

        ordered: List[int] = []
        for did in design_ids:
            ordered.extend(buckets[did])
        return ordered

    def _cp_sample_weight(y_value: float, design_id: int, design_thresholds: Dict[int, float]) -> float:
        if not design_thresholds:
            return 1.0

        thr = design_thresholds.get(design_id, None)
        if thr is None:
            return 1.0

        if loss_fn.critical_mode == "low":
            return float(loss_fn.critical_weight) if float(y_value) <= thr else 1.0
        return float(loss_fn.critical_weight) if float(y_value) >= thr else 1.0

    warmup_ep = int(bpn_cfg.get("warmup_epochs", 0))
    ramp_ep = int(bpn_cfg.get("ramp_epochs", 0))

    def get_bpn_weight(epoch: int) -> float:
        if not (use_bpn_in_loss and has_any_teacher):
            return 0.0
        if warmup_ep > 0 and epoch <= warmup_ep:
            return 0.0
        if ramp_ep > 0 and epoch <= warmup_ep + ramp_ep:
            return base_bpn_loss_weight * (epoch - warmup_ep) / max(ramp_ep, 1)
        return base_bpn_loss_weight

    def run_epoch(
        idxs: List[int],
        train: bool = True,
        bpn_weight: float = 0.0,
        design_thresholds: Optional[Dict[int, float]] = None,
    ) -> Tuple[float, float]:
        if not idxs:
            return 0.0, float("nan")

        work_idxs = _build_epoch_order(idxs, train=train)

        y_true_all: List[float] = []
        y_pred_all: List[float] = []
        group_all: List[int] = []

        total_loss_sum = 0.0
        core_loss_sum = 0.0
        mse_term_sum = 0.0
        rank_term_sum = 0.0
        cpl_term_sum = 0.0
        bpn_term_sum = 0.0
        num_samples = 0

        bpn.train(train)
        cnn.train(train)
        head.train(train)
        if imp_head is not None:
            imp_head.train(train)
        if ep_emb is not None:
            ep_emb.train(train)

        pending_preds: List[torch.Tensor] = []
        pending_targets: List[torch.Tensor] = []
        pending_graph_embs: List[torch.Tensor] = []
        pending_cpl_indices: List[List[int]] = []
        pending_sample_weights: List[torch.Tensor] = []
        pending_group_ids: List[int] = []
        pending_bpn_losses: List[torch.Tensor] = []
        pending_design_id: Optional[int] = None

        def flush_pending() -> None:
            nonlocal total_loss_sum, core_loss_sum
            nonlocal mse_term_sum, rank_term_sum, cpl_term_sum, bpn_term_sum
            nonlocal num_samples

            if not pending_preds:
                return

            batch_size_cur = len(pending_preds)

            y_pred_b = torch.stack(pending_preds, dim=0).reshape(-1)
            y_true_b = torch.stack(pending_targets, dim=0).reshape(-1)
            gnn_emb_b = torch.stack(pending_graph_embs, dim=0)
            sample_weight_b = torch.stack(pending_sample_weights, dim=0).reshape(-1)
            group_ids_b = torch.tensor(pending_group_ids, dtype=torch.long, device=device)

            core_loss, loss_detail = loss_fn(
                y_pred=y_pred_b,
                y_true=y_true_b,
                gnn_emb=gnn_emb_b,
                cpl_indices=pending_cpl_indices,
                sample_weight=sample_weight_b,
                group_ids=group_ids_b,
                return_details=True,
            )

            total_loss = core_loss
            if bpn_weight > 0.0 and pending_bpn_losses:
                bpn_aux = torch.stack(pending_bpn_losses, dim=0).mean()
                total_loss = total_loss + bpn_weight * bpn_aux
                bpn_term_sum += float((bpn_weight * bpn_aux).detach().cpu()) * batch_size_cur

            if train:
                opt.zero_grad(set_to_none=True)
                total_loss.backward()
                opt.step()

            total_loss_sum += float(total_loss.detach().cpu()) * batch_size_cur
            core_loss_sum += float(core_loss.detach().cpu()) * batch_size_cur
            mse_term_sum += float(loss_detail["mse_term"].detach().cpu()) * batch_size_cur
            rank_term_sum += float(loss_detail["rank_term"].detach().cpu()) * batch_size_cur
            cpl_term_sum += float(loss_detail["cpl_term"].detach().cpu()) * batch_size_cur
            num_samples += batch_size_cur

            pending_preds.clear()
            pending_targets.clear()
            pending_graph_embs.clear()
            pending_cpl_indices.clear()
            pending_sample_weights.clear()
            pending_group_ids.clear()
            pending_bpn_losses.clear()

        grad_ctx = torch.enable_grad() if train else torch.no_grad()
        with grad_ctx:
            for global_idx in work_idxs:
                design_id = int(global_idx_to_design[global_idx])

                if pending_design_id is not None and design_id != pending_design_id and pending_preds:
                    flush_pending()
                    pending_design_id = None

                if pending_design_id is None:
                    pending_design_id = design_id

                local_ep_id, y, ci, ep_name, current_ds = concat_ds[global_idx]
                ep_name_str = str(ep_name)

                g = current_ds.g.to(device)
                x = current_ds.x.to(device)
                maps = current_ds.maps_t.to(device)

                need_node_emb = (
                    (use_ep_condition and ep_mode in ("teacher_pool", "hybrid", "node"))  # <--- 把 "node" 加进这里
                    or (imp_head is not None and current_ds.teacher_map is not None and bpn_weight > 0.0)
                )

                if need_node_emb:
                    g_emb, node_emb = bpn(g, x, return_node_emb=True)
                else:
                    g_emb = bpn(g, x)
                    node_emb = None

                gnn_emb = g_emb.unsqueeze(0)
                cnn_emb = cnn(maps)

                if use_ep_condition:
                    if ep_mode == "teacher_pool":
                        pt = current_ds.p_teacher_t[ep_name_str].to(device=device, dtype=node_emb.dtype)
                        z_ep = torch.matmul(pt.unsqueeze(0), node_emb)

                    elif ep_mode == "id":
                        ep_idx_t = torch.tensor([int(global_idx)], dtype=torch.long, device=device)
                        z_ep = ep_dropout(ep_emb(ep_idx_t))
                    
                    # ===== 【新增代码】：使用端点自身的原生图特征 =====
                    elif ep_mode == "node":
                        ep_idx = current_ds.name_to_idx.get(ep_name_str, None)
                        if ep_idx is not None and 0 <= ep_idx < node_emb.shape[0]:
                            z_ep = node_emb[ep_idx].unsqueeze(0)   # <--- 加上 .unsqueeze(0)
                        else:
                            # 兜底：如果没找到端点，用相关路径起点的均值特征
                            valid_ci = [i for i in ci if 0 <= i < node_emb.shape[0]]
                            if valid_ci:
                                ci_tensor = torch.tensor(valid_ci, dtype=torch.long, device=device)
                                z_ep = node_emb[ci_tensor].mean(dim=0).unsqueeze(0)   # <--- 加上 .unsqueeze(0)
                            else:
                                z_ep = node_emb.mean(dim=0).unsqueeze(0)   # <--- 加上 .unsqueeze(0)
                    # =================================================

                    elif ep_mode == "hybrid":
                        pt = current_ds.p_teacher_t[ep_name_str].to(device=device, dtype=node_emb.dtype)
                        z_struct = torch.matmul(pt.unsqueeze(0), node_emb)
                        ep_idx_t = torch.tensor([int(global_idx)], dtype=torch.long, device=device)
                        z_id = ep_dropout(ep_emb(ep_idx_t))
                        z_ep = torch.cat([z_id, z_struct], dim=-1)

                    else:
                        raise ValueError(f"Unknown endpoint conditioning mode={ep_mode}")

                    y_pred = head(gnn_emb, cnn_emb, z_ep)
                else:
                    y_pred = head(gnn_emb, cnn_emb)

                y_pred_scalar = y_pred.reshape(-1)[0]
                y_target_scalar = torch.tensor(float(y), dtype=torch.float32, device=device)
                sample_w_scalar = torch.tensor(
                    _cp_sample_weight(float(y), design_id, design_thresholds or {}),
                    dtype=torch.float32,
                    device=device,
                )

                pending_preds.append(y_pred_scalar)
                pending_targets.append(y_target_scalar)
                pending_graph_embs.append(gnn_emb.squeeze(0))
                pending_cpl_indices.append(list(ci))
                pending_sample_weights.append(sample_w_scalar)
                pending_group_ids.append(design_id)

                if bpn_weight > 0.0 and imp_head is not None and node_emb is not None and current_ds.teacher_map is not None:
                    p_teacher_ref = None
                    if getattr(current_ds, "p_teacher_t", None):
                        p_teacher_ref = current_ds.p_teacher_t.get(ep_name_str, None)
                    if p_teacher_ref is None and current_ds.teacher_map is not None:
                        p_teacher_ref = current_ds.teacher_map.get(ep_name_str, None)

                    if p_teacher_ref is not None:
                        p_model = imp_head(node_emb)
                        pending_bpn_losses.append(
                            _compute_bpn_loss(p_model, p_teacher_ref, loss_type=bpn_loss_type)
                        )

                y_true_all.append(float(y))
                y_pred_all.append(float(y_pred_scalar.detach().cpu()))
                group_all.append(design_id)

                if len(pending_preds) >= loss_batch_size:
                    flush_pending()
                    pending_design_id = None

            flush_pending()

        y_true_np = np.array(y_true_all, dtype=np.float32)
        y_pred_np = np.array(y_pred_all, dtype=np.float32)
        group_np = np.array(group_all, dtype=np.int64)

        avg_total = total_loss_sum / max(num_samples, 1)
        avg_core = core_loss_sum / max(num_samples, 1)
        avg_mse_term = mse_term_sum / max(num_samples, 1)
        avg_rank_term = rank_term_sum / max(num_samples, 1)
        avg_cpl_term = cpl_term_sum / max(num_samples, 1)
        avg_bpn_term = bpn_term_sum / max(num_samples, 1)

        if y_true_np.size > 0:
            # 【核心修复 3】：在合并打印前，按设计分别打印统计值和 Tau，揪出预测塌缩的设计
            for gid in np.unique(group_np):
                mask = group_np == gid
                yt_g = y_true_np[mask]
                yp_g = y_pred_np[mask]
                if yt_g.size > 1:
                    tau_g = _kendall_tau_np(yt_g, yp_g, mode=loss_fn.critical_mode)
                    recall_g = _topk_recall_single(yt_g, yp_g, frac=recall_top_fraction, mode=loss_fn.critical_mode)
                    logger.info("Split=%s | Design %d: std_true=%.4f std_pred=%.4f Tau=%.4f Recall=%.4f", 
                                "train" if train else "val", gid, float(np.std(yt_g)), float(np.std(yp_g)), tau_g, recall_g)
            
            r2_val = float(r2_score(y_true_np, y_pred_np)) if y_true_np.size > 1 else float("nan")
            
            tau_val = _groupwise_kendall_tau(
                y_true_np,
                y_pred_np,
                group_np,
                mode=loss_fn.critical_mode,
                max_samples_per_group=tau_max_samples_per_group,
                seed=seed + (1 if train else 2),
            )
            recall_val = _groupwise_topk_recall(
                y_true_np,
                y_pred_np,
                group_np,
                frac=recall_top_fraction,
                mode=loss_fn.critical_mode,
            )

            logger.info(
                "Split=%s [Global]: total=%.4f core=%.4f mse=%.4f rank=%.4f cpl=%.4f bpn=%.4f R2=%.4f Tau=%.4f Recall@%.0f%%=%.4f",
                "train" if train else "val",
                avg_total,
                avg_core,
                avg_mse_term,
                avg_rank_term,
                avg_cpl_term,
                avg_bpn_term,
                r2_val,
                tau_val,
                recall_top_fraction * 100.0,
                recall_val,
            )

        mape_val = float(mape(y_true_np, y_pred_np)) if y_true_np.size > 0 else float("nan")
        return avg_total, mape_val

    epochs = int(train_cfg.get("epochs", 3))
    best_val = float("inf")
    save_dir = os.path.dirname(base_npz_path)
    os.makedirs(save_dir, exist_ok=True)

    for ep in range(1, epochs + 1):
        w_bpn = get_bpn_weight(ep)
        logger.info("[Epoch %d/%d] bpn_weight=%.4f", ep, epochs, w_bpn)

        tr_loss, tr_mape = run_epoch(
            tr_idx,
            train=True,
            bpn_weight=w_bpn,
            design_thresholds=train_design_thresholds,
        )

        if va_idx:
            va_loss, va_mape = run_epoch(
                va_idx,
                train=False,
                bpn_weight=w_bpn,
                design_thresholds=val_design_thresholds,
            )
            if va_loss < best_val:
                best_val = va_loss
                state = {"bpn": bpn.state_dict(), "cnn": cnn.state_dict(), "head": head.state_dict()}
                if imp_head:
                    state["imp_head"] = imp_head.state_dict()
                if ep_emb:
                    state["ep_emb"] = ep_emb.state_dict()
                torch.save(state, os.path.join(save_dir, "best.pt"))
        else:
            logger.info("[Epoch %d/%d] train_loss=%.4f mape=%.4f", ep, epochs, tr_loss, tr_mape)

    # ------------------------------------------------------------------
    # 可视化导出逻辑
    # ------------------------------------------------------------------
    vis_cfg = cfg.get("vis", {}) or {}
    if bool(vis_cfg.get("dump_node_importance", False)):
        best_ckpt = os.path.join(save_dir, "best.pt")
        if os.path.exists(best_ckpt):
            try:
                state = torch.load(best_ckpt, map_location=device, weights_only=True)
                bpn.load_state_dict(state["bpn"])
                cnn.load_state_dict(state["cnn"])
                head.load_state_dict(state["head"])
                if imp_head and "imp_head" in state:
                    imp_head.load_state_dict(state["imp_head"])
                if ep_emb and "ep_emb" in state:
                    ep_emb.load_state_dict(state["ep_emb"])
            except Exception:
                pass

        for ds in datasets:
            temp_teacher = {"map_by_name": ds.teacher_map} if ds.teacher_map else None
            dump_node_importance_for_dataset(
                npz_path=ds.npz_path,
                ds=ds,
                bpn=bpn,
                imp_head=imp_head,
                teacher=temp_teacher,
                device=device,
                vis_cfg=vis_cfg,
            )


def main():
    parser = argparse.ArgumentParser(description="Train CPL-based model with optional BPN auxiliary loss.")
    parser.add_argument("--config", type=str, default="configs/model.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    train_loop(cfg)


if __name__ == "__main__":
    main()