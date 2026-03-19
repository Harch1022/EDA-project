from __future__ import annotations
import argparse
import json
import logging
import os
import random
import glob  # <--- 新增
from typing import Dict, Tuple, List, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import yaml
from torch.utils.data import Dataset, ConcatDataset  # <--- 新增 ConcatDataset

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
        self.npz_path = npz_path  # <--- 新增：保存路径，方便后续索骥
        self.teacher_map = None   # <--- 新增：预留存放各自 BPN 老师分布的位置
        self.p_teacher_t = {}     # <--- 新增：预留存放张量化老师分布的位置

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
        self.y_arrival = self.data["y_arrival"]          # [E]
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
        # <--- 核心修改：把 self 也返回出去，充当“移动的图结构仓库”
        return ep_id, y, ci, ep_name, self


def _infer_bpn_importance_path(dataset_npz: str) -> str:
    base = os.path.splitext(os.path.basename(dataset_npz))[0]
    out_dir = os.path.dirname(dataset_npz)
    return os.path.join(out_dir, f"{base}_bpn_importance.npz")


def _load_bpn_importance(importance_npz: str, num_nodes: int) -> Optional[Dict[str, Any]]:
    if not os.path.exists(importance_npz):
        logger.info("BPN importance file not found: %s", importance_npz)
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


def _compute_bpn_loss(p_model: torch.Tensor, p_teacher_np: np.ndarray, loss_type: str = "kl") -> torch.Tensor:
    device = p_model.device
    p_teacher = torch.from_numpy(p_teacher_np).to(device=device, dtype=p_model.dtype)
    eps = 1e-12
    if loss_type == "mse":
        return F.mse_loss(p_model, p_teacher)
    if loss_type == "l1":
        return torch.mean(torch.abs(p_model - p_teacher))
    p_t = torch.clamp(p_teacher, min=eps)
    p_m = torch.clamp(p_model, min=eps)
    kl = torch.sum(p_t * (torch.log(p_t) - torch.log(p_m)))
    return kl


def dump_node_importance_for_dataset(
    npz_path: str,
    ds: EndpointDataset,
    bpn: BPN,
    imp_head: Optional[NodeImportanceHead],
    teacher: Optional[Dict[str, Any]],
    device: torch.device,
    vis_cfg: Dict[str, Any],
) -> None:
    # (此函数无需大幅修改，维持原逻辑即可，外部会循环调用它)
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

    # ------------------------------------------------------------------
    # 【多设计加载核心改造】
    # ------------------------------------------------------------------
    data_cfg = cfg.get("data", {}) or {}
    base_npz_path = data_cfg.get("dataset_npz")
    if not base_npz_path:
        raise ValueError("Please provide dataset_npz in config to infer directory.")
    
    data_dir = os.path.dirname(base_npz_path)
    # 扫描目录下所有的 .npz，但排除 importance 和 node_importance 文件
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
    # 提取特征维度 (假设所有设计的特征维度一致)
    d_node = datasets[0].node_features.shape[1] 
    logger.info("Concatenated %d endpoints from %d designs.", n, len(datasets))

    # ------------------------------------------------------------------
    # 端点条件化配置
    # ------------------------------------------------------------------
    model_cfg = cfg.get("model", {}) or {}
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
            
            if teacher is None:
                logger.warning("Teacher not found for %s. Fallback to uniform.", ds.npz_path)
                ds.teacher_map = {str(ep): np.ones(N_ds_nodes, dtype=np.float32) / N_ds_nodes for ep in ds.endpoints}
            else:
                has_any_teacher = True
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

            # 转为 Torch 张量存储在 dataset 实例中，加速前向传播
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
    
    # 全局总的端点数量 (用于 ID 嵌入)
    num_endpoints_global = n  

    if use_ep_condition and ep_mode in ("id", "hybrid"):
        ep_emb = EndpointEmbedding(num_endpoints=num_endpoints_global, d_ep=d_ep_id).to(device)
        ep_dropout = nn.Dropout(p=drop_ep).to(device)

    if use_ep_condition:
        d_node_emb = gnn_hidden
        if ep_mode == "id":
            d_ep_in = d_ep_id
        elif ep_mode == "teacher_pool":
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

    loss_fn = CplLoss(
        mse_weight=float(cfg["loss"].get("mse_weight", 1.0)),
        cpl_weight=float(cfg["loss"].get("cpl_weight", 0.1)),
    )

    param_groups: List[Dict[str, Any]] = [{
        "params": list(bpn.parameters()) + list(cnn.parameters()) + list(head.parameters()),
        "weight_decay": float(cfg["train"].get("weight_decay", 1.0e-4)),
    }]
    if imp_head is not None:
        param_groups[0]["params"] += list(imp_head.parameters())
    if ep_emb is not None:
        param_groups.append({"params": list(ep_emb.parameters()), "weight_decay": 1.0e-3})

    opt = torch.optim.Adam(param_groups, lr=float(cfg["train"].get("lr", 1e-3)))

    train_split = float(data_cfg.get("train_split", 0.8))
    indices = list(range(n))
    random.shuffle(indices)
    n_train = int(round(n * train_split))
    if n_train == 0 and n > 0: n_train = 1
    if n_train > n: n_train = n

    tr_idx = indices[:n_train]
    va_idx = indices[n_train:]

    limit_train = int(cfg.get("debug", {}).get("limit_train_samples", 0))
    if limit_train > 0:
        tr_idx = tr_idx[: min(limit_train, len(tr_idx))]

    warmup_ep = int(bpn_cfg.get("warmup_epochs", 0))
    ramp_ep = int(bpn_cfg.get("ramp_epochs", 0))

    def get_bpn_weight(epoch: int) -> float:
        if not (use_bpn_in_loss and has_any_teacher): return 0.0
        if warmup_ep > 0 and epoch <= warmup_ep: return 0.0
        if ramp_ep > 0 and epoch <= warmup_ep + ramp_ep:
            return base_bpn_loss_weight * (epoch - warmup_ep) / max(ramp_ep, 1)
        return base_bpn_loss_weight

    # ------------------------------------------------------------------
    # 动态前向传播改造
    # ------------------------------------------------------------------
    def run_epoch(idxs: List[int], train: bool = True, bpn_weight: float = 0.0) -> Tuple[float, float]:
        if not idxs:
            return 0.0, float("nan")

        y_true_all, y_pred_all = [], []
        total_loss_sum, main_loss_sum, bpn_loss_sum, count_bpn = 0.0, 0.0, 0.0, 0
        
        bpn.train(train); cnn.train(train); head.train(train)
        if imp_head: imp_head.train(train)
        if ep_emb: ep_emb.train(train)

        for global_idx in idxs:
            # 动态接住当前的数据集对象 current_ds 
            local_ep_id, y, ci, ep_name, current_ds = concat_ds[global_idx]
            ep_name_str = str(ep_name)

            # 动态加载对应设计的图和特征
            g = current_ds.g.to(device)
            x = current_ds.x.to(device)
            maps = current_ds.maps_t.to(device)

            need_node_emb = ((use_ep_condition and ep_mode in ("teacher_pool", "hybrid"))
                             or (imp_head is not None and current_ds.teacher_map is not None and bpn_weight > 0.0))

            if need_node_emb:
                g_emb, node_emb = bpn(g, x, return_node_emb=True)
            else:
                g_emb = bpn(g, x)
                node_emb = None

            gnn_emb = g_emb.unsqueeze(0)
            cnn_emb = cnn(maps)

            if use_ep_condition:
                if ep_mode == "teacher_pool":
                    pt = current_ds.p_teacher_t[ep_name_str].to(device)
                    z_ep = torch.matmul(pt.unsqueeze(0), node_emb)

                elif ep_mode == "id":
                    # 使用 global_idx 避免多个设计 id 冲撞
                    ep_idx_t = torch.tensor([int(global_idx)], dtype=torch.long, device=device)
                    z_ep = ep_dropout(ep_emb(ep_idx_t))

                elif ep_mode == "hybrid":
                    pt = current_ds.p_teacher_t[ep_name_str].to(device)
                    z_struct = torch.matmul(pt.unsqueeze(0), node_emb)
                    ep_idx_t = torch.tensor([int(global_idx)], dtype=torch.long, device=device)
                    z_id = ep_dropout(ep_emb(ep_idx_t))
                    z_ep = torch.cat([z_id, z_struct], dim=-1)

                y_pred = head(gnn_emb, cnn_emb, z_ep)
            else:
                y_pred = head(gnn_emb, cnn_emb)

            y_t = torch.tensor([y], dtype=torch.float32, device=device)
            main_loss = loss_fn(y_pred, y_t, gnn_emb, cpl_indices=[ci])
            loss = main_loss

            # 从 current_ds 获取 BPN 老师分布
            if bpn_weight > 0.0 and imp_head is not None and node_emb is not None and current_ds.teacher_map is not None:
                p_teacher_np = current_ds.teacher_map.get(ep_name_str, None)
                if p_teacher_np is not None:
                    p_model = imp_head(node_emb)
                    bpn_loss = _compute_bpn_loss(p_model, p_teacher_np, loss_type=bpn_loss_type)
                    loss = loss + bpn_weight * bpn_loss
                    bpn_loss_sum += float(bpn_loss.detach().cpu())
                    count_bpn += 1

            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()

            total_loss_sum += float(loss.detach().cpu())
            main_loss_sum += float(main_loss.detach().cpu())
            y_true_all.append(y)
            y_pred_all.append(float(y_pred.detach().cpu()))

        y_true_np = np.array(y_true_all, dtype=np.float32)
        y_pred_np = np.array(y_pred_all, dtype=np.float32)

        if y_true_np.size > 0:
            avg_total = total_loss_sum / max(len(idxs), 1)
            avg_main = main_loss_sum / max(len(idxs), 1)
            avg_bpn = (bpn_loss_sum / max(count_bpn, 1)) if count_bpn > 0 else 0.0
            r2_val = float(r2_score(y_true_np, y_pred_np))
            logger.info("Split=%s: total=%.4f main=%.4f bpn=%.4f (count=%d) R2=%.4f",
                        "train" if train else "val", avg_total, avg_main, avg_bpn, count_bpn, r2_val)

        return total_loss_sum / max(len(idxs), 1), float(mape(y_true_np, y_pred_np))

    epochs = int(cfg["train"].get("epochs", 3))
    best_val = float("inf")
    save_dir = os.path.dirname(base_npz_path)
    os.makedirs(save_dir, exist_ok=True)

    for ep in range(1, epochs + 1):
        w_bpn = get_bpn_weight(ep)
        tr_loss, tr_mape = run_epoch(tr_idx, train=True, bpn_weight=w_bpn)
        if va_idx:
            va_loss, va_mape = run_epoch(va_idx, train=False, bpn_weight=w_bpn)
            if va_loss < best_val:
                best_val = va_loss
                state = {"bpn": bpn.state_dict(), "cnn": cnn.state_dict(), "head": head.state_dict()}
                if imp_head: state["imp_head"] = imp_head.state_dict()
                if ep_emb: state["ep_emb"] = ep_emb.state_dict()
                torch.save(state, os.path.join(save_dir, "best.pt"))
        else:
            logger.info("[Epoch %d/%d] train_loss=%.4f mape=%.4f", ep, epochs, tr_loss, tr_mape)

    # ------------------------------------------------------------------
    # 修复可视化导出逻辑 (遍历每个设计)
    # ------------------------------------------------------------------
    vis_cfg = cfg.get("vis", {}) or {}
    if bool(vis_cfg.get("dump_node_importance", False)):
        best_ckpt = os.path.join(save_dir, "best.pt")
        if os.path.exists(best_ckpt):
            try:
                state = torch.load(best_ckpt, map_location=device)
                bpn.load_state_dict(state["bpn"]); cnn.load_state_dict(state["cnn"]); head.load_state_dict(state["head"])
                if imp_head and "imp_head" in state: imp_head.load_state_dict(state["imp_head"])
                if ep_emb and "ep_emb" in state: ep_emb.load_state_dict(state["ep_emb"])
            except Exception as e:
                pass
        
        # 遍历 ConcatDataset 里的每一个子数据集进行导出
        for ds in datasets:
            # 临时重组 teacher 字典形式，以兼容 dump 接口
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