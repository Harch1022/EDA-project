from __future__ import annotations
import argparse
import json
import logging
import os
import random
from typing import Dict, Tuple, List

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from src.models.bpn import BPN
from src.models.cnn import PhysCNN
from src.models.fusion import FusionRegressor
from src.models.loss_cpl import CplLoss
from src.training.metrics import mape

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
        self.endpoints = self.data["endpoints"]
        self.y_arrival = self.data["y_arrival"]
        self.cpl_indices = self.data["cpl_indices"]

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
        y = float(self.y_arrival[idx])
        ci = list(self.cpl_indices[idx])
        return idx, y, ci


def train_loop(cfg: Dict[str, object]):
    device = torch.device(cfg.get("device", "cpu"))
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    # ------------------------------------------------------------------
    # 加载数据集
    # ------------------------------------------------------------------
    data_cfg = cfg.get("data", {})
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
    # 构建模型
    # ------------------------------------------------------------------
    gnn_hidden = int(cfg["model"].get("gnn_hidden", 64))
    gnn_layers = int(cfg["model"].get("gnn_layers", 3))
    cnn_channels = cfg["model"].get("cnn_channels", [16, 32])
    fusion_hidden = int(cfg["model"].get("fusion_hidden", 64))

    bpn = BPN(d_in=d_node, hidden=gnn_hidden, layers=gnn_layers).to(device)
    cnn = PhysCNN(channels=tuple(cnn_channels), out_dim=gnn_hidden).to(device)
    head = FusionRegressor(d_gnn=gnn_hidden, d_cnn=gnn_hidden, hidden=fusion_hidden).to(device)

    loss_fn = CplLoss(
        mse_weight=float(cfg["loss"].get("mse_weight", 1.0)),
        cpl_weight=float(cfg["loss"].get("cpl_weight", 0.1)),
    )

    params = list(bpn.parameters()) + list(cnn.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(
        params,
        lr=float(cfg["train"].get("lr", 1e-3)),
        weight_decay=float(cfg["train"].get("weight_decay", 1e-4)),
    )

    # ------------------------------------------------------------------
    # 训练 / 验证 划分（对小数据集更鲁棒）
    # ------------------------------------------------------------------
    train_split = float(data_cfg.get("train_split", 0.8))
    if not (0.0 < train_split <= 1.0):
        raise ValueError(f"train_split must be in (0, 1], got {train_split}")

    indices = list(range(n))
    random.shuffle(indices)

    # 使用 round 而不是直接 int，避免 n=1 时 0.8 → 0 的问题，
    # 并且保证 n > 0 时至少有 1 个训练样本
    n_train = int(round(n * train_split))
    if n_train == 0 and n > 0:
        n_train = 1
    if n_train > n:
        n_train = n

    tr_idx = indices[:n_train]
    va_idx = indices[n_train:]  # 剩余样本作为验证集（可能为空）

    logger.info("Split: %d train endpoints, %d val endpoints.", len(tr_idx), len(va_idx))

    # ------------------------------------------------------------------
    # 训练 / 验证 epoch 循环
    # ------------------------------------------------------------------
    def run_epoch(idxs: List[int], train: bool = True) -> Tuple[float, float]:
        # 如果没有样本，避免对空数组算 mape 产生 warning
        if not idxs:
            return 0.0, float("nan")

        y_true_all: List[float] = []
        y_pred_all: List[float] = []
        total_loss = 0.0

        bpn.train(train)
        cnn.train(train)
        head.train(train)

        # 图和物理 map 是设计级别的，共享一份即可
        g = ds.g.to(device)
        x = ds.x.to(device)
        maps = ds.maps_t.to(device)

        for idx in idxs:
            _, y, ci = ds[idx]

            # 前向
            gnn_emb = bpn(g, x).unsqueeze(0)   # [1, d]
            cnn_emb = cnn(maps)                # [1, d]
            y_pred = head(gnn_emb, cnn_emb)    # [1]

            y_t = torch.tensor([y], dtype=torch.float32, device=device)

            loss = loss_fn(y_pred, y_t, gnn_emb, cpl_indices=[ci])

            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()

            total_loss += float(loss.detach().cpu())
            y_true_all.append(y)
            y_pred_all.append(float(y_pred.detach().cpu()))

        y_true_np = np.array(y_true_all, dtype=np.float32)
        y_pred_np = np.array(y_pred_all, dtype=np.float32)
        return total_loss / max(len(idxs), 1), float(mape(y_true_np, y_pred_np))

    epochs = int(cfg["train"].get("epochs", 3))
    best_val = float("inf")
    save_dir = os.path.dirname(npz_path)
    os.makedirs(save_dir, exist_ok=True)

    for ep in range(1, epochs + 1):
        tr_loss, tr_mape = run_epoch(tr_idx, train=True)

        if va_idx:
            va_loss, va_mape = run_epoch(va_idx, train=False)
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
                torch.save(
                    {
                        "bpn": bpn.state_dict(),
                        "cnn": cnn.state_dict(),
                        "head": head.state_dict(),
                    },
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
    }
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Training done. Metrics saved to %s", os.path.join(save_dir, "metrics.json"))


def main():
    parser = argparse.ArgumentParser(description="Train CPL-based model.")
    parser.add_argument("--config", type=str, default="configs/model.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    train_loop(cfg)


if __name__ == "__main__":
    main()