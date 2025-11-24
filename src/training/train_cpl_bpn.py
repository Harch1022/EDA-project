import argparse
import json
import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import dgl
import numpy as np
import torch
import yaml

from src.models.bpn_propagation import BPNPropagator

logger = logging.getLogger("train_cpl_bpn")


# --------------------
# 工具函数
# --------------------


def _setup_logging() -> None:
    if logger.handlers:
        # 已经配置过 logging（例如被 pytest / 其他脚本调用时）
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )


def _load_npz(path: str) -> Dict[str, Any]:
    """加载 npz 并转成普通 dict，方便访问。"""
    arr = np.load(path, allow_pickle=True)
    return {k: arr[k] for k in arr.files}


def _build_graph_from_edges(edges: np.ndarray, num_nodes: int) -> dgl.DGLGraph:
    """根据 edges (2, E) 和节点数构建 DGL 有向图。"""
    assert edges.shape[0] == 2, f"edges shape 应为 (2, E)，实际为 {edges.shape}"
    src = edges[0].astype(np.int64)
    dst = edges[1].astype(np.int64)

    src_t = torch.from_numpy(src)
    dst_t = torch.from_numpy(dst)

    g = dgl.graph((src_t, dst_t), num_nodes=int(num_nodes))
    # 可选：去重自环/多重边
    g = dgl.to_simple(g)
    return g


# --------------------
# 主逻辑
# --------------------


def _get_dataset_path(cfg: Dict[str, Any]) -> str:
    """从 config 里解析出 dataset npz 路径。"""
    dataset_path = cfg.get("dataset_path")
    if dataset_path:
        return dataset_path

    # 兼容老的 design/project_root 写法
    project_root = cfg.get("project_root", ".")
    design = cfg.get("design", "my_design")
    dataset_path = os.path.join(project_root, "data", "processed", f"{design}.npz")
    return dataset_path


def _get_bpn_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """读取 bpn 子配置，附带默认值。"""
    bpn_cfg = dict(cfg.get("bpn", {}))

    def _pop(k, default):
        return bpn_cfg.pop(k, default)

    out = {
        "steps_fwd": int(_pop("steps_fwd", 6)),
        "steps_bwd": int(_pop("steps_bwd", 6)),
        "decay_fwd": float(_pop("decay_fwd", 0.9)),
        "decay_bwd": float(_pop("decay_bwd", 0.9)),
        "combine": _pop("combine", "hadamard"),
        "out_norm": _pop("out_norm", "l1"),
        "degree_norm": _pop("degree_norm", "out"),
        "include_seed": bool(_pop("include_seed", True)),
        "topk": int(_pop("topk", 5)),
        "max_endpoints": int(_pop("max_endpoints", 10)),
        "save_maps": bool(_pop("save_maps", True)),
    }
    return out


def run_bpn_analysis(cfg: Dict[str, Any]) -> None:
    dataset_path = _get_dataset_path(cfg)
    _setup_logging()

    logger.info("Using dataset: %s", dataset_path)
    data = _load_npz(dataset_path)

    node_features = data["node_features"]
    edges = data["edges"]
    endpoints_raw = data["endpoints"]
    cpl_indices = data["cpl_indices"]
    name_to_idx_arr = data["name_to_idx"]

    num_nodes = int(node_features.shape[0])
    g = _build_graph_from_edges(edges, num_nodes)
    logger.info("Graph built: num_nodes=%d, num_edges=%d", num_nodes, edges.shape[1])

    # name_to_idx: {node_name(str): node_idx(int)}
    name_to_idx: Dict[str, int] = dict(name_to_idx_arr.tolist())
    idx_to_name: Dict[int, str] = {
        int(idx): str(name) for name, idx in name_to_idx.items()
    }

    # endpoints 按照 npz 里原样使用，不再 strip 下划线
    endpoints: List[str] = [str(e) for e in endpoints_raw]
    assert len(endpoints) == len(
        cpl_indices
    ), "endpoints 和 cpl_indices 长度不一致"

    bpn_cfg = _get_bpn_config(cfg)
    bpn = BPNPropagator(
        steps_fwd=bpn_cfg["steps_fwd"],
        steps_bwd=bpn_cfg["steps_bwd"],
        decay_fwd=bpn_cfg["decay_fwd"],
        decay_bwd=bpn_cfg["decay_bwd"],
        combine=bpn_cfg["combine"],
        out_norm=bpn_cfg["out_norm"],
        degree_norm=bpn_cfg["degree_norm"],
        include_seed=bpn_cfg["include_seed"],
    )

    all_importance: List[torch.Tensor] = []
    endpoint_nodes: List[Optional[int]] = []

    topk = bpn_cfg["topk"]
    max_eps_log = bpn_cfg["max_endpoints"]

    for i, ep_name in enumerate(endpoints):
        # 1) 解析 CPL 起点（已经是节点索引）
        raw_cpl = cpl_indices[i]
        start_nodes = [
            int(int(n))
            for n in np.atleast_1d(raw_cpl)
            if 0 <= int(n) < num_nodes
        ]

        # 2) endpoint 映射到图节点索引：使用原始名字（带下划线）
        ep_idx = name_to_idx.get(ep_name, None)
        ep_idx_int: Optional[int] = int(ep_idx) if ep_idx is not None else None
        endpoint_nodes.append(ep_idx_int)

        # 3) 调 BPN
        imp = bpn.compute_endpoint_importance(
            g,
            start_nodes=start_nodes,
            endpoint_node=ep_idx_int,
        )
        all_importance.append(imp.unsqueeze(0))

        # 4) 日志：只打印前若干 endpoint 的 TopK
        if i < max_eps_log:
            imp_np = imp.detach().cpu().numpy()
            if topk > 0:
                topk_idx = np.argsort(-imp_np)[:topk]
                topk_list = [
                    [idx_to_name.get(int(j), str(j)), float(imp_np[j])]
                    for j in topk_idx
                ]
            else:
                topk_list = []

            record = {
                "endpoint": ep_name,  # 原始名字，如 "_246_"
                "endpoint_node": idx_to_name.get(ep_idx_int, "N/A")
                if ep_idx_int is not None
                else "N/A",
                "num_cpl_starts": len(start_nodes),
                "topk": topk_list,
                "sum": float(imp_np.sum()),
                "max": float(imp_np.max()),
            }
            logger.info("[BPN] EP[%d] %s", i, json.dumps(record, ensure_ascii=False))

    # 汇总并保存
    importance_tensor = torch.cat(all_importance, dim=0)
    importance = importance_tensor.detach().cpu().numpy()

    logger.info(
        "BPN importance shape: %s", tuple(importance.shape)
    )  # (num_endpoints, num_nodes)

    # 输出文件名：与原 npz 同名，加 _bpn_importance 后缀
    base = os.path.splitext(os.path.basename(dataset_path))[0]
    out_dir = os.path.dirname(dataset_path)
    save_path = os.path.join(out_dir, f"{base}_bpn_importance.npz")

    if bpn_cfg["save_maps"]:
        np.savez_compressed(
            save_path,
            endpoints=np.array(endpoints, dtype=object),
            endpoint_nodes=np.array(endpoint_nodes, dtype=object),
            importance=importance,
        )
        logger.info("Saved BPN importance maps to: %s", save_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BPN analysis on CPL endpoints.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (e.g., configs/model.yaml).",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    run_bpn_analysis(cfg)


if __name__ == "__main__":
    main()