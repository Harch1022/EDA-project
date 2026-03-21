from __future__ import annotations
import argparse
import json
import logging
import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import yaml
from tqdm import tqdm

from src.eda_parser.netlist_parser import parse_gate_level_verilog
from src.eda_parser.def_parser import parse_def
from src.eda_parser.timing_parser import parse_report_checks, group_by_endpoint
from src.eda_parser.liberty_parser import (
    parse_liberty_pin_directions,
    build_cell_output_index,
)
from src.features.graph_builder import (
    build_graph,
    normalized_decay_manhattan,
)
from src.features.cnn_maps import build_physical_maps
from src.labels.cpl import compute_cpl_labels
from src.labels.mapping import name_to_node_idx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("build_dataset")


def load_metadata(project_root: str, design: str) -> Dict:
    out_dir = os.path.join(project_root, "data", "raw_eda", design)
    meta_path = os.path.join(out_dir, "metadata.json")
    if not os.path.exists(meta_path):
        logger.warning(f"metadata.json not found for {design}, falling back to heuristics.")
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_vocab(all_cell_types: List[str]) -> Dict[str, int]:
    uniq = sorted(set(all_cell_types))
    return {c: i for i, c in enumerate(uniq)}


def onehot_types(
    instances: Dict[str, Dict[str, object]], vocab: Dict[str, int]
) -> np.ndarray:
    N = len(instances)
    D = len(vocab)
    m = np.zeros((N, D), dtype=np.float32)
    for i, name in enumerate(instances.keys()):
        c = str(instances[name].get("type", "UNK"))
        if c in vocab:
            m[i, vocab[c]] = 1.0
    return m


def _to_object_array(items: List[object]) -> np.ndarray:
    """
    显式构造 1D object array，避免在所有子列表长度刚好一样时，
    numpy 意外把它堆成规则 2D 数组。
    """
    arr = np.empty(len(items), dtype=object)
    for i, item in enumerate(items):
        arr[i] = item
    return arr


def _load_cell_outputs_from_liberty(meta: Dict) -> Optional[Dict[str, set]]:
    lib_path = None
    try:
        maybe = meta.get("lib", {})
        lib_path = maybe.get("lib", None) or maybe.get("liberty", None)
    except Exception:
        lib_path = None

    if lib_path and os.path.exists(lib_path):
        try:
            pin_dirs = parse_liberty_pin_directions(lib_path)
            return build_cell_output_index(pin_dirs)
        except Exception as e:
            logger.warning(
                "Liberty parsing failed: %s. Will fallback to heuristic.", e
            )
            return None
    return None


def process_one_design(
    project_root: str,
    design: str,
    grid_size: int,
    near_cfg: Dict[str, object],
    save_dir: str,
    vocab: Dict[str, int] | None,
) -> Tuple[Dict[str, int], Dict[str, object]]:
    out_dir = os.path.join(project_root, "data", "raw_eda", design)
    synth_v = os.path.join(out_dir, f"{design}.synth.v")
    pre_def = os.path.join(out_dir, f"{design}.pre_route.def")
    post_def = os.path.join(out_dir, f"{design}.post_route.def")
    pre_timing = os.path.join(out_dir, f"{design}.pre_route_timing.rpt")
    post_timing = os.path.join(out_dir, f"{design}.post_route_timing.rpt")
    meta = load_metadata(project_root, design)

    # 1) 解析网表
    instances, net2pins, ports = parse_gate_level_verilog(synth_v)

    # 2) 先解析 DEF：后面建图时要用坐标计算 edge weight
    def_path = post_def if os.path.exists(post_def) else pre_def
    comps, die_area, pins_xy = parse_def(def_path)
    if not die_area:
        logger.warning("No DIEAREA; fallback to synthetic die area.")
        die_area = ((0, 0), (10000, 10000))

    # 3) 构建图，并在图边上写入归一化 Manhattan distance
    cell_outputs = _load_cell_outputs_from_liberty(meta)
    g, name_to_idx, base_node_feats = build_graph(
        instances,
        net2pins,
        ports,
        cell_outputs=cell_outputs,
        comps=comps,
        die_area=die_area,
    )

    # 4) 节点特征：基础特征 + cell type one-hot
    cell_types = [str(instances[n]["type"]) for n in instances.keys()]
    if vocab is None:
        vocab = build_vocab(cell_types)
    type_onehot = onehot_types(instances, vocab)
    node_features = np.concatenate([base_node_feats, type_onehot], axis=1)

    # 5) 物理 CNN 特征图（和 edge weight 并不冲突，继续保留）
    maps = build_physical_maps(
        comps,
        die_area,
        pins_xy,
        grid_size=grid_size,
        net_to_pins=net2pins,
        prefer_hpwl_rudy=True,
    )

    # 6) 解析 timing 报告，构建 near-critical CPL 标签
    tpath = post_timing if os.path.exists(post_timing) else pre_timing
    paths = parse_report_checks(tpath)
    by_ep = group_by_endpoint(paths)

    # near-critical 配置（带默认值）
    mode = str(near_cfg.get("mode", "delta_abs"))
    delta_ns = float(near_cfg.get("delta_ns", 0.02))
    quantile = float(near_cfg.get("quantile", 0.10))
    ratio_metric = str(near_cfg.get("ratio_metric", "arrival"))
    top_ratio = float(near_cfg.get("top_ratio", 1.0))
    inner_mode = str(near_cfg.get("inner_mode", "delta_abs"))
    keep_non_sel = bool(near_cfg.get("keep_non_selected", False))

    near = compute_cpl_labels(
        by_ep,
        mode=mode,
        delta_ns=delta_ns,
        quantile=quantile,
        ratio_metric=ratio_metric,
        top_ratio=top_ratio,
        inner_mode=inner_mode,
        keep_non_selected=keep_non_sel,
    )

    endpoints: List[str] = []
    y_arrival: List[float] = []
    cpl_indices: List[List[int]] = []
    nuiat_times: List[List[float]] = []  # 改动点：与 cpl_indices 一一对齐保存 NUIAT 时间

    total_eps = len(by_ep)

    # 7) 只对有 near-critical CPL 的 endpoint 组装数据
    for ep, lst in near.items():
        if not lst:
            continue

        # 找一个 arrival 时间；如果缺失，用最差 slack 的负值兜底
        arr = None
        for p in by_ep.get(ep, []):
            if p.get("arrival") is not None:
                arr = float(p["arrival"])
                break
        if arr is None:
            slks = [
                p.get("slack", None)
                for p in by_ep.get(ep, [])
                if p.get("slack", None) is not None
            ]
            arr = -float(min(slks)) if slks else 0.0

        # -------------------------
        # 改动点 2：逐条 path 映射，确保 idx 和 nuiat_times 严格对齐
        # 注意：
        #   - 这里仍然复用原有 name_to_node_idx 映射逻辑，避免破坏你现有命名兼容性
        #   - 优先使用 start_arrival（真实 startpoint arrival）
        #   - 若报告里没解析到，再退回 path arrival，保证可运行
        # -------------------------
        idxs: List[int] = []
        times_this_ep: List[float] = []

        for p in lst:
            sp_name = str(p.get("startpoint", "")).strip()
            if not sp_name:
                continue

            mapped = name_to_node_idx([sp_name], name_to_idx)
            if mapped is None or len(mapped) == 0:
                continue

            idxs.append(int(mapped[0]))

            raw_t = p.get("start_arrival", None)
            if raw_t is None:
                raw_t = p.get("arrival", None)

            try:
                t_val = float(raw_t) if raw_t is not None else 0.0
            except Exception:
                t_val = 0.0

            times_this_ep.append(float(t_val))

        if not idxs:
            continue

        endpoints.append(ep)
        y_arrival.append(arr)
        cpl_indices.append(idxs)
        nuiat_times.append(times_this_ep)

    selected_eps = len(endpoints)

    import dgl
    import torch

    # 8) 为 synthetic CPL edges 补边，并同时补 edge weight
    #    注意：这一步会引入标签侧信息；若你后续做严格泛化评估，请确认不会造成 leakage。
    if endpoints:
        num_nodes = g.num_nodes()
        idx_to_name = {i: n for n, i in name_to_idx.items()}

        extra_src: List[int] = []
        extra_dst: List[int] = []
        extra_w: List[float] = []

        for ep_name, start_idxs in zip(endpoints, cpl_indices):
            ep_idx = name_to_idx.get(ep_name, None)
            if ep_idx is None:
                logger.warning(
                    "Endpoint %s not found in name_to_idx; skip CPL edges for it.",
                    ep_name,
                )
                continue

            for s in start_idxs:
                si = int(s)

                if si < 0 or si >= num_nodes:
                    logger.warning(
                        "CPL start idx %d for endpoint %s out of range [0,%d); skip",
                        si,
                        ep_name,
                        num_nodes,
                    )
                    continue

                if si == ep_idx:
                    continue

                src_name = idx_to_name.get(si, None)
                if src_name is None:
                    logger.warning(
                        "Source idx %d not found in idx_to_name for endpoint %s; skip",
                        si,
                        ep_name,
                    )
                    continue

                w = normalized_decay_manhattan(
                    src_name,
                    ep_name,
                    comps=comps,
                    die_area=die_area,
                    eps=1e-6,
                )

                extra_src.append(si)
                extra_dst.append(ep_idx)
                extra_w.append(w)

        if extra_src:
            src_tensor = torch.tensor(extra_src, dtype=torch.int64)
            dst_tensor = torch.tensor(extra_dst, dtype=torch.int64)
            w_tensor = torch.tensor(extra_w, dtype=torch.float32)

            g = dgl.add_edges(
                g,
                src_tensor,
                dst_tensor,
                data={"weight": w_tensor},
            )
            logger.info(
                "Added %d synthetic CPL edges (start -> endpoint) for design %s",
                len(extra_src),
                design,
            )
        else:
            logger.info(
                "No synthetic CPL edges added for design %s (nothing to connect).",
                design,
            )

    # 9) 导出最终边列表和边权
    edges = np.stack(
        [g.edges()[0].cpu().numpy(), g.edges()[1].cpu().numpy()],
        axis=0,
    )
    edge_weight = g.edata["weight"].cpu().numpy().astype(np.float32)

    # 10) 保存为 npz
    os.makedirs(save_dir, exist_ok=True)
    out_npz = os.path.join(save_dir, f"{design}.npz")
    np.savez_compressed(
        out_npz,
        node_features=node_features,
        edges=edges,
        edge_weight=edge_weight,
        cell_density_map=maps["cell_density_map"],
        rudy_map=maps["rudy_map"],
        macro_mask_map=maps["macro_mask_map"],
        endpoints=np.array(endpoints, dtype=object),
        y_arrival=np.array(y_arrival, dtype=np.float32),
        cpl_indices=_to_object_array(
            [np.asarray(x, dtype=np.int64) for x in cpl_indices]
        ),
        # 改动点：新增保存 NUIAT 时间，并与 cpl_indices[i][k] 严格对齐
        nuiat_times=_to_object_array(
            [np.asarray(x, dtype=np.float32) for x in nuiat_times]
        ),
        name_to_idx=np.array(list(name_to_idx.items()), dtype=object),
        vocab=np.array(list(vocab.items()), dtype=object),
    )
    logger.info("Saved dataset: %s", out_npz)

    # 11) 更新 vocab.json（跨设计共享 vocab）
    vocab_json = os.path.join(save_dir, "vocab.json")
    if os.path.exists(vocab_json):
        old = dict(json.load(open(vocab_json, "r", encoding="utf-8")))
        merged = dict(
            sorted(
                set(list(old.items()) + list(vocab.items())),
                key=lambda x: x[0],
            )
        )
        with open(vocab_json, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2)
    else:
        with open(vocab_json, "w", encoding="utf-8") as f:
            json.dump(vocab, f, indent=2)

    # 12) 返回统计信息
    stats_this = {
        "total_endpoints": int(total_eps),
        "selected_endpoints": int(selected_eps),
        "near_mode": mode,
        "near_params": {
            "delta_ns": delta_ns,
            "quantile": quantile,
            "ratio_metric": ratio_metric,
            "top_ratio": top_ratio,
            "inner_mode": inner_mode,
            "keep_non_selected": keep_non_sel,
        },
    }
    return vocab, {design: stats_this}


def main():
    parser = argparse.ArgumentParser(
        description="Build dataset from EDA artifacts."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dataset.yaml",
        help="dataset config yaml",
    )
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, "r"))
    project_root = os.path.abspath(cfg.get("project_root", "."))

    benches: List[str] = cfg.get("benches", [])
    if not benches:
        raise ValueError("No benches specified in dataset.yaml")
    save_dir = os.path.join(project_root, cfg.get("save_dir", "data/processed"))
    grid_size = int(cfg.get("grid_size", 64))
    near_cfg = cfg.get(
        "near_critical", {"mode": "delta_abs", "delta_ns": 0.02}
    )

    vocab: Dict[str, int] | None = None
    stats: Dict[str, object] = {}
    for d in tqdm(benches, desc="build"):
        vocab, s = process_one_design(
            project_root, d, grid_size, near_cfg, save_dir, vocab
        )
        stats.update(s)

    os.makedirs(save_dir, exist_ok=True)
    with open(
        os.path.join(save_dir, "build_stats.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(stats, f, indent=2)
    logger.info("Done. Stats: %s", stats)


if __name__ == "__main__":
    main()