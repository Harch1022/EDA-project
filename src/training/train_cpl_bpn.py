import argparse
import json
import logging
import os
from collections import OrderedDict
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


def _coalesce_edges_with_weights(
    edges: np.ndarray,
    edge_weight: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将重复边合并：
      - 若有 edge_weight，则对重复边的权重求和
      - 若没有 edge_weight，则保持 to_simple 语义：重复边压成 1 条，权重设为 1
    """
    assert edges.shape[0] == 2, f"edges shape 应为 (2, E)，实际为 {edges.shape}"

    src = edges[0].astype(np.int64).reshape(-1)
    dst = edges[1].astype(np.int64).reshape(-1)

    if src.size == 0:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
        )

    if edge_weight is None:
        pairs = np.stack([src, dst], axis=1)
        uniq_pairs = np.unique(pairs, axis=0)
        uniq_src = uniq_pairs[:, 0].astype(np.int64)
        uniq_dst = uniq_pairs[:, 1].astype(np.int64)
        uniq_w = np.ones((uniq_pairs.shape[0],), dtype=np.float32)
        return uniq_src, uniq_dst, uniq_w

    w = np.asarray(edge_weight, dtype=np.float32).reshape(-1)
    assert (
        w.shape[0] == src.shape[0]
    ), f"edge_weight 长度应与边数一致，实际为 {w.shape[0]} vs {src.shape[0]}"

    order = np.lexsort((dst, src))
    src_s = src[order]
    dst_s = dst[order]
    w_s = w[order]

    new_group = np.ones(src_s.shape[0], dtype=bool)
    if src_s.shape[0] > 1:
        new_group[1:] = (src_s[1:] != src_s[:-1]) | (dst_s[1:] != dst_s[:-1])

    group_id = np.cumsum(new_group) - 1

    uniq_src = src_s[new_group]
    uniq_dst = dst_s[new_group]
    uniq_w = np.zeros((uniq_src.shape[0],), dtype=np.float32)
    np.add.at(uniq_w, group_id, w_s)

    return uniq_src, uniq_dst, uniq_w


def _build_graph_from_edges(
    edges: np.ndarray,
    num_nodes: int,
    edge_weight: Optional[np.ndarray] = None,
) -> dgl.DGLGraph:
    """
    根据 edges (2, E) 和节点数构建 DGL 有向图。
    改动点：
      1) 保留并挂载 edge_weight，避免把你已经做好的物理边权丢掉
      2) 对重复边做合并；若有权重则求和，若无权重则等价于 to_simple
    """
    try:
        src, dst, w = _coalesce_edges_with_weights(edges, edge_weight=edge_weight)
    except Exception as e:
        logger.warning("edge_weight 处理失败（%s），回退为无权图。", e)
        src, dst, w = _coalesce_edges_with_weights(edges, edge_weight=None)

    src_t = torch.from_numpy(src)
    dst_t = torch.from_numpy(dst)

    g = dgl.graph((src_t, dst_t), num_nodes=int(num_nodes))
    g.edata["weight"] = torch.from_numpy(w.astype(np.float32))
    return g


def _extract_valid_start_nodes(raw_cpl: Any, num_nodes: int) -> List[int]:
    start_nodes: List[int] = []
    for n in np.atleast_1d(raw_cpl):
        try:
            ni = int(n)
        except Exception:
            continue
        if 0 <= ni < num_nodes:
            start_nodes.append(ni)
    return start_nodes


def _coerce_time_list(raw_times: Any) -> List[float]:
    if raw_times is None:
        return []

    out: List[float] = []
    for x in np.atleast_1d(raw_times):
        try:
            out.append(float(x))
        except Exception:
            out.append(float("nan"))
    return out


def _stable_softmax(values: Sequence[float], temperature: float) -> np.ndarray:
    """
    对 startpoint arrival 做带温度的 softmax：
        weight_i = softmax(arrival_i / T)

    约定：
      - arrival 越大，seed 权重越大
      - temperature 越小，分布越尖锐
    """
    if temperature <= 0.0:
        raise ValueError(f"temperature 必须 > 0，实际为 {temperature}")

    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0,), dtype=np.float32)

    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        return np.full((arr.size,), 1.0 / float(arr.size), dtype=np.float32)

    fill_value = float(np.min(arr[finite_mask]))
    arr = np.where(finite_mask, arr, fill_value)

    z = arr / float(temperature)
    z = z - np.max(z)  # 数值稳定
    exp_z = np.exp(z)
    den = float(exp_z.sum())

    if den <= 0.0 or not np.isfinite(den):
        return np.full((arr.size,), 1.0 / float(arr.size), dtype=np.float32)

    return (exp_z / den).astype(np.float32)


def _prepare_seed_distribution(
    start_nodes: Sequence[int],
    raw_nuiat_times: Any,
    temperature: float,
    use_nuiat_softmax: bool = True,
) -> Tuple[List[int], Optional[torch.Tensor], str, List[Tuple[int, float, float]]]:
    """
    输入：
      - start_nodes：来自 cpl_indices 的节点索引列表
      - raw_nuiat_times：与 start_nodes 一一对应的 startpoint arrival

    输出：
      - effective_start_nodes：用于 BPN seed 的节点列表
      - seed_weights：若存在 NUIAT，则为 softmax 权重；否则为 None（表示保留原均匀分配）
      - seed_mode：日志用途
      - debug_info：[(node_idx, arrival_time, weight), ...]
    """
    start_nodes = [int(x) for x in start_nodes]
    if not start_nodes:
        return [], None, "empty", []

    if (not use_nuiat_softmax) or raw_nuiat_times is None:
        return list(start_nodes), None, "uniform", []

    times = _coerce_time_list(raw_nuiat_times)
    if len(times) != len(start_nodes):
        logger.warning(
            "nuiat_times 与 cpl_indices 长度不一致：len(times)=%d, len(starts)=%d；该 endpoint 回退为均匀 seed。",
            len(times),
            len(start_nodes),
        )
        return list(start_nodes), None, "uniform_len_mismatch", []

    # 先对 path-level 的 arrival 做 softmax
    path_weights = _stable_softmax(times, temperature)

    # 再按节点聚合，避免同一个 startpoint 重复出现时 seed 重复灌入
    node_to_weight = OrderedDict()
    node_to_time = OrderedDict()

    for node_idx, t, w in zip(start_nodes, times, path_weights.tolist()):
        node_to_weight[node_idx] = float(node_to_weight.get(node_idx, 0.0)) + float(w)

        prev_t = node_to_time.get(node_idx, None)
        if prev_t is None:
            node_to_time[node_idx] = float(t)
        else:
            if np.isfinite(t) and (not np.isfinite(prev_t) or t > prev_t):
                node_to_time[node_idx] = float(t)

    effective_nodes = list(node_to_weight.keys())
    effective_weights = np.array(
        [node_to_weight[n] for n in effective_nodes], dtype=np.float32
    )

    ws = float(effective_weights.sum())
    if ws <= 0.0 or not np.isfinite(ws):
        return list(start_nodes), None, "uniform_bad_softmax", []

    effective_weights /= ws

    debug_info = [
        (
            int(node_idx),
            float(node_to_time.get(node_idx, float("nan"))),
            float(weight),
        )
        for node_idx, weight in zip(effective_nodes, effective_weights.tolist())
    ]

    return (
        effective_nodes,
        torch.tensor(effective_weights, dtype=torch.float32),
        "nuiat_softmax",
        debug_info,
    )


def _seed_vector_from_nodes(
    num_nodes: int,
    start_nodes: Sequence[int],
    seed_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    把 start_nodes + optional seed_weights 转成节点级 seed 向量。
    若 seed_weights 为 None，则使用均匀分配。
    """
    vec = torch.zeros(int(num_nodes), dtype=torch.float32)

    if not start_nodes:
        return vec

    idx_t = torch.tensor(list(start_nodes), dtype=torch.long)

    if seed_weights is None:
        w_t = torch.full(
            (len(start_nodes),),
            1.0 / float(len(start_nodes)),
            dtype=torch.float32,
        )
    else:
        w_t = seed_weights.detach().cpu().float().reshape(-1)
        if w_t.numel() != len(start_nodes):
            raise ValueError(
                f"seed_weights 长度应与 start_nodes 一致，实际为 {w_t.numel()} vs {len(start_nodes)}"
            )

    vec.index_add_(0, idx_t, w_t)

    s = float(vec.sum().item())
    if s > 0.0:
        vec /= s
    return vec


# --------------------
# 一个脚本内置的加权 BPN 兼容实现
# 说明：
#   你没有贴 src/models/bpn_propagation.py，
#   所以这里做一个“兼容 fallback”：
#   - 若原 BPNPropagator 支持显式 seed 权重，则优先走原实现
#   - 否则自动回退到这里的加权传播，不需要你改第 4 个文件
# --------------------


class _WeightedBPNFallback:
    def __init__(
        self,
        steps_fwd: int,
        steps_bwd: int,
        decay_fwd: float,
        decay_bwd: float,
        combine: str,
        out_norm: str,
        degree_norm: str,
        include_seed: bool,
    ) -> None:
        self.steps_fwd = int(steps_fwd)
        self.steps_bwd = int(steps_bwd)
        self.decay_fwd = float(decay_fwd)
        self.decay_bwd = float(decay_bwd)
        self.combine = str(combine)
        self.out_norm = str(out_norm)
        self.degree_norm = str(degree_norm)
        self.include_seed = bool(include_seed)

    @staticmethod
    def _edge_tensors(
        g: dgl.DGLGraph, reverse: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        src, dst = g.edges()
        src = src.to(torch.long).cpu()
        dst = dst.to(torch.long).cpu()

        if "weight" in g.edata:
            edge_w = g.edata["weight"].detach().cpu().float().reshape(-1)
        else:
            edge_w = torch.ones(src.shape[0], dtype=torch.float32)

        if reverse:
            src, dst = dst, src

        edge_w = torch.clamp(edge_w, min=0.0)
        return src, dst, edge_w

    def _transition_weight(
        self,
        num_nodes: int,
        src: torch.Tensor,
        dst: torch.Tensor,
        edge_w: torch.Tensor,
    ) -> torch.Tensor:
        if edge_w.numel() == 0:
            return edge_w

        mode = self.degree_norm.lower()

        if mode in ("none", "raw"):
            return edge_w

        if mode == "out":
            deg = torch.zeros(num_nodes, dtype=torch.float32)
            deg.index_add_(0, src, edge_w)
            return edge_w / deg[src].clamp_min(1e-12)

        if mode == "in":
            deg = torch.zeros(num_nodes, dtype=torch.float32)
            deg.index_add_(0, dst, edge_w)
            return edge_w / deg[dst].clamp_min(1e-12)

        if mode in ("sym", "both"):
            out_deg = torch.zeros(num_nodes, dtype=torch.float32)
            in_deg = torch.zeros(num_nodes, dtype=torch.float32)
            out_deg.index_add_(0, src, edge_w)
            in_deg.index_add_(0, dst, edge_w)
            return edge_w / torch.sqrt(
                out_deg[src].clamp_min(1e-12) * in_deg[dst].clamp_min(1e-12)
            )

        logger.warning("未知 degree_norm=%s，按 raw 处理。", self.degree_norm)
        return edge_w

    def _propagate(
        self,
        num_nodes: int,
        src: torch.Tensor,
        dst: torch.Tensor,
        trans_w: torch.Tensor,
        seed_vec: torch.Tensor,
        steps: int,
        decay: float,
    ) -> torch.Tensor:
        cur = seed_vec.clone()
        acc = seed_vec.clone() if self.include_seed else torch.zeros_like(seed_vec)

        for _ in range(max(int(steps), 0)):
            nxt = torch.zeros(num_nodes, dtype=torch.float32)
            if trans_w.numel() > 0:
                nxt.index_add_(0, dst, cur[src] * trans_w)
            cur = float(decay) * nxt
            acc = acc + cur

        return acc

    def _combine_maps(self, fwd_map: torch.Tensor, bwd_map: torch.Tensor) -> torch.Tensor:
        mode = self.combine.lower()

        if mode in ("hadamard", "mul", "product"):
            return fwd_map * bwd_map
        if mode in ("sum", "add"):
            return fwd_map + bwd_map
        if mode in ("mean", "avg", "average"):
            return 0.5 * (fwd_map + bwd_map)
        if mode == "max":
            return torch.maximum(fwd_map, bwd_map)
        if mode == "min":
            return torch.minimum(fwd_map, bwd_map)

        logger.warning("未知 combine=%s，回退为 hadamard。", self.combine)
        return fwd_map * bwd_map

    def _normalize_output(self, x: torch.Tensor) -> torch.Tensor:
        mode = self.out_norm.lower()

        if mode in ("none", "raw"):
            return x

        if mode == "l1":
            s = float(x.sum().item())
            return x / s if s > 0.0 else x

        if mode == "l2":
            s = float(torch.norm(x, p=2).item())
            return x / s if s > 0.0 else x

        if mode == "max":
            s = float(x.max().item()) if x.numel() > 0 else 0.0
            return x / s if s > 0.0 else x

        logger.warning("未知 out_norm=%s，输出不做归一化。", self.out_norm)
        return x

    def compute_endpoint_importance(
        self,
        g: dgl.DGLGraph,
        start_nodes: Sequence[int],
        endpoint_node: Optional[int],
        seed_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_nodes = int(g.num_nodes())

        seed_fwd = _seed_vector_from_nodes(
            num_nodes=num_nodes,
            start_nodes=start_nodes,
            seed_weights=seed_weights,
        )

        seed_bwd = torch.zeros(num_nodes, dtype=torch.float32)
        if endpoint_node is not None and 0 <= int(endpoint_node) < num_nodes:
            seed_bwd[int(endpoint_node)] = 1.0

        src_fwd, dst_fwd, ew_fwd = self._edge_tensors(g, reverse=False)
        src_bwd, dst_bwd, ew_bwd = self._edge_tensors(g, reverse=True)

        tw_fwd = self._transition_weight(num_nodes, src_fwd, dst_fwd, ew_fwd)
        tw_bwd = self._transition_weight(num_nodes, src_bwd, dst_bwd, ew_bwd)

        has_fwd = float(seed_fwd.sum().item()) > 0.0
        has_bwd = float(seed_bwd.sum().item()) > 0.0

        fwd_map = (
            self._propagate(
                num_nodes=num_nodes,
                src=src_fwd,
                dst=dst_fwd,
                trans_w=tw_fwd,
                seed_vec=seed_fwd,
                steps=self.steps_fwd,
                decay=self.decay_fwd,
            )
            if has_fwd
            else torch.zeros(num_nodes, dtype=torch.float32)
        )

        bwd_map = (
            self._propagate(
                num_nodes=num_nodes,
                src=src_bwd,
                dst=dst_bwd,
                trans_w=tw_bwd,
                seed_vec=seed_bwd,
                steps=self.steps_bwd,
                decay=self.decay_bwd,
            )
            if has_bwd
            else torch.zeros(num_nodes, dtype=torch.float32)
        )

        if has_fwd and has_bwd:
            out = self._combine_maps(fwd_map, bwd_map)
        elif has_fwd:
            out = fwd_map
        elif has_bwd:
            out = bwd_map
        else:
            out = torch.zeros(num_nodes, dtype=torch.float32)

        return self._normalize_output(out)


class _ImportanceComputer:
    """
    兼容层：
      - 无 NUIAT 权重时，直接调用原 BPNPropagator
      - 有 NUIAT 权重时，优先尝试原 BPNPropagator 的加权接口
      - 若原类没有暴露加权接口，则回退到脚本内置实现
    """

    def __init__(self, bpn: BPNPropagator, weighted_fallback: _WeightedBPNFallback):
        self.bpn = bpn
        self.weighted_fallback = weighted_fallback

        self._force_fallback = False
        self._fallback_logged = False
        self._native_attempt_idx: Optional[int] = None

    def _log_fallback_once(self, msg: str) -> None:
        if not self._fallback_logged:
            logger.warning(msg)
            self._fallback_logged = True

    def compute(
        self,
        g: dgl.DGLGraph,
        start_nodes: Sequence[int],
        endpoint_node: Optional[int],
        seed_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 无 NUIAT 权重：保持你原先的调用路径
        if seed_weights is None:
            return self.bpn.compute_endpoint_importance(
                g,
                start_nodes=start_nodes,
                endpoint_node=endpoint_node,
            )

        # 已知必须回退
        if self._force_fallback:
            return self.weighted_fallback.compute_endpoint_importance(
                g,
                start_nodes=start_nodes,
                endpoint_node=endpoint_node,
                seed_weights=seed_weights,
            )

        seed_weights = seed_weights.detach().cpu().float()
        seed_vector = _seed_vector_from_nodes(
            num_nodes=int(g.num_nodes()),
            start_nodes=start_nodes,
            seed_weights=seed_weights,
        )

        all_attempts = [
            ("seed_weights", seed_weights),
            ("seed_weights", seed_weights.tolist()),
            ("start_weights", seed_weights),
            ("start_weights", seed_weights.tolist()),
            ("seed_scores", seed_weights),
            ("seed_scores", seed_weights.tolist()),
            ("seed_vector", seed_vector),
            ("start_distribution", seed_vector),
        ]

        if self._native_attempt_idx is not None:
            attempt_indices = [self._native_attempt_idx]
        else:
            attempt_indices = list(range(len(all_attempts)))

        for idx in attempt_indices:
            kw_name, kw_value = all_attempts[idx]
            try:
                out = self.bpn.compute_endpoint_importance(
                    g,
                    start_nodes=start_nodes,
                    endpoint_node=endpoint_node,
                    **{kw_name: kw_value},
                )
                self._native_attempt_idx = idx
                return out
            except TypeError as e:
                msg = str(e)
                # 典型“这个 kw 不支持”的情况：继续试别的 kw
                if "unexpected keyword" in msg or "got an unexpected keyword" in msg:
                    continue

                # 其他 TypeError：说明原实现加权接口不稳妥，回退
                self._log_fallback_once(
                    f"BPNPropagator 加权调用失败（{msg}），将切换到脚本内置 NUIAT 加权传播实现。"
                )
                self._force_fallback = True
                break
            except Exception as e:
                self._log_fallback_once(
                    f"BPNPropagator 加权调用失败（{e}），将切换到脚本内置 NUIAT 加权传播实现。"
                )
                self._force_fallback = True
                break

        if self._native_attempt_idx is None and not self._force_fallback:
            self._log_fallback_once(
                "当前 BPNPropagator 未暴露显式 seed 权重接口，已切换到脚本内置 NUIAT 加权传播实现。"
            )
            self._force_fallback = True

        return self.weighted_fallback.compute_endpoint_importance(
            g,
            start_nodes=start_nodes,
            endpoint_node=endpoint_node,
            seed_weights=seed_weights,
        )


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

    temp = bpn_cfg.pop("nuiat_temperature", None)
    if temp is None:
        temp = bpn_cfg.pop("temperature", None)
    if temp is None:
        temp = bpn_cfg.pop("softmax_temperature", 0.05)

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
        # 改动点：NUIAT softmax 温度参数
        "use_nuiat_softmax": bool(_pop("use_nuiat_softmax", True)),
        "nuiat_temperature": float(temp),
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
    edge_weight = data.get("edge_weight", None)
    nuiat_times_arr = data.get("nuiat_times", None)

    num_nodes = int(node_features.shape[0])
    g = _build_graph_from_edges(edges, num_nodes, edge_weight=edge_weight)
    logger.info(
        "Graph built: num_nodes=%d, num_edges=%d, has_edge_weight=%s",
        num_nodes,
        int(g.num_edges()),
        "weight" in g.edata,
    )

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

    if nuiat_times_arr is not None and len(nuiat_times_arr) != len(endpoints):
        logger.warning(
            "nuiat_times 与 endpoints 长度不一致：%d vs %d，部分 endpoint 将回退为均匀 seed。",
            len(nuiat_times_arr),
            len(endpoints),
        )

    bpn_cfg = _get_bpn_config(cfg)
    if bpn_cfg["use_nuiat_softmax"] and bpn_cfg["nuiat_temperature"] <= 0.0:
        raise ValueError(
            f"bpn.nuiat_temperature 必须 > 0，实际为 {bpn_cfg['nuiat_temperature']}"
        )

    if nuiat_times_arr is None:
        logger.info("Dataset 中未找到 nuiat_times，BPN seed 将回退为原始均匀分配。")
    else:
        logger.info(
            "NUIAT softmax is %s, temperature=%.6f",
            "enabled" if bpn_cfg["use_nuiat_softmax"] else "disabled",
            bpn_cfg["nuiat_temperature"],
        )

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

    weighted_fallback = _WeightedBPNFallback(
        steps_fwd=bpn_cfg["steps_fwd"],
        steps_bwd=bpn_cfg["steps_bwd"],
        decay_fwd=bpn_cfg["decay_fwd"],
        decay_bwd=bpn_cfg["decay_bwd"],
        combine=bpn_cfg["combine"],
        out_norm=bpn_cfg["out_norm"],
        degree_norm=bpn_cfg["degree_norm"],
        include_seed=bpn_cfg["include_seed"],
    )

    importance_engine = _ImportanceComputer(
        bpn=bpn,
        weighted_fallback=weighted_fallback,
    )

    all_importance: List[torch.Tensor] = []
    endpoint_nodes: List[Optional[int]] = []

    topk = bpn_cfg["topk"]
    max_eps_log = bpn_cfg["max_endpoints"]

    for i, ep_name in enumerate(endpoints):
        # 1) 解析 CPL 起点（已经是节点索引）
        raw_cpl = cpl_indices[i]
        start_nodes_raw = _extract_valid_start_nodes(raw_cpl, num_nodes)

        # 2) 取与 cpl_indices 对齐的 NUIAT 时间，并构建 softmax seed 分布
        raw_nuiat = None
        if nuiat_times_arr is not None and i < len(nuiat_times_arr):
            raw_nuiat = nuiat_times_arr[i]

        start_nodes, seed_weights, seed_mode, seed_debug = _prepare_seed_distribution(
            start_nodes=start_nodes_raw,
            raw_nuiat_times=raw_nuiat,
            temperature=bpn_cfg["nuiat_temperature"],
            use_nuiat_softmax=bpn_cfg["use_nuiat_softmax"],
        )

        # 3) endpoint 映射到图节点索引：使用原始名字（带下划线）
        ep_idx = name_to_idx.get(ep_name, None)
        ep_idx_int: Optional[int] = int(ep_idx) if ep_idx is not None else None
        endpoint_nodes.append(ep_idx_int)

        # 4) 调 BPN（若原类不支持 seed_weights，则自动回退到脚本内置实现）
        imp = importance_engine.compute(
            g,
            start_nodes=start_nodes,
            endpoint_node=ep_idx_int,
            seed_weights=seed_weights,
        )
        if not isinstance(imp, torch.Tensor):
            imp = torch.as_tensor(imp, dtype=torch.float32)
        imp = imp.detach().cpu().reshape(-1).float()

        if imp.numel() != num_nodes:
            raise ValueError(
                f"importance 长度应为 num_nodes={num_nodes}，实际为 {imp.numel()}"
            )

        all_importance.append(imp.unsqueeze(0))

        # 5) 日志：只打印前若干 endpoint 的 TopK
        if i < max_eps_log:
            imp_np = imp.numpy()

            if topk > 0:
                topk_idx = np.argsort(-imp_np)[:topk]
                topk_list = [
                    [idx_to_name.get(int(j), str(j)), float(imp_np[j])]
                    for j in topk_idx
                ]
            else:
                topk_list = []

            seed_top = []
            if seed_debug and topk > 0:
                for node_idx, t_val, w_val in sorted(
                    seed_debug, key=lambda x: -x[2]
                )[:topk]:
                    seed_top.append(
                        [
                            idx_to_name.get(int(node_idx), str(node_idx)),
                            None if not np.isfinite(t_val) else float(t_val),
                            float(w_val),
                        ]
                    )

            record = {
                "endpoint": ep_name,  # 原始名字，如 "_246_"
                "endpoint_node": idx_to_name.get(ep_idx_int, "N/A")
                if ep_idx_int is not None
                else "N/A",
                "seed_mode": seed_mode,
                "seed_temperature": bpn_cfg["nuiat_temperature"]
                if seed_mode == "nuiat_softmax"
                else None,
                "num_cpl_starts_raw": len(start_nodes_raw),
                "num_cpl_starts": len(start_nodes),
                "seed_top": seed_top,
                "topk": topk_list,
                "sum": float(imp_np.sum()),
                "max": float(imp_np.max()),
            }
            logger.info("[BPN] EP[%d] %s", i, json.dumps(record, ensure_ascii=False))

    if not all_importance:
        logger.warning("No endpoint importance generated. Nothing to save.")
        return

    # 汇总并保存
    importance_tensor = torch.cat(all_importance, dim=0)
    importance = importance_tensor.numpy()

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