from __future__ import annotations
import logging
import math
from typing import Any, Dict, List, Tuple, Optional, Set
import numpy as np
import dgl
import torch

logger = logging.getLogger(__name__)

# backward-compatible heuristics
OUTPUT_PIN_CANDIDATES = {"Y", "Z", "ZN", "Q", "QN", "QB", "O", "S", "CO", "SUM"}

def _extract_inst_xy(comp: Any) -> Optional[Tuple[float, float]]:
    """
    从 DEF parser 返回的 comp 结构里提取实例坐标（优先中心坐标）。
    """
    if comp is None:
        return None

    if isinstance(comp, dict):
        # 优先取已有的中心坐标
        if "cx" in comp and "cy" in comp:
            return float(comp["cx"]), float(comp["cy"])
        if "center_x" in comp and "center_y" in comp:
            return float(comp["center_x"]), float(comp["center_y"])

        # 从标准 x, y 及宽高计算中心
        if "x" in comp and "y" in comp:
            x, y = float(comp["x"]), float(comp["y"])
            if "w" in comp and "h" in comp:
                x += 0.5 * float(comp["w"])
                y += 0.5 * float(comp["h"])
            return x, y
            
        # 你的 DEF parser 存的是 'xy' 字段
        if "xy" in comp and isinstance(comp["xy"], (tuple, list)) and len(comp["xy"]) >= 2:
            return float(comp["xy"][0]), float(comp["xy"][1])

    if isinstance(comp, (tuple, list)) and len(comp) >= 2:
        return float(comp[0]), float(comp[1])

    return None

def die_half_perimeter(die_area: Optional[Tuple[Tuple[float, float], Tuple[float, float]]]) -> float:
    """计算芯片半周长 W + H，用于距离归一化"""
    try:
        (lx, ly), (ux, uy) = die_area
        w = max(float(ux) - float(lx), 1.0)
        h = max(float(uy) - float(ly), 1.0)
        return w + h
    except Exception:
        return 1.0

def normalized_decay_manhattan(
    src_inst: str,
    dst_inst: str,
    comps: Optional[Dict[str, Any]],
    die_area: Optional[Tuple[Tuple[float, float], Tuple[float, float]]],
    eps: float = 1e-6,
) -> float:
    """
    计算 src_inst -> dst_inst 的指数衰减型归一化曼哈顿距离。
    距离越近，权重越接近 1；距离越远，权重呈指数衰减。
    """
    hp = die_half_perimeter(die_area)
    
    if comps is None:
        return 1.0  # 没有物理信息时，默认全连接权重为 1.0

    sxy = _extract_inst_xy(comps.get(src_inst, None))
    dxy = _extract_inst_xy(comps.get(dst_inst, None))

    if sxy is None or dxy is None:
        return 1.0  # 找不到坐标，默认不衰减

    dist = abs(sxy[0] - dxy[0]) + abs(sxy[1] - dxy[1])
    dist_norm = dist / (hp + eps)

    # 使用指数衰减：距离为0时权重为1，距离越远权重越小
    return float(math.exp(-dist_norm))

def build_graph(
    instances: Dict[str, Dict[str, object]],
    net_to_pins: Dict[str, List[Tuple[str, str]]],
    ports: Dict[str, str],
    cell_outputs: Optional[Dict[str, Set[str]]] = None,
    comps: Optional[Dict[str, object]] = None,            # 新增：组件物理信息
    die_area: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None # 新增：版图面积
):
    """
    Build a directed DGL graph from netlist with physical edge weights.
    """
    inst_names = list(instances.keys())
    name_to_idx = {n: i for i, n in enumerate(inst_names)}
    N = len(inst_names)
    srcs: List[int] = []
    dsts: List[int] = []

    def is_seq(cell_type: str) -> bool:
        up = cell_type.upper()
        return any(x in up for x in ["DFF", "SDFF", "DLH", "LATCH", "FLIPFLOP"])

    def pin_is_output(cell_type: str, pin: str) -> bool:
        if cell_outputs is not None:
            outs = cell_outputs.get(cell_type, None)
            if outs is not None:
                return pin in outs
        return (pin.upper() in OUTPUT_PIN_CANDIDATES) or pin.upper().endswith("_Q")

    for net, conns in net_to_pins.items():
        if not conns:
            continue
        drivers: List[str] = []
        sinks: List[str] = []
        for inst, pin in conns:
            if inst not in instances:
                continue
            ctype = str(instances[inst].get('type', ''))
            if pin_is_output(ctype, pin):
                drivers.append(inst)
            else:
                sinks.append(inst)
                
        if not drivers and conns:
            drivers = [conns[0][0]]
            sinks = [c[0] for c in conns[1:]]

        # add edges driver -> sink
        for d in drivers:
            for s in sinks:
                if d in name_to_idx and s in name_to_idx and d != s:
                    srcs.append(name_to_idx[d])
                    dsts.append(name_to_idx[s])

    if not srcs:
        logger.warning("No edges inferred; constructing empty-edge graph.")

    g = dgl.graph((torch.tensor(srcs, dtype=torch.int64),
                   torch.tensor(dsts, dtype=torch.int64)), num_nodes=N)

    # 节点特征计算
    fan_in = np.zeros((N,), dtype=np.float32)
    fan_out = np.zeros((N,), dtype=np.float32)
    for s, d in zip(srcs, dsts):
        fan_out[s] += 1.0
        fan_in[d] += 1.0

    is_seq_arr = np.zeros((N,), dtype=np.float32)
    for n, idx in name_to_idx.items():
        ctype = str(instances[n].get('type', ''))
        is_seq_arr[idx] = 1.0 if is_seq(ctype) else 0.0

    node_feats = np.stack([is_seq_arr, fan_in, fan_out], axis=1)

    # ======== 核心新增：计算边权并写入 DGL 图 ========
    edge_weight: List[float] = []
    if srcs:
        for s, d in zip(srcs, dsts):
            src_name = inst_names[s]
            dst_name = inst_names[d]
            # 计算指数衰减曼哈顿距离
            w = normalized_decay_manhattan(src_name, dst_name, comps=comps, die_area=die_area)
            edge_weight.append(w)

    if edge_weight:
        g.edata["weight"] = torch.tensor(edge_weight, dtype=torch.float32)
    else:
        g.edata["weight"] = torch.empty((0,), dtype=torch.float32)

    return g, name_to_idx, node_feats