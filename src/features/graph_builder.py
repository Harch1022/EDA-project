from __future__ import annotations
import logging
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
import dgl
import torch

logger = logging.getLogger(__name__)

# backward-compatible heuristics
OUTPUT_PIN_CANDIDATES = {"Y", "Z", "ZN", "Q", "QN", "QB", "O", "S", "CO", "SUM"}

def build_graph(instances: Dict[str, Dict[str, object]],
                net_to_pins: Dict[str, List[Tuple[str, str]]],
                ports: Dict[str, str],
                cell_outputs: Optional[Dict[str, Set[str]]] = None):
    """
    Build a directed DGL graph from netlist.
    Args:
      - instances: inst_name -> {'type': cell_type, 'pins': {pin: net}}
      - net_to_pins: net -> [(inst_name, pin_name), ...]
      - ports: net_name -> direction
      - cell_outputs: optional mapping cell_type -> set(output_pin_names), from Liberty
    Returns:
      - g: DGLGraph
      - name_to_idx: instance name -> node index
      - node_features: np.ndarray [N, d], columns: [is_seq, fan_in, fan_out]
    Notes:
      - If cell_outputs is None, fallback to heuristics OUTPUT_PIN_CANDIDATES.
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
        # prefer Liberty
        if cell_outputs is not None:
            outs = cell_outputs.get(cell_type, None)
            if outs is not None:
                return pin in outs
        # fallback heuristic
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
        # fallback: if none identified as driver, assume first is driver
        if not drivers and conns:
            drivers = [conns[0][0]]
            sinks = [c[0] for c in conns[1:]]

        # add edges driver -> sink
        for d in drivers:
            for s in sinks:
                if d in name_to_idx and s in name_to_idx and d != s:
                    srcs.append(name_to_idx[d]); dsts.append(name_to_idx[s])

    if not srcs:
        logger.warning("No edges inferred; constructing empty-edge graph.")

    g = dgl.graph((torch.tensor(srcs, dtype=torch.int64),
                   torch.tensor(dsts, dtype=torch.int64)), num_nodes=N)

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
    return g, name_to_idx, node_feats