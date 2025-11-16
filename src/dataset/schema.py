from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import numpy as np

@dataclass
class GraphSample:
    node_features: np.ndarray
    edges: np.ndarray
    cell_density_map: np.ndarray
    rudy_map: np.ndarray
    macro_mask_map: np.ndarray
    endpoints: List[str]
    y_arrival: np.ndarray
    cpl_indices: List[List[int]]
    name_to_idx: Dict[str, int]
    vocab: Dict[str, int]