from __future__ import annotations
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

def compute_cpl_labels(paths_by_endpoint: Dict[str, List[dict]],
                       mode: str = "delta_abs",
                       delta_ns: float = 0.02,
                       quantile: float = 0.10) -> Dict[str, List[dict]]:
    """
    Compute near-critical path set per endpoint.
    mode:
      - delta_abs: keep paths with slack <= (min_slack + delta_ns)
      - quantile:  keep worst bottom quantile by slack
    """
    result: Dict[str, List[dict]] = {}
    for ep, lst in paths_by_endpoint.items():
        if not lst:
            result[ep] = []
            continue
        slacks = [p.get('slack', 0.0) for p in lst]
        min_slack = min(slacks)
        if mode == "delta_abs":
            thr = min_slack + abs(delta_ns)
            selected = [p for p in lst if p.get('slack', 0.0) <= thr]
        elif mode == "quantile":
            k = max(int(len(lst) * quantile), 1)
            selected = sorted(lst, key=lambda x: x.get('slack', 0.0))[:k]
        else:
            thr = min_slack + abs(delta_ns)
            selected = [p for p in lst if p.get('slack', 0.0) <= thr]
            logger.warning("Unknown mode=%s; fallback delta_abs thr=%f", mode, thr)
        result[ep] = selected
    return result