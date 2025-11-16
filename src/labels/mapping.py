from __future__ import annotations
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

def name_to_node_idx(start_names: List[str], name_to_idx: Dict[str, int]) -> List[int]:
    """
    Map instance names (with optional pin suffix like /Q) to node indices by exact/trimmed match.
    """
    idxs: List[int] = []
    for nm in start_names:
        base = nm.split('/')[0]
        if base in name_to_idx:
            idxs.append(name_to_idx[base])
        else:
            base2 = base.split('.')[-1]
            if base2 in name_to_idx:
                idxs.append(name_to_idx[base2])
            else:
                logger.debug("Unmapped startpoint name: %s", nm)
    return sorted(set(idxs))