from __future__ import annotations
import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)

Coord = Tuple[int, int]

def parse_def(filepath: str) -> Tuple[Dict[str, Dict[str, object]], Optional[Tuple[Coord, Coord]], Dict[str, Coord]]:
    """
    Parse DEF to extract component placements, die area, and pin locations.
    Returns:
      - components: inst_name -> {'macro': cell_type, 'placed': bool, 'xy': (x,y), 'orient': str}
      - die_area: ((x0,y0), (x1,y1)) or None
      - pins: pin_name -> (x,y) approx coords if available
    """
    components: Dict[str, Dict[str, object]] = {}
    pins: Dict[str, Coord] = {}
    die_area: Optional[Tuple[Coord, Coord]] = None

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    in_components = False
    in_pins = False
    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        if line.upper().startswith("DIEAREA"):
            nums = [int(n) for n in line.replace('(', ' ').replace(')', ' ').replace(';',' ').split() if n.lstrip('-').isdigit()]
            if len(nums) >= 4:
                die_area = ((nums[0], nums[1]), (nums[2], nums[3]))
            continue

        if line.upper().startswith("COMPONENTS"):
            in_components = True
            continue
        if in_components and line.upper().startswith("END COMPONENTS"):
            in_components = False
            continue
        if in_components and line.startswith("-"):
            tokens = line.split()
            if len(tokens) < 3:
                continue
            inst = tokens[1]
            macro = tokens[2]
            placed = ("+ PLACED" in line) or ("+ FIXED" in line)
            xy = (0, 0)
            orient = "N"
            if placed:
                try:
                    parts = line.replace('(', ' ').replace(')', ' ').replace(';',' ').split()
                    xs = [int(t) for t in parts if t.lstrip('-').isdigit()]
                    if len(xs) >= 2:
                        xy = (xs[-2], xs[-1])
                    orient = parts[-1] if parts else "N"
                except Exception as e:
                    logger.warning("DEF parse placed coords failed for %s: %s", inst, e)
            components[inst] = {'macro': macro, 'placed': placed, 'xy': xy, 'orient': orient}
            continue

        if line.upper().startswith("PINS"):
            in_pins = True
            continue
        if in_pins and line.upper().startswith("END PINS"):
            in_pins = False
            continue
        if in_pins and line.startswith("-"):
            name = line.split()[1]
            if "+ PLACED" in line or "+ FIXED" in line:
                parts = line.replace('(', ' ').replace(')', ' ').replace(';',' ').split()
                xs = [int(t) for t in parts if t.lstrip('-').isdigit()]
                if len(xs) >= 2:
                    pins[name] = (xs[-2], xs[-1])

    logger.info("Parsed DEF %s: %d components, pins=%d, die_area=%s",
                filepath, len(components), len(pins), str(die_area))
    return components, die_area, pins