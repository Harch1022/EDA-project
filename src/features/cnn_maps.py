from __future__ import annotations
import logging
from typing import Dict, Tuple, List, Optional
import numpy as np

logger = logging.getLogger(__name__)

Coord = Tuple[int, int]

def _to_grid(x: int, y: int, x0: int, y0: int, x1: int, y1: int, W: int, H: int):
    gx = int((x - x0) / max(x1 - x0, 1) * (W - 1))
    gy = int((y - y0) / max(y1 - y0, 1) * (H - 1))
    gx = min(max(gx, 0), W - 1)
    gy = min(max(gy, 0), H - 1)
    return gx, gy

def _rudy_smoothing_proxy(cell_density: np.ndarray) -> np.ndarray:
    # smooth to get a proxy for routing demand (RUDY-lite)
    try:
        from scipy.signal import convolve2d
        kernel = np.array([[0.25, 0.5, 0.25],
                           [0.5,  1.0, 0.5 ],
                           [0.25, 0.5, 0.25]], dtype=np.float32)
        rudy = convolve2d(cell_density, kernel, mode='same', boundary='symm')
    except Exception:
        # fallback simple blur
        rudy = cell_density.copy()
        for _ in range(2):
            rudy = (np.roll(rudy, 1, 0) + np.roll(rudy, -1, 0) +
                    np.roll(rudy, 1, 1) + np.roll(rudy, -1, 1) + rudy) / 5.0
    return rudy

def _build_cell_density_and_macro(components: Dict[str, Dict[str, object]],
                                  die_area: Tuple[Tuple[int,int], Tuple[int,int]],
                                  grid_size: int) -> Tuple[np.ndarray, np.ndarray]:
    (x0, y0), (x1, y1) = die_area
    W = H = grid_size
    cell_density = np.zeros((H, W), dtype=np.float32)
    macro_mask = np.zeros((H, W), dtype=np.float32)

    for _, info in components.items():
        xy = info.get('xy', (None, None))
        if xy is None:
            continue
        x, y = xy
        if x is None or y is None:
            continue
        gx, gy = _to_grid(x, y, x0, y0, x1, y1, W, H)
        cell_density[gy, gx] += 1.0
        macro = str(info.get('macro', '')).upper()
        if any(k in macro for k in ["SRAM", "ROM", "MACRO", "PLL"]):
            macro_mask[gy, gx] = 1.0

    if cell_density.max() > 0:
        cell_density = cell_density / (cell_density.max() + 1e-6)
    return cell_density, macro_mask

def _hpwl_rudy_map(components: Dict[str, Dict[str, object]],
                   net_to_pins: Optional[Dict[str, List[Tuple[str, str]]]],
                   die_area: Tuple[Tuple[int,int], Tuple[int,int]],
                   grid_size: int = 64) -> Optional[np.ndarray]:
    """
    Build HPWL-based RUDY:
      For each net, compute bbox from instance centers, HPWL = dx + dy,
      and distribute demand uniformly over covered grid cells with density ~ (HPWL * deg) / area.
      If insufficient data, return None.
    """
    if net_to_pins is None or not net_to_pins:
        return None
    (x0, y0), (x1, y1) = die_area
    W = H = grid_size
    rudy = np.zeros((H, W), dtype=np.float32)

    # quick access: instance -> (x,y)
    inst_xy: Dict[str, Tuple[int,int]] = {}
    for inst, info in components.items():
        xy = info.get('xy', (None, None))
        if xy and xy[0] is not None and xy[1] is not None:
            inst_xy[inst] = (int(xy[0]), int(xy[1]))

    for net, conns in net_to_pins.items():
        pts: List[Tuple[int,int]] = []
        seen_inst = set()
        for inst, _pin in conns:
            if inst in inst_xy and inst not in seen_inst:
                pts.append(inst_xy[inst])
                seen_inst.add(inst)
        if len(pts) < 2:
            continue
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        dx = max(xmax - xmin, 0); dy = max(ymax - ymin, 0)
        hpwl = float(dx + dy)
        area = float(max(dx, 1) * max(dy, 1))
        deg = max(len(pts), 2)
        demand_density = (hpwl * deg) / area  # classical rudy scaling

        gx0, gy0 = _to_grid(xmin, ymin, x0, y0, x1, y1, W, H)
        gx1, gy1 = _to_grid(xmax, ymax, x0, y0, x1, y1, W, H)
        xa, xb = sorted([gx0, gx1]); ya, yb = sorted([gy0, gy1])
        cells = max((xb - xa + 1) * (yb - ya + 1), 1)
        incr = demand_density / cells
        rudy[ya:yb+1, xa:xb+1] += incr

    if rudy.max() > 0:
        rudy = rudy / (rudy.max() + 1e-6)
    return rudy

def build_physical_maps(components: Dict[str, Dict[str, object]],
                        die_area: Tuple[Tuple[int,int], Tuple[int,int]],
                        pins_xy: Dict[str, Tuple[int,int]],
                        grid_size: int = 64,
                        net_to_pins: Optional[Dict[str, List[Tuple[str, str]]]] = None,
                        prefer_hpwl_rudy: bool = True) -> Dict[str, np.ndarray]:
    """
    Build physical feature maps:
      - cell_density_map: [H, W]
      - rudy_map: [H, W] (HPWL-based if possible, fallback to smoothed density)
      - macro_mask_map: [H, W] (binary)
    """
    cell_density, macro_mask = _build_cell_density_and_macro(components, die_area, grid_size)

    rudy_map: Optional[np.ndarray] = None
    if prefer_hpwl_rudy:
      try:
          rudy_map = _hpwl_rudy_map(components, net_to_pins, die_area, grid_size)
      except Exception as e:
          logger.warning("HPWL-RUDY failed: %s. Fallback to smoothed density.", e)

    if rudy_map is None:
        rudy_map = _rudy_smoothing_proxy(cell_density)
        if rudy_map.max() > 0:
            rudy_map = rudy_map / (rudy_map.max() + 1e-6)

    return {
        "cell_density_map": cell_density.astype(np.float32),
        "rudy_map": rudy_map.astype(np.float32),
        "macro_mask_map": macro_mask.astype(np.float32)
    }