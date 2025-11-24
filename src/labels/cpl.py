from __future__ import annotations
from typing import Dict, List, Any, Tuple
import math

PathRec = Dict[str, Any]   # expects keys like: 'startpoint', 'arrival', 'slack', ...

def _endpoint_score(paths: List[PathRec], ratio_metric: str = "arrival") -> float:
    """
    Compute endpoint-level severity score (higher is worse).
    - arrival: score = max(arrival)         (larger arrival => worse)
    - slack:   score = max(-slack)          (more negative slack => worse)
    """
    ratio_metric = (ratio_metric or "arrival").lower()
    if ratio_metric not in ("arrival", "slack"):
        ratio_metric = "arrival"
    score = 0.0
    if not paths:
        return score
    if ratio_metric == "arrival":
        vals = [p.get("arrival", None) for p in paths if p.get("arrival", None) is not None]
        score = max(vals) if vals else 0.0
    else:
        vals = [p.get("slack", None) for p in paths if p.get("slack", None) is not None]
        if vals:
            # worse slack => more negative; use -slack to make higher worse
            score = max([-float(s) for s in vals])
        else:
            score = 0.0
    return float(score)

def _select_near_paths(paths: List[PathRec],
                       inner_mode: str = "delta_abs",
                       delta_ns: float = 0.02,
                       quantile: float = 0.10,
                       base_metric: str = "arrival") -> List[PathRec]:
    """
    Select near-critical paths within an endpoint.
    inner_mode:
      - 'delta_abs': choose paths whose arrival within delta_ns to worst arrival.
      - 'quantile' : take top-q fraction by arrival (descending).
      - 'best_only': only the single worst path by arrival.
    If arrival missing, fallback to slack (min slack => worst).
    """
    if not paths:
        return []
    inner_mode = (inner_mode or "delta_abs").lower()
    base_metric = (base_metric or "arrival").lower()

    # robust retrieval of worst by arrival/slack
    def worst_by_arrival(ps: List[PathRec]) -> PathRec | None:
        ps2 = [p for p in ps if p.get("arrival", None) is not None]
        if not ps2:
            return None
        return max(ps2, key=lambda p: float(p["arrival"]))

    def worst_by_slack(ps: List[PathRec]) -> PathRec | None:
        ps2 = [p for p in ps if p.get("slack", None) is not None]
        if not ps2:
            return None
        # min slack is worst
        return min(ps2, key=lambda p: float(p["slack"]))

    if inner_mode == "best_only":
        w = worst_by_arrival(paths) if base_metric == "arrival" else worst_by_slack(paths)
        if w is None:
            # fallback
            w = worst_by_slack(paths) or paths[0]
        return [w]

    if inner_mode == "quantile":
        ps2 = [p for p in paths if p.get("arrival", None) is not None]
        if not ps2:
            # fallback to slack sorting ascending
            ps2 = paths[:]
            ps2.sort(key=lambda p: float(p.get("slack", 0.0)))
        else:
            ps2.sort(key=lambda p: float(p["arrival"]), reverse=True)
        k = max(1, int(math.ceil(len(ps2) * float(quantile))))
        return ps2[:k]

    # default: delta_abs
    w = worst_by_arrival(paths) if base_metric == "arrival" else worst_by_slack(paths)
    selected: List[PathRec] = []
    if w is not None and w.get("arrival", None) is not None:
        worst_arr = float(w["arrival"])
        for p in paths:
            a = p.get("arrival", None)
            if a is None:
                continue
            if worst_arr - float(a) <= float(delta_ns) + 1e-12:
                selected.append(p)
    else:
        # fallback to slack proximity: within delta_ns of worst slack (most negative)
        w2 = worst_by_slack(paths)
        if w2 is None or w2.get("slack", None) is None:
            return []
        worst_slk = float(w2["slack"])
        for p in paths:
            s = p.get("slack", None)
            if s is None:
                continue
            # near to worst: |s - worst_slk| <= delta_ns
            if abs(float(s) - worst_slk) <= float(delta_ns) + 1e-12:
                selected.append(p)
    if not selected and w is not None:
        selected = [w]
    return selected

def compute_cpl_labels(by_endpoint: Dict[str, List[PathRec]],
                       mode: str = "delta_abs",
                       delta_ns: float = 0.02,
                       quantile: float = 0.10,
                       ratio_metric: str = "arrival",
                       top_ratio: float = 1.0,
                       inner_mode: str = "delta_abs",
                       keep_non_selected: bool = False) -> Dict[str, List[PathRec]]:
    """
    Compute CPL labels (endpoint -> list[near-critical paths]) in multiple modes.

    Existing modes (per-endpoint path selection):
      - 'delta_abs': near to worst by absolute arrival threshold (delta_ns).
      - 'quantile' : within top quantile by arrival.
      - 'best_only': only the single worst path.

    New mode (endpoint selection first):
      - 'ratio_to_worst': select worst top_ratio endpoints according to ratio_metric
         (arrival: max arrival; slack: max -slack),
         then for each selected endpoint, choose near paths by inner_mode ('delta_abs'|'quantile'|'best_only').

    Args:
      by_endpoint: dict endpoint -> list(paths). Each path contains keys:
                   'startpoint' (str), 'arrival' (float, optional), 'slack' (float, optional), etc.
      mode: 'delta_abs' | 'quantile' | 'best_only' | 'ratio_to_worst'
      delta_ns: threshold for delta_abs.
      quantile: fraction for quantile selection [0,1].
      ratio_metric: 'arrival' | 'slack' (endpoint severity metric for ratio_to_worst).
      top_ratio: fraction of endpoints to keep in ratio_to_worst (0,1]. Default 1.0 (keep all).
      inner_mode: inner selection for ratio_to_worst: 'delta_abs' | 'quantile' | 'best_only'.
      keep_non_selected: if True, non-selected endpoints will still keep a label by best_only (compatibility option).

    Returns:
      near: endpoint -> list of selected paths (each path dict preserved).
    """
    mode = (mode or "delta_abs").lower().strip()
    near: Dict[str, List[PathRec]] = {}

    if mode != "ratio_to_worst":
        # legacy per-endpoint selection
        for ep, paths in by_endpoint.items():
            if not paths:
                continue
            sels = _select_near_paths(paths,
                                      inner_mode=mode,
                                      delta_ns=delta_ns,
                                      quantile=quantile,
                                      base_metric="arrival")
            if sels:
                near[ep] = sels
        return near

    # ratio_to_worst: endpoint-level selection
    # 1) compute endpoint scores
    ep_scores: List[Tuple[str, float]] = []
    for ep, paths in by_endpoint.items():
        score = _endpoint_score(paths, ratio_metric=ratio_metric)
        ep_scores.append((ep, score))

    if not ep_scores:
        return {}

    # 2) sort and pick top-K endpoints
    ep_scores.sort(key=lambda x: x[1], reverse=True)  # higher worse first
    top_ratio = float(top_ratio)
    if not (0.0 < top_ratio <= 1.0):
        top_ratio = 1.0
    K = max(1, int(math.ceil(len(ep_scores) * top_ratio)))
    selected_eps = set([ep for ep, _ in ep_scores[:K]])

    # 3) within each selected endpoint, choose near paths by inner_mode
    for ep, paths in by_endpoint.items():
        if not paths:
            continue
        if ep in selected_eps:
            sels = _select_near_paths(paths,
                                      inner_mode=inner_mode,
                                      delta_ns=delta_ns,
                                      quantile=quantile,
                                      base_metric="arrival")
            if sels:
                near[ep] = sels
        else:
            if keep_non_selected:
                # keep a minimal label to avoid losing endpoints (optional)
                sels = _select_near_paths(paths,
                                          inner_mode="best_only",
                                          delta_ns=delta_ns,
                                          quantile=quantile,
                                          base_metric="arrival")
                if sels:
                    near[ep] = sels
    return near