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
from src.eda_parser.liberty_parser import parse_liberty_pin_directions, build_cell_output_index
from src.features.graph_builder import build_graph
from src.features.cnn_maps import build_physical_maps
from src.labels.cpl import compute_cpl_labels
from src.labels.mapping import name_to_node_idx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("build_dataset")

def load_metadata(project_root: str, design: str) -> Dict:
    out_dir = os.path.join(project_root, "data", "raw_eda", design)
    meta_path = os.path.join(out_dir, "metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"metadata.json not found: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_vocab(all_cell_types: List[str]) -> Dict[str, int]:
    uniq = sorted(set(all_cell_types))
    return {c: i for i, c in enumerate(uniq)}

def onehot_types(instances: Dict[str, Dict[str, object]], vocab: Dict[str, int]) -> np.ndarray:
    N = len(instances)
    D = len(vocab)
    m = np.zeros((N, D), dtype=np.float32)
    for i, name in enumerate(instances.keys()):
        c = str(instances[name].get('type', 'UNK'))
        if c in vocab:
            m[i, vocab[c]] = 1.0
    return m

def _load_cell_outputs_from_liberty(meta: Dict) -> Optional[Dict[str, set]]:
    lib_path = None
    try:
        # compatible with metadata schema in earlier steps
        maybe = meta.get("lib", {})
        lib_path = maybe.get("lib", None) or maybe.get("liberty", None)
    except Exception:
        lib_path = None
    if lib_path and os.path.exists(lib_path):
        try:
            pin_dirs = parse_liberty_pin_directions(lib_path)
            return build_cell_output_index(pin_dirs)
        except Exception as e:
            logger.warning("Liberty parsing failed: %s. Will fallback to heuristic.", e)
            return None
    return None

def process_one_design(project_root: str, design: str, grid_size: int,
                       near_cfg: Dict[str, object],
                       save_dir: str,
                       vocab: Dict[str, int] | None) -> Tuple[Dict[str,int], Dict[str,int]]:
    out_dir = os.path.join(project_root, "data", "raw_eda", design)
    synth_v = os.path.join(out_dir, f"{design}.synth.v")
    pre_def = os.path.join(out_dir, f"{design}.pre_route.def")
    post_def = os.path.join(out_dir, f"{design}.post_route.def")
    pre_timing  = os.path.join(out_dir, f"{design}.pre_route_timing.rpt")
    post_timing = os.path.join(out_dir, f"{design}.post_route_timing.rpt")
    meta = load_metadata(project_root, design)

    instances, net2pins, ports = parse_gate_level_verilog(synth_v)
    # try Liberty for true pin directions
    cell_outputs = _load_cell_outputs_from_liberty(meta)
    g, name_to_idx, base_node_feats = build_graph(instances, net2pins, ports, cell_outputs=cell_outputs)

    cell_types = [str(instances[n]['type']) for n in instances.keys()]
    if vocab is None:
        vocab = build_vocab(cell_types)
    type_onehot = onehot_types(instances, vocab)
    node_features = np.concatenate([base_node_feats, type_onehot], axis=1)

    def_path = post_def if os.path.exists(post_def) else pre_def
    comps, die_area, pins_xy = parse_def(def_path)
    if not die_area:
        logger.warning("No DIEAREA; fallback to synthetic die area.")
        die_area = ((0,0),(10000,10000))

    # Prefer HPWL-based RUDY; fallback internally to smoothed proxy
    maps = build_physical_maps(comps, die_area, pins_xy, grid_size=grid_size, net_to_pins=net2pins, prefer_hpwl_rudy=True)

    tpath = post_timing if os.path.exists(post_timing) else pre_timing
    paths = parse_report_checks(tpath)
    by_ep = group_by_endpoint(paths)
    near = compute_cpl_labels(by_ep,
                              mode=str(near_cfg.get("mode","delta_abs")),
                              delta_ns=float(near_cfg.get("delta_ns",0.02)),
                              quantile=float(near_cfg.get("quantile",0.10)))
    endpoints: List[str] = []
    y_arrival: List[float] = []
    cpl_indices: List[List[int]] = []
    for ep, lst in near.items():
        if not lst:
            continue
        arr = None
        for p in by_ep.get(ep, []):
            if p.get('arrival') is not None:
                arr = float(p['arrival']); break
        if arr is None:
            arr = -float(min([p.get('slack', 0.0) for p in by_ep.get(ep, lst)]))
        starts = [p.get('startpoint','') for p in lst]
        idxs = name_to_node_idx(starts, name_to_idx)
        if not idxs:
            continue
        endpoints.append(ep)
        y_arrival.append(arr)
        cpl_indices.append(idxs)

    if not endpoints:
        logger.warning("No endpoints with labels found for %s.", design)

    import dgl, torch  # ensure tensors to numpy stable
    edges = np.stack([g.edges()[0].numpy(), g.edges()[1].numpy()], axis=0)

    os.makedirs(save_dir, exist_ok=True)
    out_npz = os.path.join(save_dir, f"{design}.npz")
    np.savez_compressed(out_npz,
                        node_features=node_features,
                        edges=edges,
                        cell_density_map=maps['cell_density_map'],
                        rudy_map=maps['rudy_map'],
                        macro_mask_map=maps['macro_mask_map'],
                        endpoints=np.array(endpoints, dtype=object),
                        y_arrival=np.array(y_arrival, dtype=np.float32),
                        cpl_indices=np.array(cpl_indices, dtype=object),
                        name_to_idx=np.array(list(name_to_idx.items()), dtype=object),
                        vocab=np.array(list(vocab.items()), dtype=object))
    logger.info("Saved dataset: %s", out_npz)

    vocab_json = os.path.join(save_dir, "vocab.json")
    if os.path.exists(vocab_json):
        old = dict(json.load(open(vocab_json, "r", encoding="utf-8")))
        merged = dict(sorted(set(list(old.items()) + list(vocab.items())), key=lambda x: x[0]))
        with open(vocab_json, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2)
    else:
        with open(vocab_json, "w", encoding="utf-8") as f:
            json.dump(vocab, f, indent=2)
    return vocab, {design: len(endpoints)}

def main():
    parser = argparse.ArgumentParser(description="Build dataset from EDA artifacts.")
    parser.add_argument("--config", type=str, default="configs/dataset.yaml", help="dataset config yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, "r"))
    project_root = os.path.abspath(cfg.get("project_root","."))

    benches: List[str] = cfg.get("benches", [])
    if not benches:
        raise ValueError("No benches specified in dataset.yaml")
    save_dir = os.path.join(project_root, cfg.get("save_dir","data/processed"))
    grid_size = int(cfg.get("grid_size", 64))
    near_cfg = cfg.get("near_critical", {"mode": "delta_abs", "delta_ns": 0.02})

    vocab: Dict[str,int] | None = None
    stats: Dict[str,int] = {}
    for d in tqdm(benches, desc="build"):
        vocab, s = process_one_design(project_root, d, grid_size, near_cfg, save_dir, vocab)
        stats.update(s)
    with open(os.path.join(save_dir, "build_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    logger.info("Done. Stats: %s", stats)

if __name__ == "__main__":
    main()