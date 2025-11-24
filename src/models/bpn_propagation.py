from __future__ import annotations
import math
from typing import List, Optional, Tuple
import torch
import dgl
import dgl.function as fn


def _reverse_graph(g: dgl.DGLGraph) -> dgl.DGLGraph:
    """
    Create a reversed graph with best-effort API compatibility across DGL versions.
    """
    try:
        return dgl.reverse(g, copy_ndata=False, copy_edata=False)
    except TypeError:
        # older/newer variants
        try:
            return dgl.reverse(g, copy_ndata=False)
        except TypeError:
            return dgl.reverse(g)


def _normalize_vector(x: torch.Tensor, mode: str = "l1", eps: float = 1e-12) -> torch.Tensor:
    """
    Normalize a vector.
    - l1: divide by sum(|x|)
    - l2: divide by sqrt(sum(x^2))
    - max: divide by max(|x|)
    """
    mode = (mode or "l1").lower()
    if mode == "l2":
        denom = torch.sqrt(torch.clamp(torch.sum(x * x), min=eps))
        return x / denom
    if mode == "max":
        denom = torch.clamp(torch.max(torch.abs(x)), min=eps)
        return x / denom
    # default l1
    denom = torch.clamp(torch.sum(torch.abs(x)), min=eps)
    return x / denom


def _propagate_mass(
    g: dgl.DGLGraph,
    seed: torch.Tensor,
    steps: int = 6,
    decay: float = 0.9,
    norm: str = "out",
    include_seed: bool = True,
) -> torch.Tensor:
    """
    Fixed-weight message passing propagation on a DGL graph.

    Args:
      g: DGLGraph (direction matters)
      seed: [N] vector, non-negative mass at seed nodes.
      steps: number of hops.
      decay: per-hop decay factor.
      norm: 'out' or 'in' (degree normalization side).
      include_seed: whether to include the seed itself in the accumulated score.

    Returns:
      score: [N] aggregated mass after multiple hops with decay.
    """
    device = seed.device
    N = g.num_nodes()
    assert seed.shape[0] == N, "seed length must match number of nodes"

    # Ensure float32
    x = seed.to(device=device, dtype=torch.float32)
    score = torch.zeros(N, device=device, dtype=torch.float32)

    if include_seed:
        score = score + x  # weight 1.0 for seed

    if steps <= 0:
        return score

    # Degree normalization
    if norm == "in":
        deg = g.in_degrees().to(device=device, dtype=torch.float32)
    else:
        deg = g.out_degrees().to(device=device, dtype=torch.float32)
    deg = torch.clamp(deg, min=1.0)
    deg_inv = 1.0 / deg

    # We reuse node data fields; avoid collision by unique keys
    for t in range(1, steps + 1):
        # normalize by degree at source side
        g.ndata["_bpn_src"] = x * deg_inv
        g.update_all(fn.copy_u("_bpn_src", "_bpn_m"), fn.sum("_bpn_m", "_bpn_acc"))
        x = g.ndata["_bpn_acc"]
        # clear node data to avoid side effects
        try:
            del g.ndata["_bpn_src"]
            del g.ndata["_bpn_acc"]
        except KeyError:
            pass
        # decay and accumulate
        score = score + (decay ** t) * x

    return score


class BPNPropagator:
    """
    Lightweight BPN-like endpoint-wise importance estimator.

    - Forward mass: propagate from CPL startpoints along original graph.
    - Backward mass: propagate from endpoint along reversed graph.
    - Combine: hadamard product (default) or sum.

    Note: This is a deterministic, non-learned approximation to get a reasonable
          importance map that highlights nodes that are both close to CPL startpoints
          and close to the endpoint.
    """

    def __init__(
        self,
        steps_fwd: int = 6,
        steps_bwd: int = 6,
        decay_fwd: float = 0.9,
        decay_bwd: float = 0.9,
        combine: str = "hadamard",  # or "sum"
        out_norm: str = "l1",       # "l1" | "l2" | "max"
        degree_norm: str = "out",   # 'out' or 'in'
        include_seed: bool = True,
    ) -> None:
        self.steps_fwd = int(steps_fwd)
        self.steps_bwd = int(steps_bwd)
        self.decay_fwd = float(decay_fwd)
        self.decay_bwd = float(decay_bwd)
        self.combine = (combine or "hadamard").lower()
        self.out_norm = (out_norm or "l1").lower()
        self.degree_norm = (degree_norm or "out").lower()
        self.include_seed = bool(include_seed)

    def _combine(self, fwd: torch.Tensor, bwd: Optional[torch.Tensor]) -> torch.Tensor:
        if bwd is None:
            return _normalize_vector(fwd, mode=self.out_norm)
        if self.combine == "sum":
            z = fwd + bwd
        else:
            z = fwd * bwd
        return _normalize_vector(z, mode=self.out_norm)

    def compute_endpoint_importance(
        self,
        g: dgl.DGLGraph,
        start_nodes: List[int],
        endpoint_node: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Compute importance for a single endpoint.

        Args:
          g: DGLGraph
          start_nodes: CPL startpoint node indices (seed forward)
          endpoint_node: endpoint node index (seed backward); if None, only forward is used.
          device: torch device; if None, use CPU.

        Returns:
          importance: [N] tensor, normalized by self.out_norm
        """
        if device is None:
            device = torch.device("cpu")
        N = g.num_nodes()
        # sanitize start_nodes
        start_nodes = [int(i) for i in start_nodes if 0 <= int(i) < N]
        if len(start_nodes) == 0 and endpoint_node is None:
            # no info
            return torch.zeros(N, dtype=torch.float32, device=device)

        # seeds
        f_seed = torch.zeros(N, dtype=torch.float32, device=device)
        if len(start_nodes) > 0:
            f_mass = 1.0 / float(len(start_nodes))
            f_seed[start_nodes] = f_mass

        # forward on g
        fwd = torch.zeros(N, dtype=torch.float32, device=device)
        if len(start_nodes) > 0:
            fwd = _propagate_mass(
                g, f_seed,
                steps=self.steps_fwd,
                decay=self.decay_fwd,
                norm=self.degree_norm,
                include_seed=self.include_seed
            )

        # backward on reversed graph if endpoint available
        bwd = None
        if endpoint_node is not None and 0 <= int(endpoint_node) < N:
            e_seed = torch.zeros(N, dtype=torch.float32, device=device)
            e_seed[int(endpoint_node)] = 1.0
            g_rev = _reverse_graph(g)
            bwd = _propagate_mass(
                g_rev, e_seed,
                steps=self.steps_bwd,
                decay=self.decay_bwd,
                norm=self.degree_norm,
                include_seed=self.include_seed
            )

        return self._combine(fwd, bwd)

    def compute_multi(
        self,
        g: dgl.DGLGraph,
        starts_per_endpoint: List[List[int]],
        endpoint_nodes: Optional[List[Optional[int]]] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Compute importance for multiple endpoints.

        Args:
          g: DGLGraph
          starts_per_endpoint: list of list of start nodes for each endpoint.
          endpoint_nodes: list of endpoint node idx (or None per endpoint).
          device: torch device

        Returns:
          importance: [E, N]
        """
        if device is None:
            device = torch.device("cpu")
        E = len(starts_per_endpoint)
        if endpoint_nodes is None:
            endpoint_nodes = [None] * E
        assert len(endpoint_nodes) == E, "endpoint_nodes size mismatch"

        outs = []
        for i in range(E):
            imp = self.compute_endpoint_importance(
                g,
                start_nodes=starts_per_endpoint[i],
                endpoint_node=endpoint_nodes[i],
                device=device,
            )
            outs.append(imp.unsqueeze(0))
        return torch.cat(outs, dim=0) if outs else torch.zeros((0, g.num_nodes()), dtype=torch.float32, device=device)