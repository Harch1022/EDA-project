from __future__ import annotations
import torch
import dgl
from src.models.bpn_propagation import BPNPropagator


def build_chain_graph(n: int) -> dgl.DGLGraph:
    # chain 0->1->...->(n-1)
    src = list(range(n - 1))
    dst = list(range(1, n))
    g = dgl.graph((torch.tensor(src), torch.tensor(dst)), num_nodes=n)
    return g


def test_bpn_chain_forward_backward():
    n = 6
    g = build_chain_graph(n)  # 0->1->2->3->4->5
    start_nodes = [0]
    endpoint_node = 5
    bpn = BPNPropagator(steps_fwd=6, steps_bwd=6, decay_fwd=0.9, decay_bwd=0.5, combine="hadamard", out_norm="l1")

    imp = bpn.compute_endpoint_importance(g, start_nodes=start_nodes, endpoint_node=endpoint_node)
    assert imp.shape[0] == n
    # All nodes on the chain should get > 0; there is no off-path node here.
    assert float(torch.sum(imp > 0).item()) == n
    # Importance near the middle/end of path should be higher than at the very start due to backward factor.
    assert imp[3].item() > imp[0].item()
    assert imp[4].item() > imp[1].item()
    # L1 norm equals 1 by design
    assert abs(float(torch.sum(imp).item()) - 1.0) < 1e-5


def test_bpn_unrelated_node_zero():
    # Create a chain 0->1->2->3 and an isolated node 4
    src = torch.tensor([0, 1, 2], dtype=torch.int64)
    dst = torch.tensor([1, 2, 3], dtype=torch.int64)
    g = dgl.graph((src, dst), num_nodes=5)
    start_nodes = [0]
    endpoint_node = 3

    bpn = BPNPropagator(steps_fwd=4, steps_bwd=4, decay_fwd=0.9, decay_bwd=0.9, combine="hadamard", out_norm="l1")
    imp = bpn.compute_endpoint_importance(g, start_nodes=start_nodes, endpoint_node=endpoint_node)
    # Isolated node should have exactly 0 importance (never reached by forward/backward)
    assert imp[4].item() == 0.0
    # Path nodes get positive mass
    assert float(torch.sum(imp[:4] > 0).item()) == 4