from __future__ import annotations
from src.labels.cpl import compute_cpl_labels

def test_ratio_to_worst_by_arrival():
    # Construct toy by_endpoint with arrivals
    by_ep = {
        "E1": [ {"startpoint":"S1","arrival":10.0,"slack":-0.10},
                {"startpoint":"S2","arrival": 9.8,"slack":-0.08} ],
        "E2": [ {"startpoint":"S3","arrival": 8.0,"slack": 0.05} ],
        "E3": [ {"startpoint":"S4","arrival": 5.0,"slack": 0.10} ],
    }
    near = compute_cpl_labels(by_ep,
                              mode="ratio_to_worst",
                              ratio_metric="arrival",
                              top_ratio=1/3,     # only top 1 endpoint should be selected: E1
                              inner_mode="best_only")
    assert set(near.keys()) == {"E1"}
    assert len(near["E1"]) == 1
    assert near["E1"][0]["startpoint"] in ("S1","S2")  # best_only picks worst arrival among E1

def test_ratio_to_worst_by_slack_keep_non_selected():
    # E1 worst (more negative), E2 medium, E3 best
    by_ep = {
        "E1": [ {"startpoint":"A","arrival":5.0,"slack":-0.50},
                {"startpoint":"B","arrival":5.2,"slack":-0.40} ],
        "E2": [ {"startpoint":"C","arrival":6.0,"slack":-0.10} ],
        "E3": [ {"startpoint":"D","arrival":7.0,"slack": 0.00} ],
    }
    near = compute_cpl_labels(by_ep,
                              mode="ratio_to_worst",
                              ratio_metric="slack",
                              top_ratio=2/3,    # keep E1/E2
                              inner_mode="delta_abs",
                              delta_ns=0.1,
                              keep_non_selected=True)
    # E1 and E2 selected; E3 kept due to keep_non_selected
    assert set(near.keys()) == {"E1","E2","E3"}
    # E3 has only best_only 1 path label under keep_non_selected
    assert len(near["E3"]) == 1