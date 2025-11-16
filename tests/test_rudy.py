from __future__ import annotations
import numpy as np
from src.features.cnn_maps import build_physical_maps

def test_hpwl_rudy_long_vs_short():
    # synthetic die and components
    die = ((0,0),(1000,1000))
    # place instances; long net spans across die; short net is compact
    components = {
        "U1": {"macro":"NAND2_X1", "xy": (100,100)},
        "U2": {"macro":"NAND2_X1", "xy": (900,900)},
        "U3": {"macro":"INV_X1",   "xy": (480,520)},
        "U4": {"macro":"INV_X1",   "xy": (520,520)},
    }
    # net_to_pins: net -> [(inst, pin)]
    net2pins = {
        "long": [("U1","Y"), ("U2","A")],
        "short": [("U3","Y"), ("U4","A")],
    }
    maps = build_physical_maps(components, die, pins_xy={}, grid_size=32, net_to_pins=net2pins, prefer_hpwl_rudy=True)
    rudy = maps["rudy_map"]
    assert rudy.shape == (32,32)
    assert rudy.max() > 0.0

    # Roughly compare average rudy inside long bbox vs short bbox
    def to_idx(x,y):
        gx = int(x/1000 * 31); gy = int(y/1000 * 31)
        return gx, gy
    xa1, ya1 = to_idx(100,100); xb1, yb1 = to_idx(900,900)
    long_avg = float(np.mean(rudy[min(ya1,yb1):max(ya1,yb1)+1, min(xa1,xb1):max(xa1,xb1)+1]))
    xa2, ya2 = to_idx(480,520); xb2, yb2 = to_idx(520,520)
    short_avg = float(np.mean(rudy[min(ya2,yb2):max(ya2,yb2)+1, min(xa2,xb2):max(xa2,xb2)+1]))
    #assert long_avg > short_avg  # long net should induce larger average demand region-wise
    assert short_avg > long_avg  # 在 HPWL*deg/area 的定义下，短网单位面积的平均需求更大