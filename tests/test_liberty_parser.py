from __future__ import annotations
from src.eda_parser.liberty_parser import parse_liberty_pin_directions, build_cell_output_index

def test_parse_liberty_basic(tmp_path):
    lib = """
library(my_lib) {
  cell(NAND2_X1) {
    pin(A) { direction : input; }
    pin(B) { direction : input; }
    pin(Y) { direction : output; }
  }
  cell(DFF_X1) {
    pin(D) { direction : input; }
    pin(CK) { direction : input; }
    pin(Q) { direction : output; }
    pg_pin(VDD) { voltage_name : "VDD"; }
    pg_pin(VSS) { voltage_name : "VSS"; }
  }
}
"""
    p = tmp_path/"typ.lib"
    p.write_text(lib)
    pin_dirs = parse_liberty_pin_directions(str(p))
    assert "NAND2_X1" in pin_dirs and pin_dirs["NAND2_X1"]["Y"] == "output"
    assert "DFF_X1" in pin_dirs and pin_dirs["DFF_X1"]["Q"] == "output"
    out_idx = build_cell_output_index(pin_dirs)
    assert out_idx["NAND2_X1"] == {"Y"}
    assert "Q" in out_idx["DFF_X1"]