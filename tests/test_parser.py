from __future__ import annotations
from src.eda_parser.netlist_parser import parse_gate_level_verilog, _strip_comments
from src.eda_parser.timing_parser import parse_report_checks
from src.eda_parser.def_parser import parse_def

def test_strip_comments():
    s = "a; // comment\n/* block */ b;"
    assert _strip_comments(s) == "a; \n b;"

def test_parse_netlist_tmp(tmp_path):
    v = """
    module top(input a, input b, output y);
    NAND2_X1 U1 (.A(a), .B(b), .Y(n1));
    INV_X1 U2 (.A(n1), .Y(y));
    endmodule
    """
    p = tmp_path/"t.v"
    p.write_text(v)
    insts, net2pins, ports = parse_gate_level_verilog(str(p))
    assert "U1" in insts and "U2" in insts
    assert ports.get("a") == "input" and ports.get("y") == "output"

def test_parse_def_tmp(tmp_path):
    d = """VERSION 5.8 ;
    DIEAREA ( 0 0 ) ( 1000 1000 ) ;
    COMPONENTS 1 ;
    - U1 NAND2_X1 + PLACED ( 100 200 ) N ;
    END COMPONENTS
    PINS 1 ;
    - clk + NET clk + DIRECTION INPUT + PLACED ( 500 0 ) N ;
    END PINS
    END DESIGN
    """
    p = tmp_path/"t.def"
    p.write_text(d)
    comps, die, pins = parse_def(str(p))
    assert die == ((0,0),(1000,1000))
    assert "U1" in comps and comps["U1"]["xy"] == (100,200)
    assert "clk" in pins

def test_parse_timing_tmp(tmp_path):
    r = """
Startpoint: U1/Q (rising edge)
Endpoint: U2/D (rising edge)
Path Group: core_clk
data arrival time 1.234
slack (MET) 0.100

Startpoint: U3/Q (rising edge)
Endpoint: U2/D (rising edge)
slack (VIOLATED) -0.020
    """
    p = tmp_path/"t.rpt"
    p.write_text(r)
    recs = parse_report_checks(str(p))
    assert len(recs) >= 2
    assert any(abs(x.get('slack',0)-0.1) < 1e-6 for x in recs)
    assert any(abs(x.get('slack',0)+0.02) < 1e-6 for x in recs)