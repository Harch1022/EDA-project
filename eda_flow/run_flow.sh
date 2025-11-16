#!/usr/bin/env bash
set -euo pipefail

export DESIGN_NAME="${DESIGN_NAME:-my_design}"
export TOP_MODULE="${TOP_MODULE:-$DESIGN_NAME}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PROJECT_ROOT

export DESIGN_DIR="$PROJECT_ROOT/data/benchmarks/$DESIGN_NAME"
export OUT_DIR="$PROJECT_ROOT/data/raw_eda/$DESIGN_NAME"
mkdir -p "$OUT_DIR"

# 二进制回退：优先系统PATH，其次项目内 build 产物
YOSYS_BIN="${YOSYS_BIN:-yosys}"
if ! command -v "$YOSYS_BIN" >/dev/null 2>&1; then
  if [ -x "$PROJECT_ROOT/tools/yosys/yosys" ]; then
    YOSYS_BIN="$PROJECT_ROOT/tools/yosys/yosys"
  fi
fi
OPENROAD_BIN="${OPENROAD_BIN:-openroad}"
if ! command -v "$OPENROAD_BIN" >/dev/null 2>&1; then
  if [ -x "$PROJECT_ROOT/tools/OpenROAD/build/src/openroad" ]; then
    OPENROAD_BIN="$PROJECT_ROOT/tools/OpenROAD/build/src/openroad"
  fi
fi

# N45 库路径：优先 eda_flow/lib；若不存在则回退 ORFS 路径
ORFS="$PROJECT_ROOT/tools/OpenROAD-flow-scripts/flow/platforms/nangate45"
EDA_LIB_DIR="$PROJECT_ROOT/eda_flow/lib"
mkdir -p "$EDA_LIB_DIR"

LIB_CAND1="$EDA_LIB_DIR/typical.lib"
LIB_CAND2="$ORFS/lib/NangateOpenCellLibrary_typical.lib"
LEF_TECH_CAND1="$EDA_LIB_DIR/tech.lef"
LEF_TECH_CAND2="$ORFS/lef/NangateOpenCellLibrary.tech.lef"
LEF_STD_CAND1="$EDA_LIB_DIR/stdcells.lef"
LEF_STD_CAND2="$ORFS/lef/NangateOpenCellLibrary.macro.lef"

# 如果 eda_flow/lib 下缺失，且 ORFS 存在，则创建软链接
if [ ! -f "$LIB_CAND1" ] && [ -f "$LIB_CAND2" ]; then ln -sf "$LIB_CAND2" "$LIB_CAND1"; fi
if [ ! -f "$LEF_TECH_CAND1" ] && [ -f "$LEF_TECH_CAND2" ]; then ln -sf "$LEF_TECH_CAND2" "$LEF_TECH_CAND1"; fi
if [ ! -f "$LEF_STD_CAND1" ] && [ -f "$LEF_STD_CAND2" ]; then ln -sf "$LEF_STD_CAND2" "$LEF_STD_CAND1"; fi

export LIB_FILE="${LIB_FILE:-$LIB_CAND1}"
export LEF_TECH="${LEF_TECH:-$LEF_TECH_CAND1}"
export LEF_STD="${LEF_STD:-$LEF_STD_CAND1}"

# RTL/SDC 路径：SDC 兼容 constraints.sdc 与 <design>.sdc；若缺失将自动创建默认 SDC
export VERILOG_GLOB="${VERILOG_GLOB:-$DESIGN_DIR/*.v}"
SDC_DEFAULT_A="$DESIGN_DIR/constraints.sdc"
SDC_DEFAULT_B="$DESIGN_DIR/$DESIGN_NAME.sdc"
if [ -z "${SDC_FILE:-}" ]; then
  if [ -f "$SDC_DEFAULT_A" ]; then
    export SDC_FILE="$SDC_DEFAULT_A"
  elif [ -f "$SDC_DEFAULT_B" ]; then
    export SDC_FILE="$SDC_DEFAULT_B"
  else
    export SDC_FILE="$SDC_DEFAULT_A"
  fi
fi

# 若 SDC 不存在，自动生成一个最简约束
if [ ! -f "$SDC_FILE" ]; then
  cat > "$SDC_FILE" <<'SDC'
# Auto-generated minimal SDC
create_clock -name core_clk -period 10 [get_ports i_Clock]
set_input_delay 0 -clock core_clk [remove_from_collection [all_inputs] [get_ports i_Clock]]
set_output_delay 0 -clock core_clk [all_outputs]
SDC
  echo "[INFO] Default SDC created at $SDC_FILE"
fi

export SYNTH_NETLIST="$OUT_DIR/$DESIGN_NAME.synth.v"

echo "=== FYP Flow ==="
echo "DESIGN: $DESIGN_NAME  TOP: $TOP_MODULE"
echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "OUT_DIR: $OUT_DIR"
echo "LIB_FILE: $LIB_FILE"
echo "LEF_TECH: $LEF_TECH"
echo "LEF_STD:  $LEF_STD"
echo "SDC_FILE: $SDC_FILE"
echo "---------------------------------------"

# 版本信息
mkdir -p "$OUT_DIR"
if [ -f "$PROJECT_ROOT/tools/capture_versions.sh" ]; then
  bash "$PROJECT_ROOT/tools/capture_versions.sh" > "$OUT_DIR/versions.txt" 2>&1 || true
fi

# 收集 Verilog 列表
shopt -s nullglob
mapfile -t VFILES < <(compgen -G "$VERILOG_GLOB")
if [ "${#VFILES[@]}" -eq 0 ]; then
  echo "ERROR: no Verilog matched: $VERILOG_GLOB"
  exit 1
fi

# 生成 Yosys 脚本（.ys）
YS_FILE="$OUT_DIR/synth.ys"
{
  echo "read_liberty -lib \"$LIB_FILE\""
  for f in "${VFILES[@]}"; do
    echo "read_verilog -sv \"$f\""
  done
  echo "hierarchy -check -top $TOP_MODULE"
  echo "proc; opt; fsm; opt"
  echo "techmap; opt"
  echo "synth -top $TOP_MODULE"
  echo "dfflibmap -liberty \"$LIB_FILE\""
  echo "abc -liberty \"$LIB_FILE\""
  echo "opt_clean -purge"
  echo "write_verilog -noattr \"$SYNTH_NETLIST\""
} > "$YS_FILE"

# 1) 综合
echo "[1/3] Synthesis (Yosys)"
"$YOSYS_BIN" -l "$OUT_DIR/yosys.log" -s "$YS_FILE"
test -s "$SYNTH_NETLIST" || { echo "ERROR: synthesis failed (netlist missing)."; exit 1; }

# 2) PnR
echo "[2/3] Place & Route (OpenROAD)"
export SDC_FILE  # 传入 pnr.tcl
"$OPENROAD_BIN" -no_init -exit "$PROJECT_ROOT/eda_flow/pnr.tcl" 2>&1 | tee "$OUT_DIR/openroad.log"

# 3) 元数据
echo "[3/3] Write metadata"
python3 - <<'PY'
import os, json, re, time
out=os.environ["OUT_DIR"]; dn=os.environ["DESIGN_NAME"]
def E(p): return os.path.join(out,p)
md={
 "design_name": dn,
 "top_module": os.environ.get("TOP_MODULE",""),
 "inputs": {"verilog_glob": os.environ.get("VERILOG_GLOB",""),
            "sdc": os.environ.get("SDC_FILE","")},
 "lib": {"lib": os.environ.get("LIB_FILE",""),
         "lef_tech": os.environ.get("LEF_TECH",""),
         "lef_std": os.environ.get("LEF_STD","")},
 "artifacts": sorted([f for f in os.listdir(out) if os.path.isfile(E(f))]),
 "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
 "tool_versions": open(E("versions.txt"),"r",errors="ignore").read() if os.path.exists(E("versions.txt")) else ""
}
drcp = E(f"{dn}.drc.rpt")
if os.path.exists(drcp):
  txt=open(drcp,"r",errors="ignore").read()
  md["drc_violations"]= len(re.findall(r"violation", txt, flags=re.I))
open(E("metadata.json"),"w").write(json.dumps(md, indent=2))
print("metadata.json written")
PY

echo "DONE. Outputs in $OUT_DIR"