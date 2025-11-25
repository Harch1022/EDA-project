# -----------------------------------------------------------
# timing_only.tcl - 只基于现有 PnR 结果做 STA + report_checks
# -----------------------------------------------------------

# 从环境变量读取值，如果不存在就用默认
proc env_or {name default} {
  if {[info exists ::env($name)]} {
    return $::env($name)
  } else {
    return $default
  }
}

# 统一封装 report_checks，兼容你当前 OpenROAD 版本的参数
proc report_checks_extended {} {
  # 先输出 max 路径
  puts "---- report_checks (path_delay max) ----"
  report_checks \
    -path_delay max \
    -format full_clock_expanded \
    -fields {capacitance slew input_pin} \
    -digits 4 \
    -group_path_count 2000 \
    -endpoint_path_count 1 \
    -unique_paths_to_endpoint

  # 再输出 min 路径
  puts "---- report_checks (path_delay min) ----"
  report_checks \
    -path_delay min \
    -format full_clock_expanded \
    -fields {capacitance slew input_pin} \
    -digits 4 \
    -group_path_count 2000 \
    -endpoint_path_count 1 \
    -unique_paths_to_endpoint
}

# 设计 / 工程目录 / 输出目录
set DESIGN_NAME [env_or DESIGN_NAME "my_design"]
set PROJECT_DIR [env_or PROJECT_DIR "/home/lzz_linux/fyp-project"]
set OUT_DIR     [env_or OUT_DIR     "$PROJECT_DIR/data/raw_eda/$DESIGN_NAME"]
set TOP_MODULE  [env_or TOP_MODULE  $DESIGN_NAME]

# 工艺/库路径（和 pnr.tcl 保持一致）
set LIB_LIB   [env_or LIB_FILE  "$PROJECT_DIR/tools/OpenROAD-flow-scripts/flow/platforms/nangate45/lib/NangateOpenCellLibrary_typical.lib"]
set LEF_TECH  [env_or LEF_TECH  "$PROJECT_DIR/tools/OpenROAD-flow-scripts/flow/platforms/nangate45/lef/NangateOpenCellLibrary.tech.lef"]
set LEF_STD   [env_or LEF_STD   ""]
set LEF_MACRO [env_or LEF_MACRO "$PROJECT_DIR/tools/OpenROAD-flow-scripts/flow/platforms/nangate45/lef/NangateOpenCellLibrary.macro.lef"]

puts "==> Timing-only STA starting"
puts "    DESIGN_NAME : $DESIGN_NAME"
puts "    TOP_MODULE  : $TOP_MODULE"
puts "    PROJECT_DIR : $PROJECT_DIR"
puts "    OUT_DIR     : $OUT_DIR"

# 1) 读库
read_liberty $LIB_LIB
read_lef     $LEF_TECH
if {$LEF_STD ne ""} {
  read_lef   $LEF_STD
} else {
  read_lef   $LEF_MACRO
}

# 2) 优先尝试读 DEF（有的话就直接用 DEF 驱动 STA）
set DEF_POST "$OUT_DIR/$DESIGN_NAME.post_route.def"
set DEF_PRE  "$OUT_DIR/$DESIGN_NAME.pre_route.def"
set has_def 0

if {[file exists $DEF_POST]} {
  puts "==> Reading DEF: $DEF_POST"
  read_def $DEF_POST
  set has_def 1
} elseif {[file exists $DEF_PRE]} {
  puts "==> Reading DEF: $DEF_PRE"
  read_def $DEF_PRE
  set has_def 1
} else {
  puts "INFO: No DEF found; will fall back to netlist-based STA."
}

# 3) 如果没有 DEF，则退回到 netlist + link_design 模式
if {!$has_def} {
  set NET_SYN  "$OUT_DIR/$DESIGN_NAME.synth.v"
  set NET_POST "$OUT_DIR/$DESIGN_NAME.post_route.v"
  if {[file exists $NET_POST]} {
    set NETLIST $NET_POST
  } else {
    set NETLIST $NET_SYN
  }
  puts "==> Reading netlist (no DEF present): $NETLIST"
  read_verilog $NETLIST
  link_design $TOP_MODULE
}

# 4) 约束：优先用外部 SDC，否则自动找时钟端口并建一个 10ns 的时钟
if {[info exists ::env(SDC_FILE)] && [file exists $::env(SDC_FILE)]} {
  puts "==> Reading SDC: $::env(SDC_FILE)"
  read_sdc $::env(SDC_FILE)
} else {
  set clk_port ""
  foreach p {i_Clock clk clock clk_i clk_in clk0} {
    if {[llength [get_ports -quiet $p]]} {
      set clk_port $p
      break
    }
  }
  if {$clk_port eq ""} {
    puts "WARN: No clock port found among {i_Clock clk clock clk_i clk_in clk0}; design will be unconstrained."
  } else {
    create_clock -name core_clk -period 10.0 [get_ports $clk_port]
    puts "INFO: Created clock 'core_clk' (10ns) on port '$clk_port'."
  }
}

# 5) 读 SPEF（如果有的话可以拿到更真实的 RC）
set SPEF "$OUT_DIR/$DESIGN_NAME.post_route.spef"
if {[file exists $SPEF]} {
  puts "==> Reading SPEF: $SPEF"
  read_spef $SPEF
} else {
  puts "INFO: No SPEF found; using internal RC estimation."
}

# 6) 时钟传播
if {[llength [all_clocks]]} {
  set_propagated_clock [all_clocks]
}

# 7) 输出 timing 报告（覆盖原来的 post_route_timing.rpt）
set post_rpt "$OUT_DIR/$DESIGN_NAME.post_route_timing.rpt"
puts "==> Writing post-route timing report -> $post_rpt"

sta::redirect_file_begin $post_rpt
puts "==== Post-route timing (report_checks) ===="
report_checks_extended
puts "\n---- Post-route Summary ----"
report_worst_slack -max
report_worst_slack -min
report_tns -max
report_tns -min
sta::redirect_file_end

puts "==> Timing-only STA done."
exit