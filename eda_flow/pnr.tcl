# -----------------------------------------------------------
# pnr.tcl - OpenROAD Place & Route Script (容错 + 报告导出)
# -----------------------------------------------------------

# 工艺/库/输入网表（按需修改）
# 允许环境变量覆盖，默认回退到 OpenROAD-flow-scripts 的 Nangate45
proc env_or {name default} {
  if {[info exists ::env($name)]} { return $::env($name) } { return $default }
}

set DESIGN_NAME [env_or DESIGN_NAME "my_design"]
set PROJECT_DIR [env_or PROJECT_DIR "/home/lzz_linux/fyp-project"]
set OUT_DIR     [env_or OUT_DIR     "$PROJECT_DIR/data/raw_eda/$DESIGN_NAME"]
set SKIP_CTS    [expr {[info exists ::env(SKIP_CTS)] && $::env(SKIP_CTS) ne ""}]

# 新增：允许使用 eda_flow/lib 路径（与 run_flow.sh 保持一致），若未设置则回退到 OpenROAD-flow-scripts
# 工艺/库/输入网表（按需修改）
set LIB_LIB   [env_or LIB_FILE  "$PROJECT_DIR/tools/OpenROAD-flow-scripts/flow/platforms/nangate45/lib/NangateOpenCellLibrary_typical.lib"]
set LEF_TECH  [env_or LEF_TECH  "$PROJECT_DIR/tools/OpenROAD-flow-scripts/flow/platforms/nangate45/lef/NangateOpenCellLibrary.tech.lef"]
set LEF_STD   [env_or LEF_STD   ""]
set LEF_MACRO [env_or LEF_MACRO "$PROJECT_DIR/tools/OpenROAD-flow-scripts/flow/platforms/nangate45/lef/NangateOpenCellLibrary.macro.lef"]
set NETLIST   "$OUT_DIR/$DESIGN_NAME.synth.v"

if {![file exists $OUT_DIR]} { file mkdir $OUT_DIR }

puts "==> Loading liberty/lef ..."
read_liberty $LIB_LIB
read_lef     $LEF_TECH
if {$LEF_STD ne ""} {
  read_lef   $LEF_STD
} else {
  read_lef   $LEF_MACRO
}

# ---------------- 2) 读取综合网表并链接 ----------------
puts "==> Reading synthesized netlist: $NETLIST"
read_verilog $NETLIST
link_design $DESIGN_NAME

# ---------------- 3) 约束：优先读取 SDC，回退自动识别 ----------------
if {[info exists ::env(SDC_FILE)] && [file exists $::env(SDC_FILE)]} {
  puts "==> Reading SDC: $::env(SDC_FILE)"
  read_sdc $::env(SDC_FILE)
} else {
  set clk_port ""
  foreach p {i_Clock clk clock clk_i clk_in clk0} {
    if {[llength [get_ports -quiet $p]]} { set clk_port $p; break }
  }
  if {$clk_port eq ""} {
    puts "WARN: No clock port found among {i_Clock clk clock clk_i clk_in clk0}; design will be unconstrained."
  } else {
    create_clock -name core_clk -period 10.0 [get_ports $clk_port]
    puts "INFO: Created clock 'core_clk' (10ns) on port '$clk_port'."
  }
}

# ---------------- 4) Floorplan ----------------
set place_site FreePDK45_38x28_10R_NP_162NW_34O
puts "==> Floorplan with site: $place_site"
initialize_floorplan -site $place_site -utilization 50 -aspect_ratio 1.0 -core_space 10

# 轨道与 IO 引脚
make_tracks
set hor_layers {metal3 metal5}
set ver_layers {metal2 metal4}
place_pins -hor_layers $hor_layers -ver_layers $ver_layers -corner_avoidance 10 -min_distance 2.0

# 允许的布线层（不同版本可能没有该命令）
if {[catch { set_routing_layers -signal metal1-metal9 -clock metal1-metal9 } msg]} {
  puts "INFO: set_routing_layers not available or failed: $msg"
}

# ---------------- 5) 初始放置 ----------------
puts "==> Global & detailed placement"
global_placement -density 0.7
detailed_placement

# 预布线导出
catch { write_def $OUT_DIR/$DESIGN_NAME.pre_route.def }

# 预估 RC（基于放置）
set_wire_rc -signal -layer metal4
if {[llength [all_clocks]]} {
  set_wire_rc -clock -layer metal4
}
estimate_parasitics -placement

# 预布局时序报告（保存到文件 + 控制台摘要）
set pre_rpt "$OUT_DIR/$DESIGN_NAME.pre_route_timing.rpt"
puts "==> Pre-route timing reports -> $pre_rpt"

# 使用 sta::redirect_file_begin / end 把输出写到文件
sta::redirect_file_begin $pre_rpt
puts "==== Pre-route timing (report_checks) ===="
report_checks -path_delay min_max -fields {slew cap input_pins} -digits 4 -format full_clock_expanded
puts "\n---- Pre-route Summary ----"
report_worst_slack -max
report_worst_slack -min
report_tns -max
report_tns -min
sta::redirect_file_end

# 控制台简要摘要（再跑一次，只在终端显示）
report_worst_slack -max
report_worst_slack -min

# ---------------- 6) CTS + 再放置 ----------------
if {$SKIP_CTS} {
  puts "INFO: SKIP_CTS is set; skipping clock_tree_synthesis."
} else {
  puts "==> Clock Tree Synthesis"
  if {[catch { clock_tree_synthesis } msg]} {
    puts "INFO: clock_tree_synthesis not available or failed: $msg"
    if {[catch { run_cts } msg2]} {
      puts "ERROR: CTS failed with both commands. Details: $msg / $msg2"
      exit 1
    }
  }
  puts "INFO: Post-CTS incremental placement..."
  if {[catch { global_placement -density 0.7 -incremental } emsg]} {
    puts "WARN: global_placement -incremental failed or not supported: $emsg"
  }
  detailed_placement

  if {[llength [all_clocks]]} {
    set_propagated_clock [all_clocks]
  }
}

# ---------------- 7) 全局/详细布线 ----------------
puts "==> Global route"
global_route

puts "==> Detailed route"
# 兼容不同版本的 DRT DRC 报告参数
if {[catch { detailed_route -output_drc $OUT_DIR/$DESIGN_NAME.drc.rpt } drt_msg]} {
  puts "INFO: detailed_route -output_drc not supported: $drt_msg"
  detailed_route
}

# ---------------- 8) （可选）后布线优化（版本兼容） ----------------
puts "==> Post-route optimization (if available)"
if {[catch { repair_timing -post_route } rt_msg]} {
  puts "WARN: repair_timing -post_route not available or failed: $rt_msg"
  if {[catch { repair_timing -setup } rts_msg]} {
    puts "INFO: repair_timing -setup not supported or skipped: $rts_msg"
  }
  if {[catch { repair_timing -hold } rth_msg]} {
    puts "INFO: repair_timing -hold not supported or skipped: $rth_msg"
  }
}

# ---------------- 9) 后布线 RC 估计（版本探测 + 回退） ----------------
puts "==> Estimating post-route parasitics"
set used_ep_mode ""
if {[catch { estimate_parasitics -routing } ep_msg]} {
  puts "INFO: estimate_parasitics -routing not supported: $ep_msg"
  if {[catch { estimate_parasitics -global_routing } ep2_msg]} {
    puts "INFO: estimate_parasitics -global_routing not supported: $ep2_msg"
    puts "INFO: Falling back to estimate_parasitics -placement."
    estimate_parasitics -placement
    set used_ep_mode "placement"
  } else {
    set used_ep_mode "global_routing"
  }
} else {
  set used_ep_mode "routing"
}
puts "INFO: Post-route parasitics mode used: $used_ep_mode"

# ---------------- 10) 后布线时序报告 ----------------
set post_rpt "$OUT_DIR/$DESIGN_NAME.post_route_timing.rpt"
puts "==> Post-route timing reports -> $post_rpt"

sta::redirect_file_begin $post_rpt
puts "==== Post-route timing (report_checks) ===="
report_checks -path_delay min_max -fields {slew cap input_pins} -digits 4 -format full_clock_expanded
puts "\n---- Post-route Summary ----"
report_worst_slack -max
report_worst_slack -min
report_tns -max
report_tns -min
sta::redirect_file_end

# 控制台简要摘要
report_worst_slack -max
report_worst_slack -min

# ---------------- 11) 导出结果 ----------------
puts "==> Writing results"
catch { write_def     $OUT_DIR/$DESIGN_NAME.pre_route.def }    ;# 若之前已写，不报错
catch { write_def     $OUT_DIR/$DESIGN_NAME.post_route.def }
catch { write_verilog $OUT_DIR/$DESIGN_NAME.post_route.v }

# SPEF 可能依赖于 RC 建模；做容错
if {[catch { write_spef $OUT_DIR/$DESIGN_NAME.post_route.spef } spef_msg]} {
  puts "WARN: write_spef failed or not supported: $spef_msg"
}

puts "==> PnR flow finished. Outputs in $OUT_DIR"
exit