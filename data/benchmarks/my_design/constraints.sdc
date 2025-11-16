# 定义一个 100MHz 时钟（周期 10ns），连到端口 i_Clock
create_clock -name core_clk -period 10 [get_ports i_Clock]

# 除了时钟外，其他输入的到达时间设为 0ns（相当于很宽松）
set_input_delay 0 -clock core_clk [all_inputs]

# 输出的要求时间也设为 0ns
set_output_delay 0 -clock core_clk [all_outputs]
