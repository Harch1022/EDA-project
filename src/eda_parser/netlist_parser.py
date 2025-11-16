from __future__ import annotations
import re
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

InstancePins = Dict[str, str]  # pin_name -> net_name
Instance = Dict[str, object]   # {'type': str, 'pins': InstancePins}
Instances = Dict[str, Instance]
NetToPins = Dict[str, List[Tuple[str, str]]]  # net -> list of (inst, pin)
Ports = Dict[str, str]  # net_name -> direction ("input"/"output"/"inout")

_PORT_DIR_RE = re.compile(r'^\s*(input|output|inout)\b([^;]*);', re.IGNORECASE)
_MODULE_RE = re.compile(r'^\s*module\s+(\w+)\b', re.IGNORECASE)
# 新增：匹配带端口列表的 module 头，例如 module top(input a, input b, output y);
_MODULE_HEADER_RE = re.compile(r'^\s*module\s+(\w+)\s*\((.*?)\)\s*;', re.IGNORECASE)
_ENDMODULE_RE = re.compile(r'^\s*endmodule\b', re.IGNORECASE)
_INST_RE = re.compile(
    r'^\s*([\w$]+)\s+([\w\d_/$\[\]\.]+)\s*\((.*?)\)\s*;\s*$',
    re.IGNORECASE | re.DOTALL,
)
_PIN_RE = re.compile(r'\.(\w+)\s*\(\s*([^\)]+)\s*\)')


def _strip_comments(text: str) -> str:
    """Remove // and /* */ comments from Verilog text."""
    text = re.sub(r'//.*', '', text)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    return text


def parse_gate_level_verilog(filepath: str) -> Tuple[Instances, NetToPins, Ports]:
    """
    Parse a (gate-level) Verilog netlist, extracting instances and net connectivity.
    Returns (instances, net_to_pins, ports).
    - instances: inst_name -> {'type': str, 'pins': {pin: net}}
    - net_to_pins: net_name -> [(inst_name, pin_name), ...]
    - ports: net_name -> direction ('input'|'output'|'inout') if parsed
    """
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        raw = f.read()
    txt = _strip_comments(raw)

    # normalize multi-line instances: replace inner newlines with spaces
    buf = []
    depth = 0
    for ch in txt:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth = max(depth - 1, 0)
        if ch == '\n' and depth > 0:
            buf.append(' ')
        else:
            buf.append(ch)
    normalized = ''.join(buf)
    normalized_lines = normalized.splitlines()

    instances: Instances = {}
    net_to_pins: NetToPins = {}
    ports: Ports = {}
    in_module = False

    for ln in normalized_lines:
        # 先处理带端口列表的 module 头
        m_header = _MODULE_HEADER_RE.match(ln)
        if m_header:
            in_module = True
            ports_blob = m_header.group(2)  # 括号里的内容：input a, input b, output y

            # 按逗号拆分每一段 "input a" / "output y"
            for part in ports_blob.split(','):
                part = part.strip()
                if not part:
                    continue

                # 匹配方向 + 其余内容
                m_dir = re.match(r'(input|output|inout)\s+(.+)', part, re.IGNORECASE)
                if not m_dir:
                    continue

                direction = m_dir.group(1).lower()
                rest = m_dir.group(2)

                # 去掉总线范围 [3:0] 之类
                rest = re.sub(r'\[[^]]+\]', ' ', rest)

                # 对于 "wire a" / "reg a" 之类，只取最后一个 token 作为端口名
                tokens = [t for t in re.split(r'[,\s]+', rest) if t]
                if not tokens:
                    continue
                name = tokens[-1]
                ports[name] = direction

            # 已处理完这一行，继续下一行
            continue

        # 其他 module 行（没有端口列表的情况）
        if _MODULE_RE.match(ln):
            in_module = True
            continue

        if _ENDMODULE_RE.match(ln):
            in_module = False
            continue

        if not in_module:
            continue

        # 解析 module 体内的端口声明，例如 "input a, b;"
        mdir = _PORT_DIR_RE.match(ln)
        if mdir:
            direction = mdir.group(1).lower()
            nets_part = mdir.group(2)
            # 去掉总线范围 [3:0]
            nets_part = re.sub(r'\[[^\]]+\]', ' ', nets_part)
            nets = [
                n.strip()
                for n in re.split(r'[,\s]+', nets_part)
                if n.strip() and n.strip() != ';'
            ]
            for n in nets:
                ports[n] = direction
            continue

        # 解析实例
        minst = _INST_RE.match(ln)
        if minst:
            cell_type = minst.group(1)
            inst_name = minst.group(2)
            pins_blob = minst.group(3)
            pins: InstancePins = {}
            for pin, net in _PIN_RE.findall(pins_blob):
                net = net.strip().replace('{', '').replace('}', '')
                pins[pin] = net
                net_to_pins.setdefault(net, []).append((inst_name, pin))
            instances[inst_name] = {'type': cell_type, 'pins': pins}

    logger.info(
        "Parsed netlist %s: %d instances, %d nets, %d ports",
        filepath,
        len(instances),
        len(net_to_pins),
        len(ports),
    )
    return instances, net_to_pins, ports