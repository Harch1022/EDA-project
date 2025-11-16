from __future__ import annotations
import re
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class PathRec(dict):
    startpoint: str
    endpoint: str
    slack: float
    group: Optional[str]
    arrival: Optional[float]


def _canonical_name(raw: str) -> str:
    """
    把 Startpoint/Endpoint 的原始字符串清洗成和 netlist 对得上的名字。

    例子：
      "i_Rx_Serial (input port clocked by core_clk)" -> "i_Rx_Serial"
      "_256_ (rising edge-triggered flip-flop clocked by core_clk)" -> "_256_"
      "u1/u2/reg_1_ (rising edge-triggered ...)" -> "u1/u2/reg_1_"
      "data[3] (in)" -> "data[3]"
    """
    s = raw.strip()
    # 先去掉括号后面的描述
    s = s.split('(')[0].strip()
    # 再只保留第一个 token（防止还有其他空格等）
    parts = s.split()
    if parts:
        s = parts[0]
    return s


def parse_report_checks(filepath: str) -> List[PathRec]:
    """
    Parse OpenROAD 'report_checks -path_delay min_max -format full_clock_expanded' report.

    兼容两种常见格式：
      1) data arrival time  0.1005
         slack (MET)        0.0234
      2) 0.1005   data arrival time
         0.0234   slack (MET)

    解析字段：
      - Startpoint   （已 canonicalize 成 netlist 名）
      - Endpoint     （已 canonicalize 成 netlist 名）
      - Path Group
      - slack
      - data arrival time（可选）
    另外保留：
      - startpoint_full / endpoint_full：原始带描述的字符串
    """
    recs: List[PathRec] = []
    cur: Optional[PathRec] = None

    re_start = re.compile(r'^\s*Startpoint:\s*(.+)$', re.IGNORECASE)
    re_end   = re.compile(r'^\s*Endpoint:\s*(.+)$', re.IGNORECASE)
    re_group = re.compile(r'^\s*Path Group:\s*(.+)$', re.IGNORECASE)

    # 通用浮点数匹配：支持 1, 1.0, -0.123, 1e-3 等
    re_float = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            # --- Startpoint ---
            m = re_start.match(line)
            if m:
                # 如果上一个 path 已经收集到完整信息，则先入列表
                if cur and 'startpoint' in cur and 'endpoint' in cur and 'slack' in cur:
                    recs.append(cur)

                cur = PathRec()
                raw_sp = m.group(1).strip()
                cur['startpoint_full'] = raw_sp
                cur['startpoint'] = _canonical_name(raw_sp)
                cur['group'] = None
                cur['arrival'] = None
                continue

            # --- Endpoint ---
            m = re_end.match(line)
            if m and cur is not None:
                raw_ep = m.group(1).strip()
                cur['endpoint_full'] = raw_ep
                cur['endpoint'] = _canonical_name(raw_ep)
                continue

            # --- Path Group ---
            m = re_group.match(line)
            if m and cur is not None:
                cur['group'] = m.group(1).strip()
                continue

            # 没有开始一个 path 的情况下，后面的 timing 信息忽略
            if cur is None:
                continue

            lower = line.lower()

            # --- slack 行：只要这一行包含 "slack" 就在里面找数字 ---
            if 'slack' in lower:
                mval = re_float.search(line)
                if mval:
                    try:
                        cur['slack'] = float(mval.group(0))
                    except ValueError:
                        cur['slack'] = 0.0
                continue

            # --- data arrival time 行 ---
            if 'data arrival time' in lower:
                mval = re_float.search(line)
                if mval:
                    try:
                        cur['arrival'] = float(mval.group(0))
                    except ValueError:
                        cur['arrival'] = None
                continue

    # 文件末尾，把最后一个 path 收尾
    if cur and 'startpoint' in cur and 'endpoint' in cur and 'slack' in cur:
        recs.append(cur)

    logger.info("Parsed timing report %s: %d paths", filepath, len(recs))
    return recs


def group_by_endpoint(paths: List[PathRec]) -> Dict[str, List[PathRec]]:
    g: Dict[str, List[PathRec]] = {}
    for p in paths:
        ep = p.get('endpoint', 'UNKNOWN')
        g.setdefault(ep, []).append(p)
    return g