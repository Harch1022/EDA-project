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
    arrival: Optional[float]        # 终点处的 path arrival（保留旧语义）
    start_arrival: Optional[float]  # 新增：startpoint 的真实到达时间（NUIAT）


_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


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
    s = s.split("(")[0].strip()
    # 再只保留第一个 token（防止还有其他空格等）
    parts = s.split()
    if parts:
        s = parts[0]
    return s


def _extract_first_float(text: str) -> Optional[float]:
    m = _FLOAT_RE.search(text)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def _extract_last_float(text: str) -> Optional[float]:
    vals = _FLOAT_RE.findall(text)
    if not vals:
        return None
    try:
        return float(vals[-1])
    except ValueError:
        return None


def _line_matches_point_row(line: str, point_name: str) -> bool:
    """
    判断 timing table 当前行是否就是某个 point 的那一行。
    这里用“前缀 + 分隔符”做一个尽量稳妥的匹配，适配：
      - foo
      - foo (in)
      - foo/Q
      - foo[3]
    """
    if not point_name:
        return False

    s = line.lstrip()
    if not s.startswith(point_name):
        return False

    if len(s) == len(point_name):
        return True

    nxt = s[len(point_name)]
    return nxt.isspace() or nxt in "([/"


def _maybe_update_start_arrival(cur: PathRec, line: str) -> None:
    """
    尝试从 path 展开内容里提取 startpoint 的真实到达时间。

    优先级：
      1) 精确命中 startpoint 对应的 timing table 行
      2) 退而求其次：命中 'input external delay' / 'input delay' / 'input arrival time'
    """
    lower = line.lower()
    sp_name = str(cur.get("startpoint", ""))

    # 优先：startpoint 自己那一行（最可信）
    if _line_matches_point_row(line, sp_name):
        cand = _extract_last_float(line)
        if cand is not None:
            cur["start_arrival"] = cand
            return

    # 次优：针对 primary input，很多报告会单独给 external delay
    if cur.get("start_arrival") is None:
        if (
            "input external delay" in lower
            or "input delay" in lower
            or "input arrival time" in lower
        ):
            cand = _extract_last_float(line)
            if cand is not None:
                cur["start_arrival"] = cand


def _flush_current_record(cur: Optional[PathRec], recs: List[PathRec]) -> None:
    if cur and "startpoint" in cur and "endpoint" in cur and "slack" in cur:
        recs.append(cur)


def parse_report_checks(filepath: str) -> List[PathRec]:
    """
    Parse OpenROAD 'report_checks -path_delay min_max -format full_clock_expanded' report.

    兼容两类信息：
      1) 终点处 path arrival（原 arrival 字段，保持原逻辑不变）
      2) 新增 startpoint 的真实到达时间 start_arrival（用于 NUIAT）

    支持常见格式：
      - data arrival time  0.1005
      - 0.1005   data arrival time
      - timing table 中 startpoint 对应行的累计时间
      - input external delay 对应累计时间
    """
    recs: List[PathRec] = []
    cur: Optional[PathRec] = None

    re_start = re.compile(r"^\s*Startpoint:\s*(.+)$", re.IGNORECASE)
    re_end = re.compile(r"^\s*Endpoint:\s*(.+)$", re.IGNORECASE)
    re_group = re.compile(r"^\s*Path Group:\s*(.+)$", re.IGNORECASE)

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            # --- Startpoint ---
            m = re_start.match(line)
            if m:
                _flush_current_record(cur, recs)

                cur = PathRec()
                raw_sp = m.group(1).strip()
                cur["startpoint_full"] = raw_sp
                cur["startpoint"] = _canonical_name(raw_sp)
                cur["group"] = None
                cur["arrival"] = None
                cur["start_arrival"] = None
                continue

            # --- Endpoint ---
            m = re_end.match(line)
            if m and cur is not None:
                raw_ep = m.group(1).strip()
                cur["endpoint_full"] = raw_ep
                cur["endpoint"] = _canonical_name(raw_ep)
                continue

            # --- Path Group ---
            m = re_group.match(line)
            if m and cur is not None:
                cur["group"] = m.group(1).strip()
                continue

            # 没有开始一个 path 的情况下，后面的 timing 信息忽略
            if cur is None:
                continue

            lower = line.lower()

            # -------------------------
            # 改动点 1：尽量提取 start_arrival
            # -------------------------
            _maybe_update_start_arrival(cur, line)

            # --- slack 行：只要这一行包含 "slack" 就在里面找数字 ---
            if "slack" in lower:
                v = _extract_first_float(line)
                if v is not None:
                    cur["slack"] = v
                else:
                    cur["slack"] = 0.0
                continue

            # --- data arrival time 行：这是 path 到 endpoint 的 arrival，保留旧语义 ---
            if "data arrival time" in lower:
                v = _extract_first_float(line)
                if v is not None:
                    cur["arrival"] = v
                else:
                    cur["arrival"] = None
                continue

    # 文件末尾收尾
    _flush_current_record(cur, recs)

    num_with_start_arrival = sum(
        1 for r in recs if r.get("start_arrival", None) is not None
    )
    logger.info(
        "Parsed timing report %s: %d paths, %d with start_arrival",
        filepath,
        len(recs),
        num_with_start_arrival,
    )
    return recs


def group_by_endpoint(paths: List[PathRec]) -> Dict[str, List[PathRec]]:
    g: Dict[str, List[PathRec]] = {}
    for p in paths:
        ep = p.get("endpoint", "UNKNOWN")
        g.setdefault(ep, []).append(p)
    return g