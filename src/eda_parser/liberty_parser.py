from __future__ import annotations
import re
from typing import Dict, Set


def parse_liberty_pin_directions(lib_path: str) -> Dict[str, Dict[str, str]]:
    """
    Minimal Liberty parser to get pin directions for each cell.
    Returns: cell_name -> { pin_name: 'input'|'output'|'inout' }
    Notes:
      - Best-effort parser: tolerant to spaces/newlines, ignores power/ground.
      - Does not handle all Liberty features; sufficient for pin direction.
      - Supports both:
          pin(A) {
            direction : input;
          }
        和：
          pin(A) { direction : input; }
    """
    cell_dir: Dict[str, Dict[str, str]] = {}
    cur_cell: str | None = None
    cur_pin: str | None = None
    # stack 里只记录我们关心的 section 类型：'cell' / 'pin' / 'pg_pin'
    stack: list[str] = []

    # Precompile regex
    re_cell = re.compile(r'\bcell\s*\(\s*([A-Za-z0-9_$./]+)\s*\)\s*\{')
    re_pin  = re.compile(r'\bpin\s*\(\s*([A-Za-z0-9_$./\[\]]+)\s*\)\s*\{')
    re_dir  = re.compile(r'\bdirection\s*:\s*(input|output|inout)\s*;')
    re_pg   = re.compile(r'\bpg_pin\s*\(')  # ignore power/ground sections

    with open(lib_path, 'r', encoding='utf-8', errors='ignore') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            # 1. 进入 cell
            mcell = re_cell.search(line)
            if mcell:
                cur_cell = mcell.group(1)
                cell_dir.setdefault(cur_cell, {})
                stack.append('cell')

            if cur_cell is not None:
                # 当前是否处在 pg_pin 区块内部（最顶层）
                in_pg_pin = bool(stack and stack[-1] == 'pg_pin')

                # 2. 进入 pg_pin：之后的内容（直到对应的 '}'）都不参与 pin 解析
                if re_pg.search(line):
                    stack.append('pg_pin')
                    # 确保不会把 pg_pin 内的 direction 误记到前一个信号 pin 上
                    cur_pin = None
                    # 不 return/continue，仍然会在本行末尾处理 '}' 关闭 pg_pin

                # 3. 在非 pg_pin 区的情况下解析 pin 和 direction
                if not in_pg_pin:
                    # 进入 pin
                    mpin = re_pin.search(line)
                    if mpin:
                        cur_pin = mpin.group(1)
                        stack.append('pin')

                    # 解析 direction（允许与 pin 同行出现）
                    mdir = re_dir.search(line)
                    if mdir and cur_pin is not None:
                        direction = mdir.group(1).lower()
                        cell_dir[cur_cell][cur_pin] = direction

            # 4. 统一处理本行的 '}'，退出相应的 section
            close_braces = line.count('}')
            for _ in range(close_braces):
                if not stack:
                    break
                sect = stack.pop()
                if sect == 'pin':
                    cur_pin = None
                elif sect == 'cell':
                    cur_cell = None
                elif sect == 'pg_pin':
                    # 离开 pg_pin 区块，不需要额外操作
                    pass

    return cell_dir


def build_cell_output_index(pin_dirs: Dict[str, Dict[str, str]]) -> Dict[str, Set[str]]:
    """
    Build a quick index: cell_name -> set(output_pin_names)
    """
    idx: Dict[str, Set[str]] = {}
    for cell, pins in pin_dirs.items():
        outs = {p for p, d in pins.items() if d == 'output'}
        idx[cell] = outs
    return idx