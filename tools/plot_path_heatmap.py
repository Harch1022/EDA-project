#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据：
  1) build_dataset 生成的设计 npz (图结构 + name_to_idx)
  2) train_cpl 导出的节点重要性 npz (endpoints / importance / node_names)
  3) STA 关键路径 txt (一行一个节点名)

绘制单个 endpoint 的关键路径节点重要性热力图。
"""

import argparse
import os
import re
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx


def parse_args():
    parser = argparse.ArgumentParser(
        description="绘制单个 endpoint 的关键路径节点重要性热力图"
    )
    parser.add_argument(
        "--design_npz",
        type=str,
        required=True,
        help="build_dataset.py 生成的设计 npz 路径（例如 data/processed/my_design.npz）",
    )
    parser.add_argument(
        "--importance_npz",
        type=str,
        required=True,
        help="train_cpl 导出的节点重要性 npz 路径（*_node_importance_*.npz）",
    )
    parser.add_argument(
        "--sta_path",
        type=str,
        required=True,
        help="STA 关键路径 txt 文件，一行一个节点名（从起点到终点）",
    )
    parser.add_argument(
        "--ep_name",
        type=str,
        default=None,
        help="要可视化的 endpoint 名称（默认从 STA 路径的最后一行推断）",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="输出文件前缀（不含扩展名），例如 outputs/vis/heatmap_EP_A",
    )
    parser.add_argument(
        "--theme",
        type=str,
        default="light",
        choices=["light", "dark"],
        help="配色主题（light 或 dark）",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="输出图像 DPI",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[8.0, 6.0],
        help="图像尺寸 (W H)，单位英寸",
    )
    parser.add_argument(
        "--name_sub",
        type=str,
        nargs="*",
        default=[],
        help=(
            "可选的名字正则替换，用于把 STA 报告中的名字符合 dataset 中的 name_to_idx。"
            "格式为 'pattern=>repl'，可写多条。"
            "例如: --name_sub '^top/=>' '^u_core/=>core/'"
        ),
    )
    return parser.parse_args()


def build_name_sub_fn(rules: List[str]):
    """把 ['pat=>repl', ...] 转成一个 apply(name)->name 的函数。"""
    pairs: List[Tuple[re.Pattern, str]] = []
    for r in rules:
        if "=>" not in r:
            continue
        pat, repl = r.split("=>", 1)
        try:
            reg = re.compile(pat)
            pairs.append((reg, repl))
        except re.error as e:
            print(f"[WARN] 无法编译正则 '{pat}': {e}")

    def apply(name: str) -> str:
        s = name
        for reg, repl in pairs:
            s = reg.sub(repl, s)
        return s

    return apply


def load_design_graph(design_npz: str):
    """加载设计图结构和 name_to_idx，返回 NetworkX 图和映射。"""
    arr = np.load(design_npz, allow_pickle=True)
    if "edges" not in arr.files:
        raise KeyError(
            f"{design_npz} 中缺少 'edges' 键，请确认使用的是 build_dataset.py 的输出。"
        )
    edges = arr["edges"]  # [2, E]

    if "name_to_idx" not in arr.files:
        raise KeyError(
            f"{design_npz} 中缺少 'name_to_idx' 键，请确认使用的是 build_dataset.py 的输出。"
        )

    pairs = arr["name_to_idx"]
    name_to_idx: Dict[str, int] = {}
    for name, idx in pairs:
        name_to_idx[str(name)] = int(idx)
    idx_to_name: Dict[int, str] = {idx: name for name, idx in name_to_idx.items()}

    # 构造有向图
    G = nx.DiGraph()
    num_nodes = len(idx_to_name)
    G.add_nodes_from(range(num_nodes))
    src = edges[0].astype(int)
    dst = edges[1].astype(int)
    G.add_edges_from(zip(src, dst))

    return G, name_to_idx, idx_to_name


def load_importance(importance_npz: str):
    """加载节点重要性文件，返回 endpoints 列表、importance 数组和可选的 node_names。"""
    arr = np.load(importance_npz, allow_pickle=True)
    if "endpoints" not in arr.files or "importance" not in arr.files:
        raise KeyError(
            f"{importance_npz} 必须包含 'endpoints' 和 'importance' 两个键。"
        )
    endpoints = [str(e) for e in arr["endpoints"]]
    importance = arr["importance"].astype(np.float32)  # [E, N]
    node_names = None
    if "node_names" in arr.files:
        node_names = [str(n) for n in arr["node_names"]]
    return endpoints, importance, node_names


def load_sta_path(path_file: str, apply_sub):
    """加载 STA 关键路径，一行一个节点名，返回 (raw_lines, canonical_lines)。"""
    raw_lines: List[str] = []
    with open(path_file, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            raw_lines.append(s)
    if not raw_lines:
        raise RuntimeError(f"STA 路径文件 {path_file} 为空。")

    canonical = [apply_sub(s) for s in raw_lines]
    return raw_lines, canonical


def choose_endpoint(
    ep_arg: Optional[str],
    sta_raw: List[str],
    sta_canonical: List[str],
    imp_endpoints: List[str],
) -> str:
    """根据命令行参数和 STA 路径，决定使用哪个 endpoint。"""
    if ep_arg is not None:
        ep_name = ep_arg
    else:
        ep_name = sta_canonical[-1]

    if ep_name in imp_endpoints:
        return ep_name

    raw_last = sta_raw[-1]
    if raw_last in imp_endpoints:
        return raw_last

    print("[ERROR] 找不到要可视化的 endpoint：")
    print(f"  尝试 endpoint='{ep_name}' 或原始末行 '{raw_last}'")
    print("  但它们都未出现在 importance_npz 的 endpoints 中。")
    print("  importance 文件中的可用 endpoints（前 20 个示例）：")
    for e in imp_endpoints[:20]:
        print("   -", e)
    raise RuntimeError(
        "无法在 importance_npz 中找到匹配的 endpoint；"
        "请使用 --ep_name 显式指定，名称需与 importance_npz 中一致。"
    )


def compute_layout(G: nx.DiGraph):
    """使用 spring_layout 为整张图计算二维坐标。"""
    # 这里使用固定随机种子保证复现性
    pos = nx.spring_layout(G, seed=42)
    return pos


def plot_heatmap(
    G: nx.DiGraph,
    pos: Dict[int, Tuple[float, float]],
    scores: np.ndarray,  # [N]，已归一化的节点重要性
    path_nodes: List[int],
    path_node_names: List[str],
    out_prefix: str,
    theme: str,
    dpi: int,
    figsize: Tuple[float, float],
    ep_name: str,
):
    """实际绘图并保存 png/pdf/svg。"""
    # 主题颜色
    if theme == "dark":
        bg = "#111111"
        fg = "#EEEEEE"
        edge_color = "#555555"
        path_edge_color = "#FFB000"
        start_color = "#00CC66"
        end_color = "#FF3366"
    else:
        bg = "#FFFFFF"
        fg = "#000000"
        edge_color = "#CCCCCC"
        path_edge_color = "#7B1FA2"  # 紫色
        start_color = "#2E7D32"     # 绿
        end_color = "#C62828"       # 红

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    node_indices = list(G.nodes())
    xs = np.array([pos[i][0] for i in node_indices])
    ys = np.array([pos[i][1] for i in node_indices])

    s = np.asarray(scores, dtype=np.float32)
    s = np.maximum(s, 0.0)
    s_sum = float(s.sum())
    if s_sum > 0:
        s = s / s_sum

    if s.max() > 0:
        s_norm = s / float(s.max())
    else:
        s_norm = s

    min_size = 10.0
    max_size = 80.0
    node_sizes = min_size + (max_size - min_size) * s_norm

    cmap = plt.get_cmap("magma")
    # 注意 colorbar 显示的是“真实概率值”，因此 norm 用原始 s（已归一化）
    norm = mcolors.Normalize(vmin=0.0, vmax=float(s.max() if s.max() > 0 else 1.0))
    node_colors = cmap(norm(s))

    # 先画所有边
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        edge_color=edge_color,
        alpha=0.08 if theme == "light" else 0.15,
        arrows=False,
        width=0.5,
    )

    # 再画所有节点（基础层）
    ax.scatter(
        xs,
        ys,
        s=node_sizes,
        c=node_colors,
        alpha=0.85,
        linewidths=0,
        zorder=2,
    )

    # 路径叠加
    if path_nodes:
        path_xy = np.array([pos[i] for i in path_nodes])
        # 路径连线
        ax.plot(
            path_xy[:, 0],
            path_xy[:, 1],
            color=path_edge_color,
            linewidth=2.0,
            alpha=0.9,
            zorder=5,
        )
        # 路径节点高亮边框
        ax.scatter(
            path_xy[:, 0],
            path_xy[:, 1],
            s=node_sizes[path_nodes] * 1.4,
            facecolors="none",
            edgecolors=path_edge_color,
            linewidths=1.2,
            zorder=6,
        )

        # 起点 / 终点标记
        start_idx = path_nodes[0]
        end_idx = path_nodes[-1]
        xs_start, ys_start = pos[start_idx]
        xs_end, ys_end = pos[end_idx]
        ax.scatter(
            [xs_start],
            [ys_start],
            s=node_sizes[start_idx] * 2.0,
            marker="s",
            facecolors=start_color,
            edgecolors="black" if theme == "light" else "white",
            linewidths=0.8,
            zorder=7,
            label="Path start",
        )
        ax.scatter(
            [xs_end],
            [ys_end],
            s=node_sizes[end_idx] * 2.2,
            marker="*",
            facecolors=end_color,
            edgecolors="black" if theme == "light" else "white",
            linewidths=0.8,
            zorder=7,
            label="Path end",
        )

    # 颜色条
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Node importance (probability)", color=fg)
    cbar.ax.yaxis.set_tick_params(color=fg)
    plt.setp(cbar.ax.get_yticklabels(), color=fg)

    # 统计：路径上累积的概率质量，构造标题
    if path_nodes:
        path_mass = float(s[path_nodes].sum())
        subtitle = f"Endpoint: {ep_name} | Sum importance on STA path = {path_mass:.3f}"
    else:
        subtitle = f"Endpoint: {ep_name} (STA path not mapped to graph)"

    main_title = "Critical-path node importance heatmap"
    full_title = main_title + "\n" + subtitle

    # 用 suptitle 统一画两行标题，避免与 Axes 标题/文字重叠
    fig.suptitle(full_title, color=fg, fontsize=13, y=0.98)

    ax.set_xticks([])
    ax.set_yticks([])

    if path_nodes:
        ax.legend(
            facecolor=bg,
            edgecolor=fg,
            labelcolor=fg,
            loc="upper right",
        )

    ax.set_aspect("equal")

    # 为 suptitle 预留顶部空间
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])

    # 保存三种格式
    for ext in ("png", "pdf", "svg"):
        out_path = f"{out_prefix}.{ext}"
        fig.savefig(out_path, dpi=dpi, facecolor=bg, bbox_inches="tight")
        print(f"[INFO] Saved figure: {out_path}")

    plt.close(fig)


def main():
    args = parse_args()
    name_sub_fn = build_name_sub_fn(args.name_sub)

    print(f"[INFO] 加载设计图: {args.design_npz}")
    G, name_to_idx, idx_to_name = load_design_graph(args.design_npz)
    N_nodes = G.number_of_nodes()
    print(f"[INFO] 图节点数 = {N_nodes}, 边数 = {G.number_of_edges()}")

    print(f"[INFO] 加载节点重要性: {args.importance_npz}")
    ep_list, importance, node_names_from_imp = load_importance(args.importance_npz)
    if importance.shape[1] != N_nodes:
        raise RuntimeError(
            f"importance 中的节点数={importance.shape[1]} 与 设计图节点数={N_nodes} 不一致。"
        )
    print(
        f"[INFO] importance 中包含 {len(ep_list)} 个 endpoints，对应 {N_nodes} 个节点。"
    )

    print(f"[INFO] 加载 STA 路径: {args.sta_path}")
    sta_raw, sta_can = load_sta_path(args.sta_path, name_sub_fn)

    ep_name = choose_endpoint(args.ep_name, sta_raw, sta_can, ep_list)
    ep_idx = ep_list.index(ep_name)
    scores = importance[ep_idx].copy()

    # 将分布标准化为概率（防御性处理）
    scores = np.maximum(scores, 0.0)
    s_sum = float(scores.sum())
    if s_sum > 0:
        scores = scores / s_sum

    # STA 路径节点映射到图节点索引
    path_nodes: List[int] = []
    path_node_names: List[str] = []
    missing_names: List[str] = []
    for raw, can in zip(sta_raw, sta_can):
        idx = name_to_idx.get(can, None)
        if idx is None:
            missing_names.append(raw)
            continue
        path_nodes.append(idx)
        path_node_names.append(raw)

    if not path_nodes:
        print(
            "[WARN] STA 路径中的节点一个都没能在 design_npz 的 name_to_idx 中找到，"
            "图中将不会绘制路径叠加。"
        )
    elif missing_names:
        print("[WARN] STA 路径中部分节点未能找到对应图节点（将被忽略）：")
        for n in missing_names:
            print("   -", n)

    print("[INFO] 计算 spring_layout 布局（可能需要一点时间）...")
    pos = compute_layout(G)

    print(f"[INFO] 绘图并保存到前缀: {args.out}")
    plot_heatmap(
        G=G,
        pos=pos,
        scores=scores,
        path_nodes=path_nodes,
        path_node_names=path_node_names,
        out_prefix=args.out,
        theme=args.theme,
        dpi=args.dpi,
        figsize=tuple(args.figsize),
        ep_name=ep_name,
    )
    print("[INFO] 完成。")


if __name__ == "__main__":
    main()