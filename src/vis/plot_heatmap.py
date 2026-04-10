import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from src.eda_parser.def_parser import parse_def

def plot_heatmap(design_name: str):
    print(f"🔥 开始渲染 {design_name} 的时序热力图...")

    # 1. 配置文件路径
    def_path = f"data/raw_eda/{design_name}/{design_name}.post_route.def"
    dataset_path = f"data/processed/{design_name}.npz"
    importance_path = f"results/vis/node_importance/{design_name}_node_importance_teacher.npz"

    # 检查文件是否存在
    for p in [def_path, dataset_path, importance_path]:
        if not os.path.exists(p):
            print(f"❌ 找不到关键文件: {p}")
            print("请确认你已经成功运行了特征提取和模型训练。")
            return

    # 2. 解析 DEF 获取物理坐标
    print("   -> 正在解析 DEF 版图坐标...")
    comps, die_area, _ = parse_def(def_path)

    # 3. 加载数据集字典，映射 名称 -> 图节点索引
    print("   -> 正在映射图网络节点...")
    ds = np.load(dataset_path, allow_pickle=True)
    # name_to_idx 保存的是 [[name, idx], ...] 格式
    name_to_idx = dict(ds['name_to_idx']) 

    # 4. 加载模型导出的重要性分数
    print("   -> 正在加载 AI 预测的热点权重...")
    imp_data = np.load(importance_path, allow_pickle=True)
    
    # 智能寻找真正的数值型分数矩阵（避开字符串数组）
    importance = None
    for key in imp_data.files:
        try:
            # 尝试将其转换为 float，如果是纯名字字符串会报错并跳过
            arr_float = imp_data[key].astype(float)
            importance = arr_float
            # 优先选择高维矩阵 (Endpoints, Nodes)
            if len(arr_float.shape) > 1:
                break
        except (ValueError, TypeError):
            continue
            
    if importance is None:
        print("❌ 在 npz 文件中找不到数值型的权重矩阵！")
        return

    # 如果 importance 是 2D 的 (Endpoints, Nodes)，我们在端点维度求均值，看全局热点
    if len(importance.shape) > 1:
        importance = importance.mean(axis=0)

    # 5. 组合坐标与权重
    bg_xs, bg_ys = [], []       # 记录整个芯片的所有组件（作为灰色底板）
    xs, ys, scores = [], [], [] # 记录被 AI 预测打分的组件（作为彩色热点）
    
    for comp_name, comp_info in comps.items():
        # 提取当前组件的物理坐标
        try:
            if isinstance(comp_info, dict) and 'xy' in comp_info:
                x, y = comp_info['xy']
            elif isinstance(comp_info, (list, tuple)) and len(comp_info) >= 2:
                x, y = comp_info[0], comp_info[1]
            else:
                continue
            # 把所有能找到坐标的组件都加入底板
            bg_xs.append(x)
            bg_ys.append(y)
        except Exception:
            continue
            
        # 如果这个组件在图网络里，且有预测分数，才给它上色
        if comp_name in name_to_idx:
            node_idx = int(name_to_idx[comp_name])
            if node_idx < len(importance):
                score = importance[node_idx]
                xs.append(x)
                ys.append(y)
                scores.append(score)

    if not bg_xs:
        print("❌ 未能成功提取任何组件的物理坐标！")
        return

    # 6. 开始画图
    print(f"   -> 准备绘制 {len(bg_xs)} 个背景组件，以及 {len(xs)} 个高亮热点...")
    plt.figure(figsize=(12, 10))
    
    # 动态计算颜色的上下界
    scores_arr = np.array(scores)
    vmin = np.percentile(scores_arr, 1) if len(scores_arr) > 0 else 0.0
    vmax = np.percentile(scores_arr, 99) if len(scores_arr) > 0 else 1.0

    # 【关键修改】：先画全芯片的浅灰色底板，再叠加彩色的热力点
    plt.scatter(bg_xs, bg_ys, c='lightgray', s=0.2, alpha=0.3) 
    sc = plt.scatter(xs, ys, c=scores, cmap='jet', s=6, alpha=0.9, vmin=vmin, vmax=vmax)

    if not xs:
        print("❌ 未能成功匹配任何节点的物理坐标，请检查字典映射！")
        return

    # 6. 开始画图
    print(f"   -> 准备绘制 {len(xs)} 个物理组件...")
    plt.figure(figsize=(12, 10))
    
    # 动态计算颜色的上下界 (砍掉 1% 的极端离群值，让热力图颜色对比更鲜明)
    scores = np.array(scores)
    vmin = np.percentile(scores, 1) if len(scores) > 0 else 0.0
    vmax = np.percentile(scores, 99) if len(scores) > 0 else 1.0

    # 绘制底噪（灰色），然后再在上面绘制带颜色的热点
    plt.scatter(xs, ys, c='lightgray', s=1, alpha=0.1)
    sc = plt.scatter(xs, ys, c=scores, cmap='jet', s=3, alpha=0.8, vmin=vmin, vmax=vmax)
    
    # 图表装饰
    cbar = plt.colorbar(sc, fraction=0.046, pad=0.04)
    cbar.set_label('Predicted Timing Criticality (Severity)', rotation=270, labelpad=20, fontsize=12)
    
    plt.title(f"Timing Bottleneck Heatmap - {design_name.upper()}", fontsize=16, fontweight='bold')
    plt.axis('equal')
    plt.axis('off') # 去掉丑陋的坐标轴边框

    # 7. 保存图像
    out_dir = "results/vis/heatmaps"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{design_name}_heatmap.png")
    
    plt.savefig(out_file, dpi=400, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ 大功告成！热力图已保存至: {out_file}\n")

if __name__ == "__main__":
    # 如果在命令行传了参数，就画指定的芯片；否则默认画 my_design
    design = sys.argv[1] if len(sys.argv) > 1 else "my_design"
    plot_heatmap(design)