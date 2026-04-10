# Multi-Modal GNN for Cross-Chip Timing Prediction and Physical Congestion Visualization

This repository contains the source code for our Final Year Project (FYP) at the Chinese University of Hong Kong (CUHK). The project focuses on accelerating Static Timing Analysis (STA) by predicting Critical Path Length (CPL) and Arrival Time using a hybrid deep learning architecture that integrates logical topology, physical layout, and bidirectional feature propagation.

## 🚀 Key Innovations

Building upon initial research on single-design prediction, this project introduces a "cross-chip" capable architecture with five core innovations:

1.  **Physical-Aware Graph Construction**: Unlike pure logical graphs, our `graph_builder.py` incorporates physical coordinates from DEF files. We utilize **Exponentially Decaying Manhattan Distance** to weight edges, accurately reflecting the impact of physical placement on timing.
2.  **Hybrid Neural Architecture**: Integration of GNN (for logical topology), CNN (for 2D layout patterns), and BPN (Bidirectional Propagation Network) for refined node-level feature spreading.
3.  **NUIAT Differentiation Mechanism**: A novel attention-based mechanism that leverages a temperature-scaled Softmax and real arrival times to guide BPN seed distribution.
4.  **Pairwise Ranking Loss**: To overcome the magnitude differences in absolute delay across different chips (e.g., Ibex vs. AES), we implemented a Ranking Loss. This enables the model to focus on relative timing criticalities, achieving SOTA results with a **Kendall's Tau of 0.496** on cross-chip tasks.
5.  **Interpretability & Visualization**: Development of 2D physical heatmap projection technology to visualize timing congestion and bottleneck areas directly on the chip layout.

## 📂 Repository Structure

```text
.
├── configs/                # YAML configurations for benchmarks, datasets, and models
├── data/                   # Benchmarks (Ibex, AES) and STA timing report paths
├── eda_flow/               # TCL/Shell scripts for synthesis and P&R
├── experiments/            # Automation scripts for multi-design execution
├── src/
│   ├── eda_parser/         # Custom parsers for .v, .def, .lib, and timing reports
│   ├── features/           # Feature engineering (graph_builder.py, cnn_maps.py)
│   ├── models/             # GNN, CNN, BPN, and FusionRegressor architectures
│   ├── training/           # Training loops with CPL and Ranking Loss
│   └── vis/                # Heatmap generation and visualization tools
└── requirements.txt        # Project dependencies (Torch, DGL, etc.)

🛠️ InstallationEnsure you have a Python 3.8+ environment. Install the dependencies using:Bashpip install -r requirements.txt

Core dependencies include:
PyTorch >= 2.0.1
DGL >= 1.1 (Deep Graph Library)NumPy, SciPy, Pandas, NetworkX

🚀 Usage
1. Data Preparation & EDA FlowTo run the underlying EDA flow (synthesis and P&R) for all designs defined in configs/benches.yaml:
python experiments/run_all_flows.py
2. Model TrainingTo train the cross-chip prediction model using the combined dataset:
python -m src.training.train_cpl.py --config configs/model.yaml
3. VisualizationGenerate timing criticality heatmaps for a specific design:
python src/vis/plot_heatmap.py

📊 PerformanceSingle Design: Achieved an $R^2 \approx 0.9294$ on fixed designs using teacher-pool conditioning.
Cross-Chip: Successfully generalized across heterogenous designs (Ibex + AES) with a Kendall's Tau of 0.496, significantly outperforming MSE-based baselines.
