# Temporal Fraud Detection in Bitcoin Transaction Networks

A machine learning research project investigating the impact of observation windows on fraud detection in the Elliptic Bitcoin transaction dataset using graph neural networks and traditional ML baselines.

## Table of Contents

- [Overview](#overview)
- [Research Question](#research-question)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Project Structure](#project-structure)
- [Models & Experiments](#models--experiments)
- [Results](#results)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Documentation](#documentation)

---

## Overview

This project addresses temporal fraud detection in Bitcoin transaction networks using the **Elliptic dataset** (13.7 GB). The core research investigates whether delaying node classification by observing nodes for **K additional timesteps** after their first appearance improves fraud detection accuracy.

**Key Innovation**: Rigorous temporal methodology with non-overlapping cohorts and per-node observation windows to prevent information leakage while exploring the trade-off between observation delay and classification accuracy.

---

## Research Question

**Does delaying node classification (waiting K timesteps after first appearance) improve fraud detection accuracy?**

We investigate this across multiple model families:
- **Baseline Models**: Logistic Regression, Random Forest, XGBoost (feature aggregation only)
- **MLP + Graph Features**: Neural network with structural graph features (centrality, PageRank, etc.)
- **Static GCN**: Graph convolutional network on static graph snapshots
- **Temporal GCN**: EvolveGCN-style model with LSTM to capture temporal dynamics

**Observation Windows Tested**: K ∈ {1, 3, 5, 7} timesteps

---

## Dataset

### Elliptic Bitcoin Transaction Dataset

- **Size**: 13.7 GB (tracked via Git LFS)
- **Nodes**: ~203,769 Bitcoin wallets
- **Edges**: Transaction relationships between wallets
- **Timesteps**: 49 discrete time periods
- **Labels**: Illicit (fraud, scams, ransomware) vs. Licit wallets
- **Features**: 166 transaction features per wallet (reduced to 36 after correlation analysis)
- **Class Distribution**: ~5-8% illicit (highly imbalanced)

### Temporal Data Split

| Split | Timesteps | Nodes | Illicit % |
|-------|-----------|-------|-----------|
| **Train** | 5-26 | 104,704 | 6.4% |
| **Validation** | 27-31 | 11,230 | 7.2% |
| **Test** | 32-40 | 45,963 | 8.0% |

**Critical**: Gaps built between splits to prevent temporal information leakage. Each node is evaluated at exactly `t_first(v) + K` where `t_first(v)` is the node's first appearance timestep.

---

## Methodology

### Core Protocol

**Per-Node Observation Windows**:
- For each node `v` appearing at timestep `t_first(v)`, classify using data from timesteps `t_first(v)` through `t_first(v) + K`
- Ensures equal observation windows across all nodes
- Prevents temporal leakage (no future information)
- Enables fair comparison across different K values

### Temporal Edge Weighting

Edges are weighted using exponential temporal decay:

```
S_ji(t) = Σ A_ji^(s) × exp(-λ(t-s))
```

where:
- `A_ji^(s)`: Binary edge indicator at timestep s
- `λ`: Decay rate parameter (controls memory)
- Temperature-softmax normalization for final edge weights

### Per-Cohort Training (Temporal Models)

For temporal GNNs:
1. **Reset** model state at start of each cohort
2. Feed graph sequence: `t, t+1, ..., t+K`
3. Compute loss **only** on nodes in current cohort
4. Prevents information leakage across training examples

### Feature Reduction

- Original: 166 features → **36 features** after removing highly correlated features (Pearson correlation > 0.95)
- Added: **1 temporal age feature** (normalized timestep of first appearance)
- Graph features (MLP baseline): 7 structural features including PageRank, degree centrality, betweenness centrality

For complete methodology details, see [METHODOLOGY.md](METHODOLOGY.md).

---

## Project Structure

```
graph_ml/
├── code_lib/                          # Core library modules
│   ├── temporal_node_classification_builder.py  # Main graph builder (1008 lines)
│   ├── temporal_edge_builder.py       # Edge construction with decay weighting
│   ├── temporal_graph_builder.py      # PyG Temporal conversion utilities
│   └── utils.py                       # Data loading helpers
│
├── elliptic_dataset/                  # Bitcoin transaction dataset (13.7 GB)
│   ├── wallets_features_until_t.csv  # Temporal features (no leakage)
│   ├── wallets_features.csv          # Wallet features
│   └── AddrTxAddr_edgelist_*.csv     # Transaction edges (8 parts)
│
├── notebooks/
│   ├── experiments/                   # Main experiment notebooks
│   │   ├── evolve_gcn.ipynb          # Temporal GCN experiments
│   │   ├── static_gcn.ipynb          # Static GCN experiments
│   │   ├── baselines.ipynb           # Traditional ML baselines
│   │   ├── graph_features_baseline.ipynb  # MLP + graph features
│   │   └── model_comparison_visualization.ipynb  # Results visualization
│   └── other/                        # Exploratory analysis
│
├── results/                          # Experimental results
│   ├── evolve_gcn_multi_seed/       # Temporal GCN (seeds: 42, 123, 456)
│   ├── static_gcn_multi_seed/       # Static GCN (seeds: 42, 123, 456)
│   ├── baselines/                   # Logistic Regression, RF, XGBoost
│   │   ├── logistic_regression/
│   │   ├── random_forest/
│   │   └── xgboost/
│   ├── graph_features_baseline/     # MLP + graph features
│   └── comparison_formatted.csv     # Unified results comparison
│
├── graph_cache/                      # Cached graph snapshots
├── tests/                           # Unit tests
└── [Documentation files]
```

---

## Models & Experiments

### 1. Traditional ML Baselines

**Models**: Logistic Regression, Random Forest, XGBoost
**Features**: 36 reduced transaction features
**Training**: Per-K retraining for proper calibration

### 2. MLP + Graph Features

**Architecture**: Multi-layer perceptron
**Features**: 36 reduced features + 7 graph structural features
- Total/in/out degree
- PageRank
- Betweenness centrality
- Degree ratio
- Normalized degree centrality

### 3. Static GCN

**Architecture**: 2-layer Graph Convolutional Network
**Training**: Multi-seed (42, 123, 456)
**Features**: 36 reduced + 1 temporal age feature
**Graph**: Static snapshot at evaluation time

### 4. Temporal GCN (EvolveGCN-style)

**Architecture**: 2-layer GCN + LSTM + classifier
**Training**: Multi-seed (42, 123, 456), per-cohort with state reset
**Features**: 36 reduced + 1 temporal age feature
**Graph**: Temporal sequence with weighted edges

---

## Results

### Summary Statistics (Test Set)

All results reported as Mean ± Std across 3 random seeds (for GNN models).

#### Graph Neural Networks

| Model | K | F1 Score | AUC | Precision | Recall |
|-------|---|----------|-----|-----------|--------|
| **Temporal GCN** | 1 | 0.312 ± 0.065 | 0.753 ± 0.043 | 0.229 ± 0.050 | 0.502 ± 0.127 |
| **Temporal GCN** | 3 | 0.338 ± 0.042 | 0.732 ± 0.062 | 0.263 ± 0.021 | 0.500 ± 0.160 |
| **Temporal GCN** | 5 | 0.301 ± 0.019 | 0.679 ± 0.015 | 0.231 ± 0.017 | 0.435 ± 0.025 |
| **Temporal GCN** | 7 | **0.332 ± 0.034** | **0.782 ± 0.003** | 0.233 ± 0.029 | **0.580 ± 0.036** |
| Static GCN | 1 | 0.168 ± 0.156 | 0.509 ± 0.154 | 0.133 ± 0.153 | 0.476 ± 0.477 |
| Static GCN | 3 | 0.123 ± 0.054 | 0.603 ± 0.013 | 0.460 ± 0.354 | 0.259 ± 0.361 |
| Static GCN | 5 | 0.179 ± 0.027 | 0.548 ± 0.042 | 0.185 ± 0.094 | 0.444 ± 0.480 |
| Static GCN | 7 | 0.166 ± 0.059 | 0.590 ± 0.050 | 0.151 ± 0.053 | 0.345 ± 0.380 |

#### Traditional ML Baselines

| Model | K | F1 Score | AUC | Precision | Recall |
|-------|---|----------|-----|-----------|--------|
| Logistic Regression | 1 | 0.249 | 0.875 | 0.143 | **0.958** |
| Logistic Regression | 3 | 0.251 | 0.874 | 0.145 | 0.958 |
| Logistic Regression | 5 | 0.247 | 0.869 | 0.142 | 0.958 |
| Logistic Regression | 7 | 0.245 | 0.865 | 0.140 | 0.960 |
| Random Forest | 1 | **0.824** | 0.930 | **0.986** | 0.707 |
| Random Forest | 3 | 0.819 | 0.915 | 0.988 | 0.699 |
| Random Forest | 5 | 0.824 | 0.925 | 0.989 | 0.707 |
| Random Forest | 7 | 0.821 | 0.924 | 0.988 | 0.702 |
| XGBoost | 1 | 0.788 | 0.943 | 0.814 | 0.763 |
| XGBoost | 3 | 0.803 | 0.942 | 0.837 | 0.771 |
| XGBoost | 5 | **0.806** | **0.948** | **0.846** | **0.770** |
| XGBoost | 7 | 0.783 | 0.949 | 0.825 | 0.745 |

#### MLP + Graph Features

| Model | K | F1 Score | AUC | Precision | Recall |
|-------|---|----------|-----|-----------|--------|
| MLP + Graph Features | 1 | 0.233 | 0.712 | 0.133 | 0.909 |
| MLP + Graph Features | 3 | 0.234 | 0.685 | 0.134 | 0.908 |
| MLP + Graph Features | 5 | 0.234 | 0.686 | 0.134 | 0.909 |
| MLP + Graph Features | 7 | 0.233 | 0.691 | 0.134 | 0.909 |

### Key Findings

1. **Best Overall Performance**: Random Forest and XGBoost significantly outperform GNN models (F1 ~0.80-0.82 vs. 0.30-0.34), suggesting transaction features are more informative than graph structure for this task

2. **Temporal GCN vs. Static GCN**: Temporal models consistently outperform static models, validating the importance of temporal dynamics

3. **Observation Window Effects**:
   - Non-monotonic relationship with K
   - Temporal GCN shows best performance at K=7 (AUC 0.782)
   - XGBoost peaks at K=5 (F1 0.806)
   - Random Forest relatively stable across K values

4. **Class Imbalance Challenge**: High recall but low precision in many models reflects the severe class imbalance (~5-8% illicit)

5. **Model Stability**: Multi-seed experiments reveal variable stability:
   - Temporal GCN: relatively stable (std ~0.02-0.06 for F1)
   - Static GCN: high variance (std up to 0.16 for F1)

---

## Setup & Installation

### Prerequisites

- Python 3.11+
- CUDA-capable GPU (recommended) or CPU/MPS support
- Git LFS for large dataset files

### 1. Dataset Setup

```bash
# Install git-lfs
brew install git-lfs  # macOS
# or
apt-get update && apt-get install git-lfs  # Linux

# Initialize LFS
git lfs install

# Pull large files
git lfs pull
```

### 2. Environment Setup

```bash
# Create conda environment
conda env create -f env.yml

# Activate environment
conda activate graph_ml
```

### 3. Dependencies

Key packages:
- PyTorch (CPU/CUDA/MPS)
- PyTorch Geometric
- PyTorch Geometric Temporal
- scikit-learn
- XGBoost
- pandas, numpy, matplotlib
- JupyterLab

---

## Usage

### Running Experiments

1. **Temporal GCN**: Open [notebooks/experiments/evolve_gcn.ipynb](notebooks/experiments/evolve_gcn.ipynb)
2. **Static GCN**: Open [notebooks/experiments/static_gcn.ipynb](notebooks/experiments/static_gcn.ipynb)
3. **Baselines**: Open [notebooks/experiments/baselines.ipynb](notebooks/experiments/baselines.ipynb)
4. **MLP + Graph Features**: Open [notebooks/experiments/graph_features_baseline.ipynb](notebooks/experiments/graph_features_baseline.ipynb)
5. **Visualize Results**: Open [notebooks/experiments/model_comparison_visualization.ipynb](notebooks/experiments/model_comparison_visualization.ipynb)

### Graph Building API

```python
from code_lib.temporal_node_classification_builder import TemporalNodeClassificationGraphBuilder

# Initialize builder
builder = TemporalNodeClassificationGraphBuilder(
    observation_windows=[1, 3, 5, 7],
    use_cache=True
)

# Build graphs for static models
train_data, val_data, test_data = builder.prepare_observation_window_graphs(K=3)

# Build sequences for temporal models
temporal_graphs = builder.prepare_temporal_model_graphs(K=3)
```

### Caching System

Graph snapshots are automatically cached to `graph_cache/` for faster repeated experiments. Cache keys include all relevant configuration parameters to ensure consistency.

---

## RunPod Cluster Setup

**For collaborative development on GPU cluster:**

1. Go to RunPod and select "Franek's Team" from the dropdown
2. Navigate to "Pods" tab
3. Select the "GraphML" storage volume
4. Click "Change Template" and select "graph_ml_updated"
5. Click "Deploy On-Demand"
6. Once deployed, click "jupyterlab" to access JupyterLab in browser

**Important**:
- Do NOT delete the storage volume
- Remember to terminate the pod after use to avoid idle costs
- Git credentials are pre-configured in the network volume

---

## Documentation

- [METHODOLOGY.md](METHODOLOGY.md) - Detailed research protocol and temporal methodology
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Temporal edge weighting implementation
- [MULTI_SEED_IMPLEMENTATION_GUIDE.md](MULTI_SEED_IMPLEMENTATION_GUIDE.md) - Multi-seed training guide
- [SIMPLIFIED_IMPLEMENTATION_GUIDE.md](SIMPLIFIED_IMPLEMENTATION_GUIDE.md) - Step-by-step implementation guide
- [FINAL_TODO.md](FINAL_TODO.md) - Implementation checklist

---

## Citation

If you use this code or methodology in your research, please cite the Elliptic dataset:

```
Weber, M., Domeniconi, G., Chen, J., Weidele, D. K. I., Bellei, C., Robinson, T., & Leiserson, C. E. (2019).
Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics.
KDD '19 Workshop on Anomaly Detection in Finance.
```

---

## License

This project is for academic research purposes. Please ensure proper attribution when using this code or methodology.

---

## Contact

For questions or issues, please open an issue in the repository.
