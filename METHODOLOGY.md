# Temporal Node Classification Methodology

## Core Protocol

### Per-Node Observation Windows
For each K ∈ {0, 3, 5}, classify each node v using **at most K timesteps after its first appearance** t_first(v).

**Key Principle**: Every node gets evaluated at exactly `t_first(v) + K`, ensuring:
- Equal observation windows across all nodes
- No temporal leakage (no future information)
- Fair comparison across different K values

---

## Data & Graph Construction

### Cumulative Temporal Graphs
- Graphs are cumulative over time (t = 1...T)
- Nodes and edges persist after first appearance
- Node features computed only from data ≤ current timestep
- Labels are static (nodes don't change class)

### Feature Computation
Uses `wallets_features_until_t.csv` where features are aggregated **only up to each timestep t**, preventing temporal leakage.

---

## Temporal Splits (Non-Overlapping Cohorts)

### Cohort Definition
At each base time t, define **C_t = { v : t_first(v) = t }** (nodes appearing at timestep t).

### Split Configuration
```
Train cohorts:  C_5, C_6, ..., C_24    (timesteps 5-24)
Val cohorts:    C_29, C_30, C_31       (timesteps 29-31, gap: 25-28)
Test cohorts:   C_37, ..., C_43        (timesteps 37-43, gap: 32-36)
```

**Gaps ensure no overlap** between splits and allow observation windows to extend beyond split boundaries without contamination.

### Split Statistics

| Split | Timesteps | Nodes | % of Total | Illicit | Licit | Illicit % |
|-------|-----------|-------|------------|---------|-------|-----------|
| Train | 5-24 | 96,470 | 66.23% | 4,888 | 91,582 | 5.07% |
| Val | 29-31 | 9,884 | 6.79% | 664 | 9,220 | 6.72% |
| Test | 37-43 | 39,305 | 26.98% | 1,849 | 37,456 | 4.70% |
| **Total** | | **145,659** | **100%** | **7,401** | **138,258** | **5.08%** |

---

## Training & Evaluation Protocol

### For Non-Temporal Models (Baselines, Static GCNs)

**Training (per K value)**:
```
For each node v in train cohorts with `t_first(v) ∈ [5, 24]`:
    1. Extract features from graph at time `t_first(v) + K`
    2. Build training dataset from ALL nodes across their respective evaluation times
    3. Train ONE model per K value (for proper calibration)
    4. Only use graphs with `timestep ≤ train_end + K` during training
```

**Evaluation**:
```
For each node v in test cohorts with `t_first(v) ∈ [37, 43]`:
    1. Extract features from graph at time `t_first(v) + K`
    2. Predict on each node using its K-timestep observation
    3. Aggregate metrics across all test nodes
```

### For Temporal Models (EvolveGCN, TGN)

**Training**:
```
For each cohort t ∈ [5, 24]:
    1. RESET model state (if stateful)
    2. Feed graph sequence: t, t+1, ..., t+K
    3. Compute loss ONLY on nodes in C_t
    4. Mask out all other nodes
```

**Testing**:
```
For each cohort t ∈ [37, 43]:
    1. RESET model state
    2. Feed graph sequence: t, t+1, ..., t+K
    3. Predict ONLY on nodes in C_t
    4. Accumulate predictions across all test cohorts
```

**Critical**: Model state MUST be reset per cohort to enforce exact K-step context.

---

## Why Retrain Baselines Per K?

**Calibration Issues**:
- Feature distributions shift with K (more evidence ⇒ larger aggregates)
- Optimal weights/regularization differ by K
- One model trained across multiple Ks will be poorly calibrated (festure mean, median etc will involve longer times, so we want them to correspond to what it will se at eval time)

**Fair Comparison**:
- Per-K retraining matches test node's information set exactly
- Prevents bias toward any single K value
- Enables meaningful comparison of observation window effects

---

## Hard Rules (No Information Leakage)

1. ✅ **No training example may depend on data after `t_first(v) + K`**
2. ✅ **No training snapshot may exceed `train_end + max(K)`**
3. ✅ **In test, always reset state per cohort** (for temporal models)
4. ✅ **Each node evaluated at exactly `t_first(v) + K`**, not at `split_end + K`
5. ✅ **Features computed only from data ≤ current timestep**

---

## Research Questions

### Primary Question
**Does delaying node classification (larger K) yield better accuracy?**

### Model Comparisons

| Model Family | Benefits from K |
|--------------|-----------------|
| **Baselines (LR, RF, XGB)** | Extra feature aggregation only |
| **MLP + Graph Features** | Updated centrality values + aggregates |
| **Static GCN** | Updated feature aggregates propagated via message passing |
| **Temporal GCN (EvolveGCN)** | Temporal structure + evolving graph dynamics |
|**Temporal Graph Network**| Temporal structure + evolving graph dynamics

### Key Hypotheses
1. All models should improve with larger K (more information)
2. Temporal models should benefit most (exploit temporal structure)
3. Graph-based models should outperform feature-only baselines
4. There exists optimal K balancing information vs. classification delay

---

## Implementation Details

See `code_lib/temporal_node_classification_builder.py`:
- `prepare_observation_window_graphs()`: Creates per-node evaluation graphs
- `prepare_temporal_model_graphs()`: Creates sequences for temporal GNNs

### Observation Windows Tested
K ∈ {0, 3, 5}

- **K=0**: Classify immediately upon first appearance
- **K=3**: Wait 3 timesteps after first appearance
- **K=5**: Wait 5 timesteps after first appearance

---

## Metrics

- **AUC**: Area Under ROC Curve
- **Precision/Recall/F1**: Standard classification metrics
- **Per-K Performance**: Track how metrics change with observation window

Results reported separately for each K value to show effect of observation windows.
