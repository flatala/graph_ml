# Simplified Multi-Seed Implementation Guide

## Summary

**Multi-seed training**: Only Temporal GCN and Static GCN
**Single run**: Baselines (LR, RF, XGB) and MLP + Graph Features
**All models**: Save results in consistent format for unified comparison

---

## 1. static_gcn.ipynb - Add Multi-Seed Training

Already covered in detail in `MULTI_SEED_IMPLEMENTATION_GUIDE.md` Section 1.

**Key points**:
- Add feature reduction
- Use `cache_dir='../../graph_cache_reduced_features_fixed'`
- Wrap in multi-seed loop with `SEEDS = [42, 123, 456]`
- Save to `../../results/static_gcn_multi_seed/`

---

## 2. graph_features_baseline.ipynb - Enhanced Features + Single Run Save

### Add Enhanced Graph Features

```python
from torch_geometric.utils import degree
import torch

def compute_pagerank(edge_index, num_nodes, alpha=0.85, max_iter=10):
    """Compute PageRank centrality."""
    pr = torch.ones(num_nodes) / num_nodes
    deg_out = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.float)
    deg_out[deg_out == 0] = 1

    for _ in range(max_iter):
        pr_new = torch.zeros(num_nodes)
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            pr_new[dst] += pr[src] / deg_out[src]
        pr = alpha * pr_new + (1 - alpha) / num_nodes

    return pr

def compute_graph_features(data):
    """Add comprehensive graph structural features."""
    num_nodes = data.num_nodes
    edge_index = data.edge_index

    # 1-3. Degree centrality
    deg_total = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.float)
    deg_in = degree(edge_index[1], num_nodes=num_nodes, dtype=torch.float)
    deg_out = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.float)

    # 4. PageRank centrality
    pagerank = compute_pagerank(edge_index, num_nodes, max_iter=5)

    # 5. Betweenness approximation
    betweenness_approx = (deg_in * deg_out) / (deg_total + 1e-8)

    # 6. Degree ratio (in/out balance)
    degree_ratio = deg_in / (deg_out + 1e-8)

    # 7. Normalized degree (compared to max)
    deg_normalized = deg_total / (deg_total.max() + 1e-8)

    # Stack all features
    graph_feats = torch.stack([
        deg_total, deg_in, deg_out,
        pagerank, betweenness_approx,
        degree_ratio, deg_normalized
    ], dim=1)

    combined_feats = torch.cat([data.x, graph_feats], dim=1)
    return combined_feats
```

### Add Feature Reduction

Add before loading data:
```python
def remove_correlated_features(nodes_df, threshold=0.95, verbose=True):
    """Remove highly correlated features."""
    exclude_cols = {'address', 'Time step', 'class'}
    feature_cols = [col for col in nodes_df.columns
                    if col not in exclude_cols and
                    pd.api.types.is_numeric_dtype(nodes_df[col])]

    sample_size = min(10000, len(nodes_df))
    sample_df = nodes_df[feature_cols].sample(n=sample_size, random_state=42)
    corr_matrix = sample_df.corr().abs()

    to_remove = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > threshold:
                to_remove.add(corr_matrix.columns[j])

    features_to_keep = [col for col in feature_cols if col not in to_remove]

    if verbose:
        print(f"Original features: {len(feature_cols)}")
        print(f"Kept features: {len(features_to_keep)}")

    return features_to_keep
```

### Update Data Loading

```python
nodes_df, edges_df = load_elliptic_data(CONFIG['data_dir'], use_temporal_features=True)

# Add feature reduction
kept_features = remove_correlated_features(nodes_df, threshold=0.95, verbose=False)
print(f"Before: {nodes_df.shape[1]}")
print(f"After: {len(kept_features)}")

builder = TemporalNodeClassificationBuilder(
    nodes_df=nodes_df,
    edges_df=edges_df,
    feature_cols=kept_features,  # ADD THIS
    include_class_as_feature=False,
    add_temporal_features=True,
    cache_dir='../../graph_cache_reduced_features_fixed',  # CHANGE THIS
    use_cache=True,
    verbose=True
)
```

### Add Results Saving (at end of notebook)

```python
from pathlib import Path

# Create results directory
RESULTS_DIR = Path('../../results/graph_features_baseline')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Save metrics
all_metrics = []
for K in CONFIG['observation_windows']:
    for split_name in ['train', 'val', 'test']:
        metrics = results[K][split_name]
        all_metrics.append({
            'K': K,
            'split': split_name,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'auc': metrics['auc']
        })

metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_csv(RESULTS_DIR / 'all_metrics.csv', index=False)
print(f"Saved metrics to {RESULTS_DIR / 'all_metrics.csv'}")

# Save summary (for comparison notebook compatibility)
summary_stats = []
test_metrics = metrics_df[metrics_df['split'] == 'test']
for K in CONFIG['observation_windows']:
    k_metrics = test_metrics[test_metrics['K'] == K].iloc[0]
    summary_stats.append({
        'K': K,
        'f1_mean': k_metrics['f1'],
        'f1_std': 0.0,  # No std for single run
        'auc_mean': k_metrics['auc'],
        'auc_std': 0.0,
        'precision_mean': k_metrics['precision'],
        'precision_std': 0.0,
        'recall_mean': k_metrics['recall'],
        'recall_std': 0.0,
        'accuracy_mean': k_metrics['accuracy'],
        'accuracy_std': 0.0
    })

summary_df = pd.DataFrame(summary_stats)
summary_df.to_csv(RESULTS_DIR / 'summary_statistics.csv', index=False)
print(f"Saved summary to {RESULTS_DIR / 'summary_statistics.csv'}")
```

---

## 3. baselines.ipynb - Add Results Saving Only

Already has feature reduction! Just add result saving at the end:

```python
from pathlib import Path

# Create results directories for each model
RESULTS_DIR_BASE = Path('../../results/baselines')
RESULTS_DIR_BASE.mkdir(parents=True, exist_ok=True)

model_dirs = {
    'LogisticRegression': RESULTS_DIR_BASE / 'logistic_regression',
    'RandomForest': RESULTS_DIR_BASE / 'random_forest',
    'XGBoost': RESULTS_DIR_BASE / 'xgboost'
}

for model_name, model_dir in model_dirs.items():
    model_dir.mkdir(parents=True, exist_ok=True)

    # Collect metrics for this model
    all_metrics = []
    for K in CONFIG['observation_windows']:
        for split_name in ['train', 'val', 'test']:
            metrics = results[model_name][K][split_name]
            all_metrics.append({
                'K': K,
                'split': split_name,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'auc': metrics['auc']
            })

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(model_dir / 'all_metrics.csv', index=False)

    # Save summary (for comparison notebook compatibility)
    summary_stats = []
    test_metrics = metrics_df[metrics_df['split'] == 'test']
    for K in CONFIG['observation_windows']:
        k_metrics = test_metrics[test_metrics['K'] == K].iloc[0]
        summary_stats.append({
            'K': K,
            'f1_mean': k_metrics['f1'],
            'f1_std': 0.0,  # No std for single run
            'auc_mean': k_metrics['auc'],
            'auc_std': 0.0,
            'precision_mean': k_metrics['precision'],
            'precision_std': 0.0,
            'recall_mean': k_metrics['recall'],
            'recall_std': 0.0,
            'accuracy_mean': k_metrics['accuracy'],
            'accuracy_std': 0.0
        })

    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(model_dir / 'summary_statistics.csv', index=False)

    print(f"Saved {model_name} results to {model_dir}")

print("\n✅ All baseline results saved!")
```

---

## 4. Update results_comparison.ipynb

Update the loading section to handle both multi-seed and single-run results:

```python
# Define result directories for each model
results_dirs = {
    'Temporal GCN': Path('../../results/evolve_gcn_multi_seed'),
    'Static GCN': Path('../../results/static_gcn_multi_seed'),
    'MLP + Graph Features': Path('../../results/graph_features_baseline'),  # Single run
    'Logistic Regression': Path('../../results/baselines/logistic_regression'),  # Single run
    'Random Forest': Path('../../results/baselines/random_forest'),  # Single run
    'XGBoost': Path('../../results/baselines/xgboost')  # Single run
}

# Load all results
all_models_summary = {}

for model_name, result_dir in results_dirs.items():
    # Try multi-seed format first
    csv_path = result_dir / 'summary_statistics.csv'
    if not csv_path.exists():
        # Try single-seed format
        csv_path = result_dir / 'multi_seed_summary_statistics.csv'

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df['model'] = model_name
        all_models_summary[model_name] = df
        print(f"✓ Loaded {model_name}: {len(df)} K values")
    else:
        print(f"✗ Missing: {csv_path}")

# Combine all results
combined_summary = pd.concat(all_models_summary.values(), ignore_index=True)
print(f"\nTotal models: {len(combined_summary['model'].unique())}")
print(f"K values: {sorted(combined_summary['K'].unique())}")
```

---

## Summary of Changes

### Multi-Seed (with variance):
1. ✅ **evolve_gcn.ipynb** - Already done
2. 🔄 **static_gcn.ipynb** - Add multi-seed loop + feature reduction

### Single Run (no variance):
3. 🔄 **graph_features_baseline.ipynb** - Add enhanced features + result saving
4. 🔄 **baselines.ipynb** - Add result saving only

### Comparison:
5. 🔄 **results_comparison.ipynb** - Handle both formats

---

## Expected File Structure

```
results/
├── evolve_gcn_multi_seed/
│   ├── all_seeds_all_metrics.csv
│   ├── multi_seed_summary_statistics.csv
│   └── ... (predictions, per-seed files)
│
├── static_gcn_multi_seed/
│   ├── all_seeds_all_metrics.csv
│   ├── multi_seed_summary_statistics.csv
│   └── ... (predictions, per-seed files)
│
├── graph_features_baseline/
│   ├── all_metrics.csv
│   └── summary_statistics.csv  (std=0 for all)
│
├── baselines/
│   ├── logistic_regression/
│   │   ├── all_metrics.csv
│   │   └── summary_statistics.csv
│   ├── random_forest/
│   │   ├── all_metrics.csv
│   │   └── summary_statistics.csv
│   └── xgboost/
│       ├── all_metrics.csv
│       └── summary_statistics.csv
│
├── all_models_comparison.csv
└── comparison_summary_statistics.csv
```

---

## Why This Makes Sense

**Multi-seed for GNNs**:
- Neural networks are sensitive to initialization
- Training dynamics vary with random seeds
- Important to measure stability

**Single run for traditional ML**:
- Deterministic algorithms (especially tree-based)
- Much faster to train
- Less variance from initialization
- MLP also relatively stable

**Result**: Fair comparison with appropriate uncertainty quantification where it matters most.
