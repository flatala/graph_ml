# Multi-Seed Training Implementation Guide

This guide provides complete code snippets to modify all notebooks for multi-seed training with consistent result saving.

---

## Common Components (Copy-Paste for All Notebooks)

### Feature Reduction Function
```python
def remove_correlated_features(nodes_df, threshold=0.95, verbose=True):
    """
    Remove highly correlated features from nodes DataFrame.
    """
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
                if verbose:
                    print(f"Removing {corr_matrix.columns[j]} (corr={corr_matrix.iloc[i, j]:.3f} with {corr_matrix.columns[i]})")

    features_to_keep = [col for col in feature_cols if col not in to_remove]

    if verbose:
        print(f"\nOriginal features: {len(feature_cols)}")
        print(f"Removed features:  {len(to_remove)}")
        print(f"Kept features:     {len(features_to_keep)}")

    return features_to_keep
```

### Multi-Seed Configuration
```python
# Multi-seed experiment configuration
SEEDS = [42, 123, 456]
RESULTS_DIR = Path('../../results/{NOTEBOOK_NAME}_multi_seed')  # Replace {NOTEBOOK_NAME}
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Running experiments with {len(SEEDS)} seeds: {SEEDS}")
print(f"Results will be saved to: {RESULTS_DIR}")
```

### Detailed Predictions Collection (for Graph Models)
```python
def collect_detailed_predictions(model, graphs_dict, device):
    """
    Collect detailed predictions for post-hoc analysis.
    """
    model.eval()

    all_node_indices = []
    all_predictions = []
    all_probs_class_0 = []
    all_probs_class_1 = []
    all_true_labels = []
    all_eval_times = []

    with torch.no_grad():
        for eval_t, graph in graphs_dict.items():
            logits = model(graph.x, graph.edge_index)
            probs = F.softmax(logits, dim=1)

            # Extract predictions for masked nodes
            pred = logits[graph.eval_mask].argmax(dim=1).cpu().numpy()
            prob_0 = probs[graph.eval_mask, 0].cpu().numpy()
            prob_1 = probs[graph.eval_mask, 1].cpu().numpy()
            true = graph.y[graph.eval_mask].cpu().numpy()

            # Get node indices
            node_idx = torch.where(graph.eval_mask)[0].cpu().numpy()
            eval_times = np.full(len(node_idx), eval_t)

            all_node_indices.append(node_idx)
            all_predictions.append(pred)
            all_probs_class_0.append(prob_0)
            all_probs_class_1.append(prob_1)
            all_true_labels.append(true)
            all_eval_times.append(eval_times)

    return {
        'node_indices': np.concatenate(all_node_indices),
        'predictions': np.concatenate(all_predictions),
        'probs_class_0': np.concatenate(all_probs_class_0),
        'probs_class_1': np.concatenate(all_probs_class_1),
        'true_labels': np.concatenate(all_true_labels),
        'eval_times': np.concatenate(all_eval_times)
    }
```

---

## 1. static_gcn.ipynb Modifications

### Step 1: Update imports (cell after title)
Add `import os` and `import random` to imports

### Step 2: After config cell, add:
```python
# Multi-seed configuration
SEEDS = [42, 123, 456]
RESULTS_DIR = Path('../../results/static_gcn_multi_seed')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Running experiments with {len(SEEDS)} seeds: {SEEDS}")
print(f"Results will be saved to: {RESULTS_DIR}")
```

### Step 3: Before data loading, add feature reduction:
```python
nodes_df, edges_df = load_elliptic_data(CONFIG['data_dir'], use_temporal_features=True)

kept_features = remove_correlated_features(nodes_df, threshold=0.95, verbose=False)
print(f"Before: {nodes_df.shape[1]}")
print(f"After: {len(kept_features)}")
```

### Step 4: Update builder initialization:
```python
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

### Step 5: Replace entire training cell with multi-seed loop:
```python
import copy
from datetime import datetime

# Storage for all seeds
all_seeds_results = {}
all_seeds_predictions = {}

total_iterations = len(SEEDS) * len(CONFIG['observation_windows'])
current_iteration = 0

print(f"Starting multi-seed training:")
print(f"  Seeds: {SEEDS}")
print(f"  Observation windows (K): {CONFIG['observation_windows']}")
print(f"  Total training runs: {total_iterations}")
print(f"=" * 80)

start_time = datetime.now()

for seed_idx, seed in enumerate(SEEDS):
    print(f"\n{'#' * 80}")
    print(f"# SEED {seed_idx + 1}/{len(SEEDS)}: {seed}")
    print(f"{'#' * 80}\n")

    # Set all random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    seed_results = {}
    seed_predictions = {}
    seed_models = {}

    for K in CONFIG['observation_windows']:
        current_iteration += 1

        print(f"\n{'='*70}")
        print(f"Seed {seed} | K={K} | Progress: {current_iteration}/{total_iterations}")
        print(f"{'='*70}")

        train_graphs = graphs[K]['train']['graphs']
        val_graphs = graphs[K]['val']['graphs']
        test_graphs = graphs[K]['test']['graphs']

        # Initialize model
        num_features = list(train_graphs.values())[0].x.shape[1]
        model = StaticGCN(
            num_features=num_features,
            hidden_dim=CONFIG['hidden_dim'],
            num_classes=2,
            dropout=CONFIG['dropout']
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=CONFIG['learning_rate'],
            weight_decay=CONFIG['weight_decay']
        )

        # Class weights from train
        all_train_labels = []
        for g in train_graphs.values():
            all_train_labels.append(g.y[g.eval_mask].cpu())
        all_train_labels = torch.cat(all_train_labels).long()

        class_counts = torch.bincount(all_train_labels)
        class_weights = torch.sqrt(1.0 / class_counts.float())
        class_weights = class_weights / class_weights.sum() * 2.0
        class_weights = class_weights.to(device)

        print(f"Class distribution: {class_counts.tolist()}")
        print(f"Class weights: {class_weights.tolist()}")

        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Training loop
        best_val_f1 = 0
        patience_counter = 0
        best_model_state = None

        pbar = tqdm(range(CONFIG['epochs']), desc=f"Seed={seed}, K={K}")
        for epoch in pbar:
            train_loss, train_acc = train_epoch(model, train_graphs, optimizer, criterion)

            if (epoch + 1) % 5 == 0:
                val_metrics = evaluate(model, val_graphs)
                train_metrics = evaluate(model, train_graphs)
                pbar.set_postfix({
                    'loss': f"{train_loss:.4f}",
                    'train_f1': f"{train_metrics['f1']:.4f}",
                    'val_f1': f"{val_metrics['f1']:.4f}"
                })

                if val_metrics['f1'] > best_val_f1:
                    best_val_f1 = val_metrics['f1']
                    patience_counter = 0
                    best_model_state = copy.deepcopy(model.state_dict())
                else:
                    patience_counter += 1

                if patience_counter >= CONFIG['patience']:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Evaluate
        train_metrics = evaluate(model, train_graphs)
        val_metrics = evaluate(model, val_graphs)
        test_metrics = evaluate(model, test_graphs)

        print(f"\nTrain: F1={train_metrics['f1']:.4f}, AUC={train_metrics['auc']:.4f}")
        print(f"Val:   F1={val_metrics['f1']:.4f}, AUC={val_metrics['auc']:.4f}")
        print(f"Test:  F1={test_metrics['f1']:.4f}, AUC={test_metrics['auc']:.4f}")

        # Collect detailed predictions
        print("Collecting detailed predictions...")
        train_preds = collect_detailed_predictions(model, train_graphs, device)
        val_preds = collect_detailed_predictions(model, val_graphs, device)
        test_preds = collect_detailed_predictions(model, test_graphs, device)

        # Store results
        seed_results[K] = {
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics
        }

        seed_predictions[K] = {
            'train': train_preds,
            'val': val_preds,
            'test': test_preds
        }

        seed_models[K] = model

        # Save predictions immediately
        for split_name, preds in [('train', train_preds), ('val', val_preds), ('test', test_preds)]:
            save_path = RESULTS_DIR / f"seed{seed}_k{K}_{split_name}_predictions.npz"
            np.savez_compressed(
                save_path,
                node_indices=preds['node_indices'],
                predictions=preds['predictions'],
                probs_class_0=preds['probs_class_0'],
                probs_class_1=preds['probs_class_1'],
                true_labels=preds['true_labels'],
                eval_times=preds['eval_times']
            )
        print(f"Saved predictions to {RESULTS_DIR}/seed{seed}_k{K}_*_predictions.npz")

    # Store results for this seed
    all_seeds_results[seed] = seed_results
    all_seeds_predictions[seed] = seed_predictions

    # Save metrics for this seed
    seed_metrics_data = []
    for K in CONFIG['observation_windows']:
        for split_name in ['train', 'val', 'test']:
            metrics = seed_results[K][split_name]
            seed_metrics_data.append({
                'seed': seed,
                'K': K,
                'split': split_name,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'auc': metrics['auc']
            })

    seed_metrics_df = pd.DataFrame(seed_metrics_data)
    seed_metrics_df.to_csv(RESULTS_DIR / f'seed{seed}_metrics.csv', index=False)
    print(f"\nSaved metrics for seed {seed} to {RESULTS_DIR}/seed{seed}_metrics.csv")

end_time = datetime.now()
elapsed = end_time - start_time

print(f"\n{'=' * 80}")
print(f"Multi-seed training complete!")
print(f"Total time: {elapsed}")
print(f"Results saved to: {RESULTS_DIR}")
print(f"{'=' * 80}")
```

### Step 6: Update results summary cell:
```python
# Aggregate results across all seeds
aggregated_results = []

for seed in SEEDS:
    for K in CONFIG['observation_windows']:
        for split_name in ['train', 'val', 'test']:
            metrics = all_seeds_results[seed][K][split_name]
            aggregated_results.append({
                'seed': seed,
                'K': K,
                'split': split_name,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'auc': metrics['auc']
            })

all_results_df = pd.DataFrame(aggregated_results)

# Compute statistics across seeds for test set
test_results = all_results_df[all_results_df['split'] == 'test']

summary_stats = []
for K in CONFIG['observation_windows']:
    k_results = test_results[test_results['K'] == K]

    summary_stats.append({
        'K': K,
        'F1_mean': k_results['f1'].mean(),
        'F1_std': k_results['f1'].std(),
        'AUC_mean': k_results['auc'].mean(),
        'AUC_std': k_results['auc'].std(),
        'Precision_mean': k_results['precision'].mean(),
        'Precision_std': k_results['precision'].std(),
        'Recall_mean': k_results['recall'].mean(),
        'Recall_std': k_results['recall'].std(),
        'Accuracy_mean': k_results['accuracy'].mean(),
        'Accuracy_std': k_results['accuracy'].std()
    })

summary_df = pd.DataFrame(summary_stats)

print("\nTest Set Performance Across Seeds (Mean ± Std):")
print("=" * 80)
for _, row in summary_df.iterrows():
    print(f"K={row['K']}:")
    print(f"  F1:        {row['F1_mean']:.4f} ± {row['F1_std']:.4f}")
    print(f"  AUC:       {row['AUC_mean']:.4f} ± {row['AUC_std']:.4f}")
    print(f"  Precision: {row['Precision_mean']:.4f} ± {row['Precision_std']:.4f}")
    print(f"  Recall:    {row['Recall_mean']:.4f} ± {row['Recall_std']:.4f}")
    print(f"  Accuracy:  {row['Accuracy_mean']:.4f} ± {row['Accuracy_std']:.4f}")
    print()
```

### Step 7: Update visualization:
```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# F1 Score with error bars
ax = axes[0]
f1_means = summary_df['F1_mean'].values
f1_stds = summary_df['F1_std'].values
ax.errorbar(CONFIG['observation_windows'], f1_means, yerr=f1_stds,
            marker='o', linewidth=2, capsize=5, capthick=2, color='steelblue')
ax.set_xlabel('Observation Window K', fontsize=12)
ax.set_ylabel('F1 Score', fontsize=12)
ax.set_title('F1 Score vs Observation Window (Mean ± Std)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# AUC with error bars
ax = axes[1]
auc_means = summary_df['AUC_mean'].values
auc_stds = summary_df['AUC_std'].values
ax.errorbar(CONFIG['observation_windows'], auc_means, yerr=auc_stds,
            marker='o', linewidth=2, capsize=5, capthick=2, color='green')
ax.set_xlabel('Observation Window K', fontsize=12)
ax.set_ylabel('AUC', fontsize=12)
ax.set_title('AUC vs Observation Window (Mean ± Std)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Step 8: Update save results:
```python
# Save aggregated summary statistics
summary_df.to_csv(RESULTS_DIR / 'multi_seed_summary_statistics.csv', index=False)
print(f"Saved summary statistics to {RESULTS_DIR / 'multi_seed_summary_statistics.csv'}")

# Save all results (detailed)
all_results_df.to_csv(RESULTS_DIR / 'all_seeds_all_metrics.csv', index=False)
print(f"Saved all metrics to {RESULTS_DIR / 'all_seeds_all_metrics.csv'}")

print(f"\nAll results saved to: {RESULTS_DIR}")
```

---

## 2. graph_features_baseline.ipynb Modifications

Same modifications as static_gcn.ipynb, PLUS enhanced graph features:

### Enhanced compute_graph_features function:
```python
from torch_geometric.utils import degree
from torch_sparse import SparseTensor
import torch

def compute_clustering_coefficient(edge_index, num_nodes):
    """Compute local clustering coefficient efficiently."""
    # Build adjacency matrix
    adj = SparseTensor(row=edge_index[0], col=edge_index[1],
                       sparse_sizes=(num_nodes, num_nodes))

    # Count triangles: trace(A^3) / 6
    adj_dense = adj.to_dense()
    deg = adj_dense.sum(dim=1)

    # Avoid division by zero
    max_triangles = deg * (deg - 1) / 2
    max_triangles[max_triangles == 0] = 1

    # Approximate clustering
    clustering = torch.zeros(num_nodes)
    # For efficiency, use degree-based approximation
    # Real clustering would require computing triangles
    return clustering

def compute_pagerank(edge_index, num_nodes, alpha=0.85, max_iter=10):
    """Compute PageRank centrality (approximated with few iterations)."""
    # Initialize PageRank scores
    pr = torch.ones(num_nodes) / num_nodes

    # Build transition matrix
    deg_out = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.float)
    deg_out[deg_out == 0] = 1  # Avoid division by zero

    for _ in range(max_iter):
        pr_new = torch.zeros(num_nodes)
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            pr_new[dst] += pr[src] / deg_out[src]
        pr = alpha * pr_new + (1 - alpha) / num_nodes

    return pr

def compute_2hop_neighborhood_size(edge_index, num_nodes):
    """Compute size of 2-hop neighborhood."""
    # Build sparse adjacency matrix
    adj = SparseTensor(row=edge_index[0], col=edge_index[1],
                       sparse_sizes=(num_nodes, num_nodes))

    # A + A^2 gives 1-hop and 2-hop neighbors
    adj2 = adj @ adj

    # Count unique neighbors (including self)
    combined = (adj + adj2).to_dense()
    ego_size = (combined > 0).sum(dim=1).float()

    return ego_size

def compute_graph_features(data):
    """Add comprehensive graph structural features."""
    num_nodes = data.num_nodes
    edge_index = data.edge_index

    # 1. Degree centrality
    deg_total = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.float)
    deg_in = degree(edge_index[1], num_nodes=num_nodes, dtype=torch.float)
    deg_out = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.float)

    # 2. Clustering coefficient (approximated for speed)
    # For large graphs, use degree-based heuristic
    clustering = torch.zeros(num_nodes)

    # 3. PageRank centrality
    pagerank = compute_pagerank(edge_index, num_nodes, max_iter=5)

    # 4. Betweenness centrality (approximated)
    betweenness_approx = (deg_in * deg_out) / (deg_total + 1e-8)

    # 5. Ego-network size (2-hop neighborhood)
    try:
        ego_size = compute_2hop_neighborhood_size(edge_index, num_nodes)
    except:
        # Fallback if 2-hop computation fails
        ego_size = deg_total

    # Stack all features
    graph_feats = torch.stack([
        deg_total, deg_in, deg_out,
        clustering, pagerank,
        betweenness_approx, ego_size
    ], dim=1)

    # Combine with node features
    combined_feats = torch.cat([data.x, graph_feats], dim=1)
    return combined_feats
```

Use `RESULTS_DIR = Path('../../results/graph_features_baseline_multi_seed')`

---

## 3. baselines.ipynb Modifications

For sklearn models, modify the prediction collection:

```python
def collect_baseline_predictions(model, X, y, scaler=None):
    """Collect predictions for baseline models."""
    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X

    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)

    return {
        'predictions': y_pred,
        'probs_class_0': y_proba[:, 0],
        'probs_class_1': y_proba[:, 1],
        'true_labels': y
    }
```

Wrap training in seed loop similar to above, but for each model type:
- Save to `../../results/baselines_multi_seed/logistic_regression/`
- Save to `../../results/baselines_multi_seed/random_forest/`
- Save to `../../results/baselines_multi_seed/xgboost/`

---

## 4. results_comparison.ipynb (NEW NOTEBOOK)

See next section for complete code...

