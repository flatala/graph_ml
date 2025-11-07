# Multi-Seed EvolveGCN Results

This directory contains results from multi-seed training experiments.

## File Structure

### Per-seed predictions (NPZ files):
- `seed{seed}_k{K}_{split}_predictions.npz`
  - Contains detailed predictions for each node
  - Fields:
    - `node_indices`: Node indices in the graph
    - `predictions`: Binary class predictions (0 or 1)
    - `probs_class_0`: Probability of class 0 (licit)
    - `probs_class_1`: Probability of class 1 (illicit)
    - `true_labels`: Ground truth labels
    - `timesteps`: Timestep for each node

### Metrics files:
- `seed{seed}_metrics.csv`: All metrics for a specific seed
- `all_seeds_all_metrics.csv`: Combined metrics from all seeds
- `multi_seed_summary_statistics.csv`: Mean and std statistics across seeds

## Usage Example

```python
import numpy as np
import pandas as pd

# Load predictions for a specific seed/K/split
data = np.load('seed42_k5_test_predictions.npz')
probs = data['probs_class_1']  # Probability of illicit class
labels = data['true_labels']

# Calculate F1 with custom threshold
from sklearn.metrics import f1_score
threshold = 0.3
custom_preds = (probs >= threshold).astype(int)
f1 = f1_score(labels, custom_preds)
print(f"F1 with threshold {threshold}: {f1:.4f}")

# Load summary statistics
summary = pd.read_csv('multi_seed_summary_statistics.csv')
print(summary)
```

## Seeds Used
[42, 123, 456]

## Observation Windows (K)
[1, 3, 5, 7]

## Training Configuration
- Epochs: 350
- Patience: 350
- Hidden dim: 128
- Learning rate: 0.0002
- Dropout: 0.3
