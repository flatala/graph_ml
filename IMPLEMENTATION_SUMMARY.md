# Temporal Edge Weighting System Implementation

This repository now contains a complete implementation of the temporal edge weighting system for graph neural networks operating on transaction data.

## 🎯 What Was Implemented

The system implements the three-step recipe from your specification:

1. **Within-step sum** → Aggregate multiple transactions in same timestep
2. **Exponential decay** → Apply temporal decay across timesteps  
3. **Temperature-softmax** → Normalize incoming weights to sum to 1 per destination

### Mathematical Implementation

- **Value Transform**: `g(v) = log(1 + v/σₛ)` with robust outlier handling
- **Decay Function**: `S_ji(t) = Σ A_ji^(s) × exp(-λ(t-s))`  
- **Normalization**: `α_ji(t) = exp(S_ji(t)/τ) / Σ exp(S_ℓi(t)/τ)`

## 📁 Files Added/Modified

### Core Implementation
- **`code_lib/graph_builder.py`** - Added 3 main functions:
  - `compute_temporal_edge_weights()` - Core implementation  
  - `compute_temporal_edge_weights_with_defaults()` - With recommended defaults
  - `compute_temporal_edge_weights_from_raw_transactions()` - For raw transaction data

### Testing & Validation
- **`test_temporal_edge_weights.py`** - Comprehensive test suite that verifies:
  - Mathematical properties (non-negative weights, softmax normalization)
  - Causality constraints (only past transactions influence current weights)
  - Consistency between aggregated and raw data
  - Hyperparameter effects

### Integration Examples
- **`integration_example.py`** - Shows how to integrate with existing graph building pipeline
- **`hyperparameter_tuning.py`** - Utilities for tuning λ and τ parameters

### Documentation
- **`TEMPORAL_EDGE_WEIGHTS_DOCS.md`** - Complete documentation with examples and best practices

## 🚀 Quick Start

### Basic Usage

```python
from code_lib.graph_builder import compute_temporal_edge_weights_with_defaults

# Load your edge data (pre-aggregated)
edges_df = pd.read_csv('AddrTxAddr_edgelist_aggregated.csv')

# Create address mapping
addresses = set(edges_df['input_address']).union(set(edges_df['output_address']))
address_id_map = {addr: i for i, addr in enumerate(addresses)}

# Compute temporal weights
current_time = 10
edge_index, edge_weights = compute_temporal_edge_weights_with_defaults(
    edges_df, address_id_map, current_time
)

# Use in PyTorch Geometric
from torch_geometric.data import Data
graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_weights)
```

### Custom Hyperparameters

```python
# High decay for recent focus
edge_index, edge_weights = compute_temporal_edge_weights(
    edges_df, address_id_map, current_time,
    decay_lambda=0.2,    # High decay
    temperature_tau=0.5  # Sharp focus
)

# Long memory
edge_index, edge_weights = compute_temporal_edge_weights(
    edges_df, address_id_map, current_time,  
    decay_lambda=0.01,   # Low decay
    temperature_tau=2.0  # More uniform
)
```

## ✅ Verified Properties

The implementation has been tested and verified to have:

1. **Mathematical Correctness**:
   - ✅ All edge weights are non-negative
   - ✅ Weights sum to 1 for each destination (softmax property)
   - ✅ Temporal decay reduces influence of older transactions
   - ✅ Numerical stability with max-subtraction in softmax

2. **Causality**:
   - ✅ Only transactions with `s ≤ t` influence snapshot at time `t`
   - ✅ No future information leakage

3. **Robustness**:
   - ✅ Handles extreme transaction values via robust transform
   - ✅ Outlier clipping at 99th percentile
   - ✅ Division-by-zero protection

## 🔧 Hyperparameter Recommendations

From the specification and testing:

- **λ (decay rate)**:
  - `[0.01, 0.05]` - Long memory
  - `[0.03, 0.1]` - Balanced (recommended default: 0.05)
  - `[0.1, 0.5]` - Recent focus

- **τ (temperature)**:
  - `[0.5, 1.0]` - Sharp focus on high-value edges
  - `[1.0, 1.5]` - Balanced (recommended default: 1.0)
  - `[1.5, 2.0]` - More uniform distribution

## 🔄 Integration with Existing Code

To integrate with your existing graph building pipeline:

1. **Replace edge index calls**:
   ```python
   # Before
   edge_index = build_edge_index(edges_up_to_t, address_id_map)
   
   # After  
   edge_index, edge_weights = compute_temporal_edge_weights_with_defaults(
       edges_up_to_t, address_id_map, current_time_t
   )
   ```

2. **Add edge weights to Data objects**:
   ```python
   graph_data = Data(
       x=node_features,
       edge_index=edge_index,
       edge_attr=edge_weights.unsqueeze(1),  # Add feature dimension
       y=node_classes
   )
   ```

3. **Use in GNN message passing**:
   - PyTorch Geometric automatically uses `edge_attr` in message passing
   - Edge weights will influence how neighbor information is aggregated

## 🧪 Testing

Run the test suite to verify everything works:

```bash
python test_temporal_edge_weights.py
python integration_example.py
python hyperparameter_tuning.py
```

## 📊 Performance Notes

- **Time Complexity**: O(E × T) where E = edges, T = timesteps
- **Space Complexity**: O(E) for edge storage
- **Memory Usage**: Scales with number of unique edges and timesteps

For large datasets:
- Consider batching by time windows
- Use sparse representations for very large graphs
- Cache computed step medians and address mappings

## 🔬 Next Steps

The system is ready for use. Potential enhancements:

1. **Adaptive Parameters**: Learn λ and τ during training
2. **Multi-scale Decay**: Different decay rates for different transaction types
3. **Attention Mechanisms**: Replace fixed decay with learned attention
4. **GPU Acceleration**: Optimize for large-scale processing

## 🎉 Summary

You now have a complete, tested, and documented temporal edge weighting system that:

- ✅ Implements your exact specification (3-step recipe)
- ✅ Handles both aggregated and raw transaction data
- ✅ Maintains causality and numerical stability
- ✅ Includes comprehensive testing and validation
- ✅ Provides easy integration with existing code
- ✅ Offers hyperparameter tuning utilities
- ✅ Has detailed documentation and examples

The system is ready for use in your GNN pipeline for illicit activity emergence detection!