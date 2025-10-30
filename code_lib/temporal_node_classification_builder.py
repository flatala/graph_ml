"""
Graph builder for Temporal Node Classification task.

This module builds cumulative temporal graphs for the task of classifying nodes
based on limited observation windows after their first appearance. Designed for
fraud detection scenarios where nodes (wallets) need to be classified quickly
after their first transaction.

Key characteristics:
- Cumulative graphs: nodes and edges persist after first appearance
- Node features remain static (from first appearance)
- Graph structure evolves as new nodes/edges are added
- Labels are static (nodes don't change class over time)

Use case: Classify Bitcoin wallets as illicit/licit after observing K timesteps
"""

import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm


class TemporalNodeClassificationBuilder:
    """
    Builder for temporal node classification graphs.
    
    Creates cumulative graphs where nodes and edges persist over time,
    optimized for classifying nodes at their first appearance with optional
    observation windows.
    """
    
    def __init__(
        self,
        nodes_df: pd.DataFrame,
        edges_df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        include_class_as_feature: bool = False,
        add_temporal_features: bool = True,
        add_edge_weights: bool = False,
        edge_weight_col: Optional[str] = None,
        verbose: bool = True,
        cache_dir: Optional[str] = None,
        use_cache: bool = True
    ):
        """
        Initialize the graph builder.
        
        Args:
            nodes_df: DataFrame with columns ['address', 'Time step', 'class', ...features]
            edges_df: DataFrame with columns ['Time step', 'input_address', 'output_address', ...]
            feature_cols: List of column names to use as features. If None, uses all numeric columns
            include_class_as_feature: Whether to include 'class' label as a node feature
            add_temporal_features: Whether to add temporal features (age, degree over time)
            add_edge_weights: Whether to include edge weights in the graph
            edge_weight_col: Column name for edge weights. If None and add_edge_weights=True, 
                           uses edge count (number of transactions between address pair)
            verbose: Whether to print progress information
            cache_dir: Directory to store cached graphs. If None, uses './graph_cache'
            use_cache: Whether to use caching for built graphs
        """
        self.nodes_df = nodes_df.copy()
        self.edges_df = edges_df.copy()
        self.verbose = verbose
        self.include_class_as_feature = include_class_as_feature
        self.add_temporal_features = add_temporal_features
        self.add_edge_weights = add_edge_weights
        self.edge_weight_col = edge_weight_col
        self.use_cache = use_cache
        
        # Setup cache directory
        if cache_dir is None:
            cache_dir = './graph_cache'
        self.cache_dir = cache_dir
        if use_cache and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            if self.verbose:
                print(f"Created cache directory: {cache_dir}")
        
        # Determine feature columns
        if feature_cols is None:
            # Use all numeric columns except address, Time step, and optionally class
            exclude_cols = {'address', 'Time step'}
            if not include_class_as_feature:
                exclude_cols.add('class')
            self.feature_cols = [col for col in nodes_df.columns 
                                if col not in exclude_cols and 
                                pd.api.types.is_numeric_dtype(nodes_df[col])]
        else:
            self.feature_cols = feature_cols
        
        # Pre-compute useful mappings
        self._preprocess_data()
        
        if self.verbose:
            print("Initialized TemporalNodeClassificationBuilder")
            print(f"  Total nodes: {len(self.all_addresses)}")
            print(f"  Total edges: {len(self.edges_df)}")
            print(f"  Time steps: {self.min_timestep} to {self.max_timestep}")
            print(f"  Feature columns ({len(self.feature_cols)}): {self.feature_cols[:5]}...")
            print(f"  Include class as feature: {self.include_class_as_feature}")
            print(f"  Add temporal features: {self.add_temporal_features}")
            print(f"  Add edge weights: {self.add_edge_weights}")
            if self.add_edge_weights and self.edge_weight_col:
                print(f"  Edge weight column: {self.edge_weight_col}")
    
    def _preprocess_data(self):
        """Pre-compute mappings and statistics for efficient graph building."""
        # Get all unique addresses
        self.all_addresses = self.nodes_df['address'].unique()
        
        # Time step range
        self.min_timestep = int(self.nodes_df['Time step'].min())
        self.max_timestep = int(self.nodes_df['Time step'].max())
        
        # Get first appearance time for each node
        self.first_appearance = self.nodes_df.groupby('address')['Time step'].min().to_dict()
        
        # Get class labels (static, don't change over time)
        self.node_classes = self.nodes_df.groupby('address')['class'].first().to_dict()
        
        # Pre-group nodes by address and timestep for O(1) lookup
        # This is important because wallets_features_until_t.csv has multiple rows per address
        if self.verbose:
            print("  Pre-processing node features by (address, timestep)...")
        
        # Create a multi-index for fast lookup
        self.nodes_by_addr_time = self.nodes_df.set_index(['address', 'Time step'])
        
        # Pre-group edges by timestep for O(1) lookup
        if self.verbose:
            print("  Pre-processing edges by timestep...")
        self.edges_by_timestep = {
            t: group for t, group in self.edges_df.groupby('Time step')
        }
        
        # Count nodes appearing at each timestep  
        first_appearances = self.nodes_df.groupby('address')['Time step'].min()
        nodes_per_timestep = first_appearances.value_counts().sort_index()
        if self.verbose:
            print(f"  Average new nodes per timestep: {nodes_per_timestep.mean():.1f}")
    
    def _get_cache_path(self, timestep: int, return_node_metadata: bool) -> str:
        """Generate cache file path for a specific graph."""
        # Create a unique identifier based on configuration
        config_str = f"t{timestep}_meta{return_node_metadata}_class{self.include_class_as_feature}_temp{self.add_temporal_features}_weights{self.add_edge_weights}"
        return os.path.join(self.cache_dir, f"graph_{config_str}.pt")
    
    def _save_graph_to_cache(self, graph: Data, timestep: int, return_node_metadata: bool):
        """Save a graph to cache using PyTorch's native format."""
        if not self.use_cache:
            return
        
        cache_path = self._get_cache_path(timestep, return_node_metadata)
        try:
            torch.save(graph, cache_path)
            if self.verbose:
                print(f"  💾 Cached graph to {cache_path}")
        except Exception as e:
            if self.verbose:
                print(f"  ⚠️  Failed to cache graph: {e}")
    
    def _load_graph_from_cache(self, timestep: int, return_node_metadata: bool) -> Optional[Data]:
        """Load a graph from cache using PyTorch's native format."""
        if not self.use_cache:
            return None

        cache_path = self._get_cache_path(timestep, return_node_metadata)
        if os.path.exists(cache_path):
            try:
                # PyTorch 2.6+ requires weights_only=False for custom objects like PyG Data
                graph = torch.load(cache_path, weights_only=False)
                if self.verbose:
                    print(f"  ✅ Loaded cached graph from {cache_path}")
                return graph
            except Exception as e:
                if self.verbose:
                    print(f"  ⚠️  Failed to load cached graph: {e}")
                return None
        return None
    
    def clear_cache(self):
        """Clear all cached graphs."""
        if not os.path.exists(self.cache_dir):
            return
        
        cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.pt')]
        for cache_file in cache_files:
            os.remove(os.path.join(self.cache_dir, cache_file))
        
        if self.verbose:
            print(f"🗑️  Cleared {len(cache_files)} cached graphs from {self.cache_dir}")
    
    def build_graph_at_timestep(
        self,
        timestep: int,
        return_node_metadata: bool = True
    ) -> Data:
        """
        Build cumulative graph at a specific timestep.
        
        The graph includes all nodes that have appeared up to this timestep,
        and all edges that have occurred up to this timestep.
        
        Args:
            timestep: Target timestep
            return_node_metadata: Whether to include additional node metadata
                                (first_appearance, original_address)
        
        Returns:
            PyG Data object with:
                - x: node features [num_nodes, num_features]
                - edge_index: edge indices [2, num_edges]
                - y: node class labels [num_nodes] (1=illicit, 2=licit, 3=unknown)
                - num_nodes: number of nodes
                - timestep: current timestep
                - (optional) edge_attr: edge weights [num_edges, 1] if add_edge_weights=True
                - (optional) node_first_appearance: first appearance time [num_nodes]
                - (optional) node_address: original addresses [num_nodes]
        """
        # Try to load from cache first
        cached_graph = self._load_graph_from_cache(timestep, return_node_metadata)
        if cached_graph is not None:
            return cached_graph
        # OPTIMIZATION 1: Use numpy array and vectorized operations for active addresses
        first_appearance_array = np.array(list(self.first_appearance.items()), dtype=object)
        active_mask = first_appearance_array[:, 1].astype(int) <= timestep
        active_addresses = first_appearance_array[active_mask, 0]
        
        # Create address to local index mapping
        addr_to_idx = {addr: idx for idx, addr in enumerate(active_addresses)}
        num_nodes = len(active_addresses)
        
        # OPTIMIZATION 2: Build edge index using vectorized operations and pre-grouped edges
        edge_src_list = []
        edge_dst_list = []
        edge_weight_list = [] if self.add_edge_weights else None
        
        for t in range(self.min_timestep, timestep + 1):
            if t in self.edges_by_timestep:
                edges_t = self.edges_by_timestep[t]
                
                # Vectorized filtering: check if both endpoints are in active addresses
                src_in_active = edges_t['input_address'].isin(addr_to_idx.keys())
                dst_in_active = edges_t['output_address'].isin(addr_to_idx.keys())
                valid_mask = src_in_active & dst_in_active
                
                if valid_mask.any():
                    valid_edges = edges_t[valid_mask]
                    
                    # Vectorized mapping to local indices
                    src_indices = valid_edges['input_address'].map(addr_to_idx).values
                    dst_indices = valid_edges['output_address'].map(addr_to_idx).values
                    
                    edge_src_list.append(src_indices)
                    edge_dst_list.append(dst_indices)
                    
                    # Add edge weights if requested
                    if self.add_edge_weights:
                        if self.edge_weight_col and self.edge_weight_col in valid_edges.columns:
                            # Use specified column for weights
                            weights = valid_edges[self.edge_weight_col].values
                        else:
                            # Default: all edges have weight 1.0
                            weights = np.ones(len(valid_edges))
                        edge_weight_list.append(weights)
        
        # Convert to tensor (single concatenation instead of growing list)
        if edge_src_list:
            edge_src = np.concatenate(edge_src_list)
            edge_dst = np.concatenate(edge_dst_list)
            edge_index = torch.tensor(np.stack([edge_src, edge_dst], axis=0), dtype=torch.long)
            
            # Add edge weights if requested
            if self.add_edge_weights:
                edge_weights = np.concatenate(edge_weight_list)
                edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)
            else:
                edge_attr = None
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = None
        
        # OPTIMIZATION 3: Vectorized feature extraction using DataFrame operations
        # Filter nodes up to current timestep
        nodes_up_to_t = self.nodes_df[self.nodes_df['Time step'] <= timestep]
        
        # Get most recent features for each active address
        nodes_filtered = nodes_up_to_t[nodes_up_to_t['address'].isin(active_addresses)]
        nodes_sorted = nodes_filtered.sort_values('Time step', ascending=True)
        latest_features = nodes_sorted.groupby('address')[self.feature_cols].last()
        
        # Build feature matrix efficiently
        features_array = latest_features.loc[active_addresses].values
        
        if self.add_temporal_features:
            # Add node age feature (vectorized)
            node_ages = np.array([timestep - self.first_appearance[addr] for addr in active_addresses])
            features_array = np.column_stack([features_array, node_ages])
        
        x = torch.tensor(features_array, dtype=torch.float)
        
        # OPTIMIZATION 4: Vectorized label extraction
        y = torch.tensor([self.node_classes[addr] for addr in active_addresses], 
                        dtype=torch.long)
        
        # Create Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            num_nodes=num_nodes,
            timestep=timestep
        )
        
        # Add edge weights if they exist
        if edge_attr is not None:
            data.edge_attr = edge_attr
        
        # Add metadata if requested
        if return_node_metadata:
            data.node_first_appearance = torch.tensor(
                [self.first_appearance[addr] for addr in active_addresses],
                dtype=torch.long
            )
            data.node_address = list(active_addresses)  # Keep as list for lookup
        
        # Save to cache
        self._save_graph_to_cache(data, timestep, return_node_metadata)
        
        return data
    
    def build_temporal_sequence(
        self,
        start_timestep: int,
        end_timestep: int,
        return_node_metadata: bool = True
    ) -> List[Data]:
        """
        Build a sequence of cumulative graphs over a time range.
        
        Args:
            start_timestep: First timestep (inclusive)
            end_timestep: Last timestep (inclusive)
            return_node_metadata: Whether to include node metadata
        
        Returns:
            List of PyG Data objects, one per timestep
        """
        graphs = []
        timesteps = range(start_timestep, end_timestep + 1)
        
        iterator = tqdm(timesteps, desc="Building graphs") if self.verbose else timesteps
        
        for t in iterator:
            graph = self.build_graph_at_timestep(t, return_node_metadata)
            graphs.append(graph)
            
            if self.verbose and not isinstance(iterator, tqdm):
                if t % 10 == 0:
                    print(f"  Built graph for t={t}: {graph.num_nodes} nodes, "
                          f"{graph.edge_index.shape[1]} edges")
        
        return graphs
    
    def get_train_val_test_split(
        self,
        train_timesteps: Tuple[int, int],
        val_timesteps: Tuple[int, int],
        test_timesteps: Tuple[int, int],
        filter_unknown: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Split nodes by their first appearance timestep for temporal evaluation.
        
        This creates a temporal split where training nodes appear earlier than
        validation nodes, which appear earlier than test nodes. This ensures
        the model is evaluated on genuinely new nodes it hasn't seen during training.
        
        Args:
            train_timesteps: (start, end) range for training nodes
            val_timesteps: (start, end) range for validation nodes  
            test_timesteps: (start, end) range for test nodes
            filter_unknown: Whether to exclude unknown (class 3) nodes
        
        Returns:
            Dict with keys 'train', 'val', 'test', each containing DataFrame with:
                - address: node address
                - first_timestep: when node first appeared
                - class: node label (1, 2, or 3)
        """
        # Get first appearance per node
        node_info = pd.DataFrame({
            'address': list(self.first_appearance.keys()),
            'first_timestep': list(self.first_appearance.values()),
            'class': [self.node_classes[addr] for addr in self.first_appearance.keys()]
        })
        
        # Filter to labeled nodes if requested
        if filter_unknown:
            node_info = node_info[node_info['class'].isin([1, 2])]
        
        # Split by first appearance time
        train_nodes = node_info[
            (node_info['first_timestep'] >= train_timesteps[0]) &
            (node_info['first_timestep'] <= train_timesteps[1])
        ].reset_index(drop=True)
        
        val_nodes = node_info[
            (node_info['first_timestep'] >= val_timesteps[0]) &
            (node_info['first_timestep'] <= val_timesteps[1])
        ].reset_index(drop=True)
        
        test_nodes = node_info[
            (node_info['first_timestep'] >= test_timesteps[0]) &
            (node_info['first_timestep'] <= test_timesteps[1])
        ].reset_index(drop=True)
        
        if self.verbose:
            print("\nTemporal Split Summary:")
            print(f"  Train: timesteps {train_timesteps[0]}-{train_timesteps[1]}, "
                  f"{len(train_nodes)} nodes")
            print(f"    Illicit: {(train_nodes['class']==1).sum()}, "
                  f"Licit: {(train_nodes['class']==2).sum()}")
            print(f"Training illicit ratio: {(train_nodes['class']==1).sum() / len(train_nodes)}")
            
            print(f"  Val:   timesteps {val_timesteps[0]}-{val_timesteps[1]}, "
                  f"{len(val_nodes)} nodes")
            print(f"    Illicit: {(val_nodes['class']==1).sum()}, "
                  f"Licit: {(val_nodes['class']==2).sum()}")
            print(f"Validation illicit ratio: {(val_nodes['class']==1).sum() / len(val_nodes)}")
            
            print(f"  Test:  timesteps {test_timesteps[0]}-{test_timesteps[1]}, "
                  f"{len(test_nodes)} nodes")
            print(f"    Illicit: {(test_nodes['class']==1).sum()}, "
                  f"Licit: {(test_nodes['class']==2).sum()}")
            print(f"Test illicit ratio: {(test_nodes['class']==1).sum() / len(test_nodes)}")
        
        return {
            'train': train_nodes,
            'val': val_nodes,
            'test': test_nodes
        }
    
    def get_node_index_in_graph(
        self,
        graph: Data,
        address: str
    ) -> Optional[int]:
        """
        Get the local node index for an address in a specific graph.
        
        Args:
            graph: PyG Data object with node_address metadata
            address: Node address to look up
        
        Returns:
            Local node index, or None if address not in graph
        """
        if not hasattr(graph, 'node_address'):
            raise ValueError("Graph must have node_address metadata. "
                           "Build with return_node_metadata=True")
        
        try:
            return graph.node_address.index(address)
        except ValueError:
            return None
    
    def get_graph_statistics(self, graph: Data) -> Dict:
        """Get detailed statistics about a graph."""
        stats = {
            'timestep': graph.timestep,
            'num_nodes': graph.num_nodes,
            'num_edges': graph.edge_index.shape[1],
            'num_features': graph.x.shape[1],
        }
        
        # Class distribution
        if hasattr(graph, 'y'):
            unique, counts = torch.unique(graph.y, return_counts=True)
            stats['class_distribution'] = {
                int(cls): int(cnt) for cls, cnt in zip(unique, counts)
            }
        
        # Degree statistics
        if graph.edge_index.shape[1] > 0:
            degrees = torch.zeros(graph.num_nodes, dtype=torch.long)
            degrees.scatter_add_(0, graph.edge_index[0], 
                               torch.ones(graph.edge_index.shape[1], dtype=torch.long))
            degrees.scatter_add_(0, graph.edge_index[1],
                               torch.ones(graph.edge_index.shape[1], dtype=torch.long))
            
            stats['avg_degree'] = float(degrees.float().mean())
            stats['max_degree'] = int(degrees.max())
            stats['min_degree'] = int(degrees.min())
        else:
            stats['avg_degree'] = 0.0
            stats['max_degree'] = 0
            stats['min_degree'] = 0
        
        # Node age statistics (if temporal features added)
        if hasattr(graph, 'node_first_appearance'):
            ages = graph.timestep - graph.node_first_appearance
            stats['avg_node_age'] = float(ages.float().mean())
            stats['max_node_age'] = int(ages.max())
            stats['nodes_just_appeared'] = int((ages == 0).sum())
        
        return stats
    
    def print_graph_summary(self, graph: Data):
        """Print a human-readable summary of a graph."""
        stats = self.get_graph_statistics(graph)
        
        print(f"\n{'='*60}")
        print(f"Graph at Timestep {stats['timestep']}")
        print(f"{'='*60}")
        print(f"Nodes: {stats['num_nodes']:,}")
        print(f"Edges: {stats['num_edges']:,}")
        print(f"Features: {stats['num_features']}")
        print(f"Average degree: {stats['avg_degree']:.2f}")
        
        if 'class_distribution' in stats:
            print("\nClass distribution:")
            class_names = {1: 'Illicit', 2: 'Licit', 3: 'Unknown'}
            for cls, count in stats['class_distribution'].items():
                pct = 100 * count / stats['num_nodes']
                print(f"  {class_names.get(cls, f'Class {cls}')}: "
                      f"{count:,} ({pct:.1f}%)")
        
        if 'avg_node_age' in stats:
            print("\nTemporal statistics:")
            print(f"  Average node age: {stats['avg_node_age']:.2f} timesteps")
            print(f"  Nodes just appeared: {stats['nodes_just_appeared']}")
        
        print(f"{'='*60}")


def load_elliptic_data(data_dir: str, use_temporal_features: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load Elliptic Bitcoin dataset.
    
    Args:
        data_dir: Directory containing the dataset files
        use_temporal_features: If True, uses wallets_features_until_t.csv (no temporal leakage)
                              If False, uses wallets_features_classes_combined.csv (has leakage)
    
    Returns:
        (nodes_df, edges_df) tuple
    """
    from code_lib.utils import load_parts
    
    if use_temporal_features:
        # Load temporal features (computed only up to each timestep - NO LEAKAGE)
        features_df = pd.read_csv(
            os.path.join(data_dir, "wallets_features_until_t.csv")
        )
        
        # Load class labels separately
        classes_df = pd.read_csv(
            os.path.join(data_dir, "wallets_features_classes_combined.csv"),
            usecols=['address', 'class']
        ).drop_duplicates()
        
        # Merge features with classes
        nodes_df = features_df.merge(classes_df, on='address', how='left')
        
        # Fill missing classes with 3 (unknown)
        nodes_df['class'] = nodes_df['class'].fillna(3).astype(int)
        
    else:
        # Load combined features (WARNING: may have temporal leakage)
        nodes_df = pd.read_csv(
            os.path.join(data_dir, "wallets_features_classes_combined.csv")
        )
    
    # Load edge data (using parts)
    edges_df = load_parts(data_dir, "AddrTxAddr_edgelist_part_")
    
    return nodes_df, edges_df


# ==============================================================================
# DEPRECATED FUNCTIONS - DO NOT USE
# ==============================================================================
# The functions below implement INCORRECT logic for observation window experiments.
# They have been commented out to prevent misuse.
#
# Issues:
# 1. prepare_data_for_observation_window: Uses features from max_timestep for all
#    nodes, causing temporal leakage
# 2. prepare_data_for_observation_window_no_leakage: Extracts features at per-node
#    evaluation times, but this is NOT the correct approach for observation windows
#
# Use prepare_observation_window_graphs() instead (see below)
# ==============================================================================

# def prepare_data_for_observation_window(K, builder, train_nodes, val_nodes, test_nodes, device):
#     """DEPRECATED - DO NOT USE - Has temporal leakage"""
#     pass

# def prepare_data_for_observation_window_no_leakage(K, builder, train_nodes, val_nodes, test_nodes, device):
#     """DEPRECATED - DO NOT USE - Incorrect logic for observation windows"""
#     pass


def prepare_observation_window_graphs(
    builder: TemporalNodeClassificationBuilder,
    train_nodes: pd.DataFrame,
    val_nodes: pd.DataFrame,
    test_nodes: pd.DataFrame,
    K_values: List[int],
    device: torch.device
) -> Dict[int, Dict[str, Data]]:
    """
    Prepare graphs for observation window experiments (CORRECT implementation).

    This function creates the proper setup for studying how observation windows affect
    classification performance. For each K value, it builds separate graphs for
    train/val/test splits where:

    1. Graph structure and features come from timestep (split_end + K)
    2. ALL nodes in the graph use features from that timestep
    3. Evaluation mask identifies which nodes to compute loss/metrics on
    4. Only nodes with first_appearance <= split_end are evaluated

    This allows answering: "Does waiting K timesteps after a node appears improve
    classification accuracy for nodes from a specific time period?"

    Key insight: For different K values, you evaluate the SAME nodes (those in the
    split period), but with different amounts of temporal context. This lets you
    measure the value of observation windows.

    Args:
        builder: TemporalNodeClassificationBuilder instance
        train_nodes: DataFrame with columns ['address', 'first_timestep', 'class']
                    Contains nodes that first appeared in train period
        val_nodes: Similar DataFrame for validation nodes
        test_nodes: Similar DataFrame for test nodes
        K_values: List of observation windows to test (e.g., [0, 3, 5, 7])
        device: torch device (cpu/cuda/mps)

    Returns:
        Dictionary mapping K values to split dictionaries:
        {
            0: {
                'train': Data(graph, eval_mask, ...),
                'val': Data(...),
                'test': Data(...)
            },
            3: { ... },
            ...
        }

        Each Data object contains:
        - x: node features [num_nodes, num_features]
        - edge_index: edge indices [2, num_edges]
        - y: node labels [num_nodes] (0=licit, 1=illicit)
        - eval_mask: which nodes to evaluate [num_nodes] (bool tensor)
        - num_nodes: total nodes in graph
        - timestep: the timestep this graph represents
        - node_address: list of addresses (for debugging)
        - node_first_appearance: when each node first appeared

    Example:
        >>> # Setup
        >>> builder = TemporalNodeClassificationBuilder(nodes_df, edges_df)
        >>> split = builder.get_train_val_test_split(
        ...     train_timesteps=(5, 29),
        ...     val_timesteps=(30, 33),
        ...     test_timesteps=(34, 42)
        ... )
        >>>
        >>> # Prepare graphs for different observation windows
        >>> graphs = prepare_observation_window_graphs(
        ...     builder, split['train'], split['val'], split['test'],
        ...     K_values=[0, 3, 5, 7],
        ...     device=torch.device('cpu')
        ... )
        >>>
        >>> # Train model with K=5
        >>> train_data = graphs[5]['train']
        >>> output = model(train_data.x, train_data.edge_index)
        >>> loss = criterion(output[train_data.eval_mask], train_data.y[train_data.eval_mask])
        >>>
        >>> # Compare performance across K values
        >>> for K in [0, 3, 5, 7]:
        ...     test_data = graphs[K]['test']
        ...     # Evaluate only on nodes with first_appearance <= test_end
        ...     accuracy = evaluate(model, test_data, test_data.eval_mask)

    Notes:
        - Uses builder's caching system for efficiency
        - All nodes in graph contribute to GNN message passing
        - Only nodes in eval_mask contribute to loss/metrics
        - This is the CORRECT way to implement observation window experiments
        - Avoids temporal leakage while allowing proper temporal context
    """
    print("\n" + "="*70)
    print("PREPARING OBSERVATION WINDOW GRAPHS")
    print("="*70)

    # Determine split boundaries
    train_end = int(train_nodes['first_timestep'].max())
    val_end = int(val_nodes['first_timestep'].max())
    test_end = int(test_nodes['first_timestep'].max())

    print(f"\nSplit boundaries:")
    print(f"  Train: first_appearance <= {train_end}")
    print(f"  Val:   first_appearance <= {val_end}")
    print(f"  Test:  first_appearance <= {test_end}")
    print(f"\nObservation windows: K = {K_values}")

    # Prepare result dictionary
    results = {}

    # For each observation window K
    for K in K_values:
        print(f"\n{'='*70}")
        print(f"K = {K} (Observe nodes for {K} timesteps after first appearance)")
        print('='*70)

        results[K] = {}

        # Process each split
        for split_name, nodes_df, split_end in [
            ('train', train_nodes, train_end),
            ('val', val_nodes, val_end),
            ('test', test_nodes, test_end)
        ]:
            eval_timestep = split_end + K

            print(f"\n{split_name.upper()} split:")
            print(f"  Split period: nodes with first_appearance <= {split_end}")
            print(f"  Evaluation time: t = {split_end} + {K} = {eval_timestep}")
            print(f"  Building graph at t={eval_timestep}...")

            # Build graph at evaluation timestep
            graph = builder.build_graph_at_timestep(eval_timestep, return_node_metadata=True)

            # Create address to index mapping
            addr_to_idx = {addr: idx for idx, addr in enumerate(graph.node_address)}

            # Create evaluation mask: True for nodes with first_appearance <= split_end
            eval_mask = torch.zeros(graph.num_nodes, dtype=torch.bool)

            eval_count = 0
            for _, node_row in nodes_df.iterrows():
                addr = node_row['address']
                if addr in addr_to_idx:
                    idx = addr_to_idx[addr]
                    eval_mask[idx] = True
                    eval_count += 1

            # Convert labels to 0/1 (licit=0, illicit=1) for binary classification
            y = 2 - graph.y

            # Move to device
            x = graph.x.to(device)
            edge_index = graph.edge_index.to(device)
            y = y.to(device)
            eval_mask = eval_mask.to(device)

            # Create Data object
            data = Data(
                x=x,
                edge_index=edge_index,
                y=y,
                eval_mask=eval_mask,
                num_nodes=graph.num_nodes,
                timestep=eval_timestep,
                node_address=graph.node_address,
                node_first_appearance=graph.node_first_appearance
            )

            # Add edge attributes if they exist
            if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
                data.edge_attr = graph.edge_attr.to(device)

            print(f"  Graph: {graph.num_nodes:,} nodes, {graph.edge_index.shape[1]:,} edges")
            print(f"  Eval nodes: {eval_count:,} ({100*eval_count/graph.num_nodes:.2f}% of graph)")

            # Verify class distribution in eval nodes
            eval_labels = y[eval_mask].cpu().numpy()
            licit_count = (eval_labels == 0).sum()
            illicit_count = (eval_labels == 1).sum()
            print(f"    Licit: {licit_count:,} ({100*licit_count/len(eval_labels):.1f}%)")
            print(f"    Illicit: {illicit_count:,} ({100*illicit_count/len(eval_labels):.1f}%)")

            results[K][split_name] = data

    print("\n" + "="*70)
    print("✅ OBSERVATION WINDOW GRAPHS PREPARED")
    print("="*70)
    print(f"\nCreated graphs for {len(K_values)} observation windows × 3 splits")
    print(f"Total: {len(K_values) * 3} graphs")
    print("\nUsage:")
    print("  train_data = graphs[K]['train']")
    print("  output = model(train_data.x, train_data.edge_index)")
    print("  loss = criterion(output[train_data.eval_mask], train_data.y[train_data.eval_mask])")
    print("="*70 + "\n")

    return results


def prepare_temporal_model_graphs(
    builder: TemporalNodeClassificationBuilder,
    train_nodes: pd.DataFrame,
    val_nodes: pd.DataFrame,
    test_nodes: pd.DataFrame,
    K_values: List[int],
    device: torch.device
) -> Dict[int, Dict[str, Dict]]:
    """
    Prepare graph sequences for temporal GNN models (e.g., EvolveGCN, TGN).

    Unlike static models that use a single graph snapshot, temporal models process
    sequences of graphs over time. This function creates proper sequences where:

    1. Input: Graph sequence from split_start to (split_end + K)
    2. Evaluation masks change over time:
       - For t <= split_end: Evaluate all nodes with first_appearance <= t
       - For t > split_end: Only evaluate nodes with first_appearance <= split_end

    This setup allows temporal models to learn from the evolution of the graph structure
    and node features over time, while maintaining proper temporal evaluation boundaries.

    Args:
        builder: TemporalNodeClassificationBuilder instance
        train_nodes: DataFrame with columns ['address', 'first_timestep', 'class']
        val_nodes: DataFrame for validation nodes
        test_nodes: DataFrame for test nodes
        K_values: List of observation windows (e.g., [0, 3, 5, 7])
        device: torch device (cpu/cuda/mps)

    Returns:
        Dictionary mapping K values to split dictionaries:
        {
            0: {
                'train': {
                    'graphs': [Data, Data, ...],  # List of graphs from train_start to train_end+K
                    'split_start': 5,
                    'split_end': 29,
                    'sequence_length': 25  # Number of timesteps
                },
                'val': {...},
                'test': {...}
            },
            3: {...},
            ...
        }

        Each Data object in the sequence contains:
        - x: node features [num_nodes, num_features]
        - edge_index: edge indices [2, num_edges]
        - y: node labels [num_nodes] (0=licit, 1=illicit)
        - eval_mask: which nodes to evaluate at this timestep [num_nodes]
        - num_nodes: total nodes in graph
        - timestep: the timestep this graph represents
        - node_address: list of addresses
        - node_first_appearance: when each node first appeared

    Example:
        >>> # Setup
        >>> builder = TemporalNodeClassificationBuilder(nodes_df, edges_df)
        >>> split = builder.get_train_val_test_split(
        ...     train_timesteps=(5, 29),
        ...     val_timesteps=(30, 33),
        ...     test_timesteps=(34, 42)
        ... )
        >>>
        >>> # Prepare temporal sequences
        >>> sequences = prepare_temporal_model_graphs(
        ...     builder, split['train'], split['val'], split['test'],
        ...     K_values=[0, 5],
        ...     device=torch.device('cpu')
        ... )
        >>>
        >>> # Train temporal model with K=5
        >>> train_seq = sequences[5]['train']
        >>> for t, graph in enumerate(train_seq['graphs']):
        ...     # Process graph at timestep t
        ...     output = model(graph.x, graph.edge_index)
        ...     # Only compute loss for nodes in eval_mask
        ...     loss = criterion(output[graph.eval_mask], graph.y[graph.eval_mask])

    Notes:
        - Temporal models see the entire sequence during training
        - Evaluation masks ensure only appropriate nodes are evaluated at each timestep
        - For t > split_end, only nodes from the split period are evaluated
        - This tests if temporal evolution helps classification
    """
    print("\n" + "="*70)
    print("PREPARING TEMPORAL MODEL GRAPH SEQUENCES")
    print("="*70)

    # Determine split boundaries
    train_start = int(train_nodes['first_timestep'].min())
    train_end = int(train_nodes['first_timestep'].max())
    val_start = int(val_nodes['first_timestep'].min())
    val_end = int(val_nodes['first_timestep'].max())
    test_start = int(test_nodes['first_timestep'].min())
    test_end = int(test_nodes['first_timestep'].max())

    print(f"\nSplit boundaries:")
    print(f"  Train: t={train_start} to t={train_end}")
    print(f"  Val:   t={val_start} to t={val_end}")
    print(f"  Test:  t={test_start} to t={test_end}")
    print(f"\nObservation windows: K = {K_values}")

    results = {}

    for K in K_values:
        print(f"\n{'='*70}")
        print(f"K = {K} (Observation window)")
        print('='*70)

        results[K] = {}

        for split_name, nodes_df, split_start, split_end in [
            ('train', train_nodes, train_start, train_end),
            ('val', val_nodes, val_start, val_end),
            ('test', test_nodes, test_start, test_end)
        ]:
            # Sequence goes from split_start to split_end + K
            sequence_end = split_end + K
            timesteps = list(range(split_start, sequence_end + 1))

            print(f"\n{split_name.upper()} split:")
            print(f"  Sequence: t={split_start} to t={sequence_end} ({len(timesteps)} timesteps)")
            print(f"  Split period: nodes with first_appearance <= {split_end}")

            # Build graphs for each timestep in sequence
            graphs_sequence = []

            for t in timesteps:
                # Build graph at timestep t
                graph = builder.build_graph_at_timestep(t, return_node_metadata=True)

                # Create address to index mapping
                addr_to_idx = {addr: idx for idx, addr in enumerate(graph.node_address)}

                # Create evaluation mask based on timestep
                eval_mask = torch.zeros(graph.num_nodes, dtype=torch.bool)

                if t <= split_end:
                    # For timesteps within split period: evaluate all nodes that appeared by t
                    eval_count = 0
                    for addr in graph.node_address:
                        idx = addr_to_idx[addr]
                        if builder.first_appearance[addr] <= t:
                            eval_mask[idx] = True
                            eval_count += 1
                else:
                    # For timesteps in observation window: only evaluate nodes from split period
                    eval_count = 0
                    for _, node_row in nodes_df.iterrows():
                        addr = node_row['address']
                        if addr in addr_to_idx:
                            idx = addr_to_idx[addr]
                            eval_mask[idx] = True
                            eval_count += 1

                # Convert labels to 0/1 (licit=0, illicit=1)
                y = 2 - graph.y

                # Move to device
                x = graph.x.to(device)
                edge_index = graph.edge_index.to(device)
                y = y.to(device)
                eval_mask = eval_mask.to(device)

                # Create Data object
                data = Data(
                    x=x,
                    edge_index=edge_index,
                    y=y,
                    eval_mask=eval_mask,
                    num_nodes=graph.num_nodes,
                    timestep=t,
                    node_address=graph.node_address,
                    node_first_appearance=graph.node_first_appearance
                )

                # Add edge attributes if they exist
                if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
                    data.edge_attr = graph.edge_attr.to(device)

                graphs_sequence.append(data)

            print(f"  Built {len(graphs_sequence)} graphs")
            print(f"  First graph: {graphs_sequence[0].num_nodes:,} nodes, {graphs_sequence[0].edge_index.shape[1]:,} edges")
            print(f"  Last graph:  {graphs_sequence[-1].num_nodes:,} nodes, {graphs_sequence[-1].edge_index.shape[1]:,} edges")

            # Store sequence
            results[K][split_name] = {
                'graphs': graphs_sequence,
                'split_start': split_start,
                'split_end': split_end,
                'sequence_length': len(graphs_sequence)
            }

    print("\n" + "="*70)
    print("✅ TEMPORAL MODEL SEQUENCES PREPARED")
    print("="*70)
    print(f"\nCreated sequences for {len(K_values)} observation windows × 3 splits")
    print("\nUsage:")
    print("  train_seq = sequences[K]['train']")
    print("  for graph in train_seq['graphs']:")
    print("      output = model(graph.x, graph.edge_index)")
    print("      loss = criterion(output[graph.eval_mask], graph.y[graph.eval_mask])")
    print("="*70 + "\n")

    return results


if __name__ == "__main__":
    """Example usage and testing."""
    
    print("="*60)
    print("Testing TemporalNodeClassificationBuilder")
    print("="*60)
    
    # Load data (use temporal features to avoid leakage)
    data_dir = "../elliptic_dataset"
    nodes_df, edges_df = load_elliptic_data(data_dir, use_temporal_features=True)
    
    print("\nLoaded data:")
    print(f"  Nodes: {nodes_df.shape}")
    print(f"  Edges: {edges_df.shape}")
    
    # Create builder
    builder = TemporalNodeClassificationBuilder(
        nodes_df=nodes_df,
        edges_df=edges_df,
        include_class_as_feature=False,
        add_temporal_features=True,
        verbose=True
    )
    
    # Build a single graph
    print("\n" + "="*60)
    print("Building single graph at t=20")
    print("="*60)
    graph_20 = builder.build_graph_at_timestep(20)
    builder.print_graph_summary(graph_20)
    
    # Build temporal sequence
    print("\n" + "="*60)
    print("Building temporal sequence t=1 to t=10")
    print("="*60)
    graphs = builder.build_temporal_sequence(1, 10, return_node_metadata=True)
    
    print(f"\nBuilt {len(graphs)} graphs")
    for i in [0, 4, 9]:
        print(f"\nGraph {i} (t={graphs[i].timestep}):")
        print(f"  Nodes: {graphs[i].num_nodes}, Edges: {graphs[i].edge_index.shape[1]}")
    
    # Get train/val/test split
    print("\n" + "="*60)
    print("Creating temporal split")
    print("="*60)
    split = builder.get_train_val_test_split(
        train_timesteps=(1, 20),
        val_timesteps=(21, 30),
        test_timesteps=(31, 40),
        filter_unknown=True
    )
    
    print("\nSplit created successfully!")
