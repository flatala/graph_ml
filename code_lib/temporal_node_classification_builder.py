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
                graph = torch.load(cache_path)
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
            
            print(f"  Val:   timesteps {val_timesteps[0]}-{val_timesteps[1]}, "
                  f"{len(val_nodes)} nodes")
            print(f"    Illicit: {(val_nodes['class']==1).sum()}, "
                  f"Licit: {(val_nodes['class']==2).sum()}")
            
            print(f"  Test:  timesteps {test_timesteps[0]}-{test_timesteps[1]}, "
                  f"{len(test_nodes)} nodes")
            print(f"    Illicit: {(test_nodes['class']==1).sum()}, "
                  f"Licit: {(test_nodes['class']==2).sum()}")
        
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
