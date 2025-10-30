"""
Temporal Edge Builder for weighted transaction graphs.

This module implements temporal edge weighting with:
1. Same-step aggregation: Sum multiple transactions between same node pair within timestep
2. Temporal decay: Weight past transactions with exponential decay

Mathematical formulation:
Step 1: A_{ji}^{(s)} = Σ g(v_k) where g(v) = log(1 + v/σ_s)
Step 2: S_{ji}(t) = Σ_{s≤t} A_{ji}^{(s)} * exp(-λ(t-s))

Compatible with TemporalNodeClassificationBuilder for building weighted temporal graphs.
"""

import pandas as pd
import numpy as np
import torch
from typing import Dict, Tuple, Optional, Union
from tqdm import tqdm
import warnings


class TemporalEdgeBuilder:
    """
    Builder for temporal edge weights with aggregation and decay.
    
    Implements two-step process:
    1. Same-step sum: Aggregate transactions within each timestep
    2. Temporal decay: Weight historical transactions with exponential decay
    """
    
    def __init__(
        self,
        raw_edges_df: pd.DataFrame,
        value_column: str = 'total_BTC',
        timestep_column: str = 'Time step',
        src_column: str = 'input_address',
        dst_column: str = 'output_address',
        use_log_transform: bool = True,
        decay_lambda: float = 0.1,
        verbose: bool = True
    ):
        """
        Initialize the temporal edge builder.
        
        Args:
            raw_edges_df: DataFrame with individual transactions
            value_column: Column name containing transaction values
            timestep_column: Column name containing timesteps
            src_column: Column name for source addresses
            dst_column: Column name for destination addresses
            use_log_transform: Whether to apply log(1 + v/σ) transformation
            decay_lambda: Decay parameter λ for exponential decay
            verbose: Whether to print progress information
        """
        self.raw_edges_df = raw_edges_df.copy()
        self.value_column = value_column
        self.timestep_column = timestep_column
        self.src_column = src_column
        self.dst_column = dst_column
        self.use_log_transform = use_log_transform
        self.decay_lambda = decay_lambda
        self.verbose = verbose
        
        # Validate required columns
        required_cols = [value_column, timestep_column, src_column, dst_column]
        missing_cols = [col for col in required_cols if col not in raw_edges_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Get timestep range
        self.min_timestep = int(raw_edges_df[timestep_column].min())
        self.max_timestep = int(raw_edges_df[timestep_column].max())
        
        if self.verbose:
            print("Initialized TemporalEdgeBuilder")
            print(f"  Raw transactions: {len(raw_edges_df):,}")
            print(f"  Timestep range: {self.min_timestep} to {self.max_timestep}")
            print(f"  Value column: {value_column}")
            print(f"  Use log transform: {use_log_transform}")
            print(f"  Decay lambda: {decay_lambda}")
    
    def _compute_scale_parameters(self) -> Dict[int, float]:
        """
        Compute scale parameter σ_s for each timestep (median transaction value).
        
        Returns:
            Dictionary mapping timestep -> scale parameter
        """
        if self.verbose:
            print("Computing scale parameters (median transaction values per timestep)...")
        
        scale_params = {}
        for timestep in range(self.min_timestep, self.max_timestep + 1):
            timestep_data = self.raw_edges_df[
                self.raw_edges_df[self.timestep_column] == timestep
            ]
            
            if len(timestep_data) > 0:
                median_value = timestep_data[self.value_column].median()
                # Avoid division by zero
                scale_params[timestep] = max(median_value, 1e-8)
            else:
                # No transactions at this timestep
                scale_params[timestep] = 1.0
        
        if self.verbose:
            sample_scales = list(scale_params.values())[:5]
            print(f"  Sample scale parameters: {sample_scales}")
        
        return scale_params
    
    def _aggregate_same_timestep_transactions(
        self,
        timestep: int,
        scale_params: Dict[int, float]
    ) -> pd.DataFrame:
        """
        Step 1: Aggregate transactions within the same timestep.
        
        For each (src, dst) pair at timestep s:
        A_{dst,src}^{(s)} = Σ g(v_k) where g(v) = log(1 + v/σ_s) or just v
        
        Args:
            timestep: Target timestep
            scale_params: Scale parameters for log transformation
            
        Returns:
            DataFrame with columns [src_address, dst_address, aggregated_value]
        """
        # Get transactions for this timestep
        timestep_data = self.raw_edges_df[
            self.raw_edges_df[self.timestep_column] == timestep
        ].copy()
        
        if len(timestep_data) == 0:
            # No transactions at this timestep
            return pd.DataFrame(columns=[self.src_column, self.dst_column, 'aggregated_value'])
        
        # Apply transformation g(v)
        if self.use_log_transform:
            sigma_s = scale_params[timestep]
            timestep_data['transformed_value'] = np.log(
                1 + timestep_data[self.value_column] / sigma_s
            )
        else:
            timestep_data['transformed_value'] = timestep_data[self.value_column]
        
        # Aggregate by (src, dst) pair
        aggregated = timestep_data.groupby([self.src_column, self.dst_column]).agg({
            'transformed_value': 'sum'
        }).reset_index()
        
        aggregated = aggregated.rename(columns={'transformed_value': 'aggregated_value'})
        
        return aggregated
    
    def _compute_temporal_decay_weights(
        self,
        target_timestep: int,
        all_aggregated: Dict[int, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Step 2: Combine aggregated values across timesteps with exponential decay.
        
        For each (src, dst) pair:
        S_{dst,src}(t) = Σ_{s≤t} A_{dst,src}^{(s)} * exp(-λ(t-s))
        
        Args:
            target_timestep: Target timestep t
            all_aggregated: Dictionary mapping timestep -> aggregated DataFrame
            
        Returns:
            DataFrame with columns [src_address, dst_address, temporal_weight]
        """
        # Collect all unique (src, dst) pairs up to target timestep
        all_pairs = []
        
        for s in range(self.min_timestep, target_timestep + 1):
            if s in all_aggregated and len(all_aggregated[s]) > 0:
                pairs_at_s = all_aggregated[s][[self.src_column, self.dst_column]].copy()
                pairs_at_s['timestep'] = s
                all_pairs.append(pairs_at_s)
        
        if not all_pairs:
            # No transactions up to this timestep
            return pd.DataFrame(columns=[self.src_column, self.dst_column, 'temporal_weight'])
        
        # Combine all pairs
        all_pairs_df = pd.concat(all_pairs, ignore_index=True)
        unique_pairs = all_pairs_df[[self.src_column, self.dst_column]].drop_duplicates()
        
        # Compute temporal weights for each unique pair
        temporal_weights = []
        
        for _, row in unique_pairs.iterrows():
            src_addr = row[self.src_column]
            dst_addr = row[self.dst_column]
            
            total_weight = 0.0
            
            # Sum over all timesteps s ≤ t where this pair had transactions
            for s in range(self.min_timestep, target_timestep + 1):
                if s in all_aggregated and len(all_aggregated[s]) > 0:
                    # Find aggregated value for this pair at timestep s
                    pair_data = all_aggregated[s][
                        (all_aggregated[s][self.src_column] == src_addr) &
                        (all_aggregated[s][self.dst_column] == dst_addr)
                    ]
                    
                    if len(pair_data) > 0:
                        aggregated_value = pair_data['aggregated_value'].iloc[0]
                        decay_factor = np.exp(-self.decay_lambda * (target_timestep - s))
                        total_weight += aggregated_value * decay_factor
            
            temporal_weights.append({
                self.src_column: src_addr,
                self.dst_column: dst_addr,
                'temporal_weight': total_weight
            })
        
        return pd.DataFrame(temporal_weights)
    
    def build_edges_at_timestep(self, timestep: int) -> pd.DataFrame:
        """
        Build temporal edge weights for a specific timestep.
        
        Args:
            timestep: Target timestep
            
        Returns:
            DataFrame with columns [input_address, output_address, temporal_weight, Time step]
            Compatible with TemporalNodeClassificationBuilder
        """
        if timestep < self.min_timestep or timestep > self.max_timestep:
            warnings.warn(f"Timestep {timestep} outside data range [{self.min_timestep}, {self.max_timestep}]")
            return pd.DataFrame(columns=[self.src_column, self.dst_column, 'temporal_weight', self.timestep_column])
        
        # Step 1: Compute scale parameters for all timesteps
        scale_params = self._compute_scale_parameters()
        
        # Step 2: Aggregate transactions for each timestep up to target
        all_aggregated = {}
        
        for s in range(self.min_timestep, timestep + 1):
            all_aggregated[s] = self._aggregate_same_timestep_transactions(s, scale_params)
        
        # Step 3: Compute temporal decay weights
        temporal_edges = self._compute_temporal_decay_weights(timestep, all_aggregated)
        
        # Add timestep column and rename for compatibility
        if len(temporal_edges) > 0:
            temporal_edges[self.timestep_column] = timestep
            
            # Rename columns to match expected format
            temporal_edges = temporal_edges.rename(columns={
                self.src_column: 'input_address',
                self.dst_column: 'output_address'
            })
        else:
            # Empty result
            temporal_edges = pd.DataFrame(columns=[
                'input_address', 'output_address', 'temporal_weight', self.timestep_column
            ])
        
        return temporal_edges
    
    def build_temporal_edge_sequence(
        self,
        start_timestep: int,
        end_timestep: int
    ) -> pd.DataFrame:
        """
        Build temporal edge weights for a sequence of timesteps.
        
        Args:
            start_timestep: First timestep (inclusive)
            end_timestep: Last timestep (inclusive)
            
        Returns:
            DataFrame with all temporal edges across timesteps
        """
        all_edges = []
        
        timesteps = range(start_timestep, end_timestep + 1)
        iterator = tqdm(timesteps, desc="Building temporal edges") if self.verbose else timesteps
        
        for t in iterator:
            edges_t = self.build_edges_at_timestep(t)
            if len(edges_t) > 0:
                all_edges.append(edges_t)
        
        if all_edges:
            return pd.concat(all_edges, ignore_index=True)
        else:
            return pd.DataFrame(columns=[
                'input_address', 'output_address', 'temporal_weight', self.timestep_column
            ])
    
    def build_optimized_temporal_edges(
        self,
        target_timestep: int
    ) -> pd.DataFrame:
        """
        Optimized version that computes temporal weights more efficiently.
        
        Uses vectorized operations and avoids nested loops.
        """
        if self.verbose:
            print(f"Building optimized temporal edges for timestep {target_timestep}")
        
        # Get all data up to target timestep
        data_up_to_t = self.raw_edges_df[
            self.raw_edges_df[self.timestep_column] <= target_timestep
        ].copy()
        
        if len(data_up_to_t) == 0:
            return pd.DataFrame(columns=[
                'input_address', 'output_address', 'temporal_weight', self.timestep_column
            ])
        
        # Compute scale parameters
        scale_params = self._compute_scale_parameters()
        
        # Apply transformations vectorized
        if self.use_log_transform:
            data_up_to_t['sigma'] = data_up_to_t[self.timestep_column].map(scale_params)
            data_up_to_t['transformed_value'] = np.log(
                1 + data_up_to_t[self.value_column] / data_up_to_t['sigma']
            )
        else:
            data_up_to_t['transformed_value'] = data_up_to_t[self.value_column]
        
        # Compute decay factors vectorized
        data_up_to_t['decay_factor'] = np.exp(
            -self.decay_lambda * (target_timestep - data_up_to_t[self.timestep_column])
        )
        
        # Apply decay to transformed values
        data_up_to_t['weighted_value'] = (
            data_up_to_t['transformed_value'] * data_up_to_t['decay_factor']
        )
        
        # Aggregate by (src, dst) pair
        temporal_edges = data_up_to_t.groupby([self.src_column, self.dst_column]).agg({
            'weighted_value': 'sum'
        }).reset_index()
        
        temporal_edges = temporal_edges.rename(columns={
            'weighted_value': 'temporal_weight',
            self.src_column: 'input_address',
            self.dst_column: 'output_address'
        })
        
        temporal_edges[self.timestep_column] = target_timestep
        
        return temporal_edges


def create_temporal_edges_from_raw_data(
    raw_edges_df: pd.DataFrame,
    timestep: int,
    value_column: str = 'total_BTC',
    decay_lambda: float = 0.1,
    use_log_transform: bool = True,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Convenience function to create temporal edges from raw transaction data.
    
    Args:
        raw_edges_df: Raw transaction DataFrame
        timestep: Target timestep for edge building
        value_column: Column containing transaction values
        decay_lambda: Temporal decay parameter
        use_log_transform: Whether to use log transformation
        verbose: Print progress information
        
    Returns:
        Temporal edges DataFrame compatible with TemporalNodeClassificationBuilder
    """
    builder = TemporalEdgeBuilder(
        raw_edges_df=raw_edges_df,
        value_column=value_column,
        decay_lambda=decay_lambda,
        use_log_transform=use_log_transform,
        verbose=verbose
    )
    
    return builder.build_optimized_temporal_edges(timestep)


if __name__ == "__main__":
    """Simple example: Create builder → calculate temporal edges → pass to temporal graph builder"""
    import pandas as pd
    from temporal_node_classification_builder import TemporalNodeClassificationBuilder
    
    # Step 1: Sample transaction data
    raw_edges_df = pd.DataFrame([
        {'input_address': 'A', 'output_address': 'B', 'total_BTC': 1.5, 'Time step': 1},
        {'input_address': 'B', 'output_address': 'C', 'total_BTC': 0.8, 'Time step': 2},
        {'input_address': 'A', 'output_address': 'C', 'total_BTC': 2.1, 'Time step': 3},
    ])
    
    # Step 2: Create temporal edge builder
    builder = TemporalEdgeBuilder(raw_edges_df, decay_lambda=0.1, verbose=False)
    
    # Step 3: Calculate temporal edges for timestep 3
    temporal_edges = builder.build_optimized_temporal_edges(target_timestep=3)
    print("Temporal edges with decay weights:")
    print(temporal_edges[['input_address', 'output_address', 'temporal_weight']])
    
    # Step 4: Pass edges to temporal graph builder
    nodes_df = pd.DataFrame([
        {'address': 'A', 'Time step': 1, 'class': 1, 'feature1': 0.5},
        {'address': 'B', 'Time step': 1, 'class': 2, 'feature1': 0.3},
        {'address': 'C', 'Time step': 2, 'class': 1, 'feature1': 0.7},
    ])
    
    graph_builder = TemporalNodeClassificationBuilder(
        nodes_df=nodes_df, 
        edges_df=temporal_edges, 
        add_edge_weights=True,
        edge_weight_col='temporal_weight',
        verbose=False
    )
    
    # Step 5: Build temporal graph with weighted edges
    graph = graph_builder.build_graph_at_timestep(timestep=3)
    
    print(f"\n✅ Created temporal graph:")
    print(f"   Nodes: {graph.num_nodes}, Edges: {graph.edge_index.shape[1]}")
    print(f"   Edge weights: {graph.edge_attr.squeeze()}")
    print(f"   Ready for GNN training!")