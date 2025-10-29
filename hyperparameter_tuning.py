#!/usr/bin/env python3
"""
Hyperparameter Tuning Utilities for Temporal Edge Weighting

This module provides utilities for tuning the hyperparameters (decay_lambda and temperature_tau)
of the temporal edge weighting system.
"""

import numpy as np
import torch
from code_lib.graph_builder import compute_temporal_edge_weights


def tune_temporal_hyperparameters(edges_up_to_t, address_id_map_until_t, current_time_t,
                                 lambda_range=(0.01, 0.1), tau_range=(0.5, 2.0),
                                 num_lambda_steps=5, num_tau_steps=5,
                                 value_column='total_BTC_sum', timestep_column='Time step_max'):
    """
    Utility function for hyperparameter tuning of temporal edge weights.
    
    This function computes edge weights for different combinations of decay_lambda and 
    temperature_tau to help with hyperparameter selection.
    
    Args:
        edges_up_to_t: DataFrame with edge data up to current time step
        address_id_map_until_t: dict mapping address -> local node ID
        current_time_t: int, current time step for snapshot
        lambda_range: tuple, (min, max) for decay rate λ
        tau_range: tuple, (min, max) for temperature τ  
        num_lambda_steps: int, number of λ values to test
        num_tau_steps: int, number of τ values to test
        value_column: str, column name containing transaction values
        timestep_column: str, column name containing timesteps
        
    Returns:
        tuple: (results_dict, lambda_values, tau_values)
               results_dict[lambda_val][tau_val] = (edge_index, edge_weights)
    """
    
    # Generate parameter grids
    lambda_values = np.logspace(np.log10(lambda_range[0]), np.log10(lambda_range[1]), num_lambda_steps)
    tau_values = np.linspace(tau_range[0], tau_range[1], num_tau_steps)
    
    results = {}
    
    print(f"Tuning temporal hyperparameters...")
    print(f"λ range: {lambda_range} ({num_lambda_steps} values)")
    print(f"τ range: {tau_range} ({num_tau_steps} values)")
    print(f"Total combinations: {len(lambda_values) * len(tau_values)}")
    
    for lambda_val in lambda_values:
        results[lambda_val] = {}
        for tau_val in tau_values:
            edge_index, edge_weights = compute_temporal_edge_weights(
                edges_up_to_t, address_id_map_until_t, current_time_t,
                decay_lambda=lambda_val, temperature_tau=tau_val,
                value_column=value_column, timestep_column=timestep_column
            )
            results[lambda_val][tau_val] = (edge_index, edge_weights)
    
    return results, lambda_values, tau_values


def analyze_edge_weight_statistics(edge_weights, edge_index, description=""):
    """
    Analyze statistical properties of edge weights for hyperparameter selection.
    
    Args:
        edge_weights: torch.Tensor, edge weights
        edge_index: torch.Tensor, edge indices [2, num_edges]
        description: str, description for printing
        
    Returns:
        dict: Dictionary with statistical measures
    """
    
    stats = {}
    
    if len(edge_weights) == 0:
        return {"description": description, "num_edges": 0}
    
    # Basic statistics
    stats["description"] = description
    stats["num_edges"] = len(edge_weights)
    stats["weight_mean"] = float(edge_weights.mean())
    stats["weight_std"] = float(edge_weights.std())
    stats["weight_min"] = float(edge_weights.min())
    stats["weight_max"] = float(edge_weights.max())
    
    # Entropy (measure of weight distribution uniformity)
    # Higher entropy = more uniform, lower entropy = more concentrated
    weights_np = edge_weights.numpy()
    weights_np = weights_np[weights_np > 1e-10]  # Remove near-zero weights
    if len(weights_np) > 0:
        entropy = -np.sum(weights_np * np.log(weights_np + 1e-10))
        stats["entropy"] = float(entropy)
    else:
        stats["entropy"] = 0.0
    
    # Gini coefficient (measure of inequality in weight distribution)
    # 0 = perfect equality, 1 = maximum inequality
    sorted_weights = np.sort(weights_np)
    n = len(sorted_weights)
    if n > 1:
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_weights)) / (n * np.sum(sorted_weights)) - (n + 1) / n
        stats["gini_coefficient"] = float(gini)
    else:
        stats["gini_coefficient"] = 0.0
    
    # Per-destination statistics
    unique_dests = torch.unique(edge_index[1])
    dest_weight_sums = []
    dest_weight_entropies = []
    
    for dest in unique_dests:
        dest_mask = edge_index[1] == dest
        dest_weights = edge_weights[dest_mask]
        dest_weight_sums.append(float(dest_weights.sum()))
        
        # Entropy for this destination
        dest_weights_np = dest_weights.numpy()
        dest_weights_np = dest_weights_np[dest_weights_np > 1e-10]
        if len(dest_weights_np) > 1:
            dest_entropy = -np.sum(dest_weights_np * np.log(dest_weights_np + 1e-10))
            dest_weight_entropies.append(dest_entropy)
    
    if dest_weight_sums:
        stats["dest_weight_sum_mean"] = float(np.mean(dest_weight_sums))
        stats["dest_weight_sum_std"] = float(np.std(dest_weight_sums))
    
    if dest_weight_entropies:
        stats["dest_entropy_mean"] = float(np.mean(dest_weight_entropies))
        stats["dest_entropy_std"] = float(np.std(dest_weight_entropies))
    
    return stats


def print_hyperparameter_analysis(tuning_results, lambda_values, tau_values):
    """
    Print analysis of hyperparameter tuning results.
    
    Args:
        tuning_results: dict, results from tune_temporal_hyperparameters
        lambda_values: array, tested λ values
        tau_values: array, tested τ values
    """
    
    print(f"\nHyperparameter Tuning Analysis")
    print("=" * 50)
    
    # Analyze each combination
    for i, lambda_val in enumerate(lambda_values):
        print(f"\nDecay rate λ = {lambda_val:.3f}:")
        for j, tau_val in enumerate(tau_values):
            edge_index, edge_weights = tuning_results[lambda_val][tau_val]
            
            stats = analyze_edge_weight_statistics(
                edge_weights, edge_index, 
                f"λ={lambda_val:.3f}, τ={tau_val:.1f}"
            )
            
            print(f"  τ={tau_val:.1f}: {stats['num_edges']} edges, "
                  f"entropy={stats['entropy']:.2f}, "
                  f"gini={stats['gini_coefficient']:.3f}, "
                  f"weight_std={stats['weight_std']:.3f}")
    
    # Recommendations
    print(f"\nRecommendations:")
    print(f"- For sharp focus on high-value recent edges: λ ∈ [0.1, 0.5], τ ∈ [0.5, 1.0]")
    print(f"- For balanced temporal decay: λ ∈ [0.03, 0.1], τ ∈ [1.0, 1.5]")
    print(f"- For longer memory: λ ∈ [0.01, 0.05], τ ∈ [1.5, 2.0]")
    print(f"- Monitor entropy (higher = more uniform) and Gini (lower = more equal)")


def demo_hyperparameter_tuning():
    """Demonstrate hyperparameter tuning with sample data."""
    
    import pandas as pd
    
    print("Hyperparameter Tuning Demo")
    print("=" * 30)
    
    # Create sample data
    edges_data = pd.DataFrame({
        'input_address': ['addr_A', 'addr_A', 'addr_B', 'addr_B', 'addr_C', 'addr_A'],
        'output_address': ['addr_B', 'addr_C', 'addr_C', 'addr_D', 'addr_D', 'addr_D'],
        'total_BTC_sum': [1.5, 0.8, 2.1, 0.5, 1.2, 0.9],
        'Time step_max': [1, 1, 2, 2, 3, 3]
    })
    
    # Create address to ID mapping
    unique_addresses = set(edges_data['input_address']).union(set(edges_data['output_address']))
    address_id_map = {addr: i for i, addr in enumerate(sorted(unique_addresses))}
    
    current_time = 3
    
    # Tune hyperparameters
    results, lambda_vals, tau_vals = tune_temporal_hyperparameters(
        edges_data, address_id_map, current_time,
        lambda_range=(0.01, 0.2), tau_range=(0.5, 2.0),
        num_lambda_steps=3, num_tau_steps=3
    )
    
    # Print analysis
    print_hyperparameter_analysis(results, lambda_vals, tau_vals)
    
    return results, lambda_vals, tau_vals


if __name__ == "__main__":
    demo_hyperparameter_tuning()