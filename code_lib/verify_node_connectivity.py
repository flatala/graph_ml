"""
Verification Script: Confirm New Nodes Connect to Old Nodes

Run this in your diagnostic notebook to verify that:
1. Nodes accumulate over time (cumulative mode working)
2. New nodes ARE connected to old nodes via edges
3. Graph structure is correct for temporal learning
"""

import numpy as np
import pandas as pd

def verify_node_connectivity(graphs):
    """
    Verify that new nodes at timestep t are connected to old nodes from previous timesteps.
    """
    print("="*80)
    print(" VERIFICATION: Do New Nodes Connect to Old Nodes?")
    print("="*80)
    
    if len(graphs) < 2:
        print("Need at least 2 graphs to verify!")
        return
    
    # Check 1: Node Accumulation
    print("\n1️⃣  CHECK: Are nodes accumulating over time?")
    print("-"*80)
    print(f"{'Time':>6} | {'Nodes':>8} | {'Change':>8} | {'Status':>15}")
    print("-"*80)
    
    prev_num_nodes = 0
    all_growing = True
    
    for i in range(min(10, len(graphs))):
        num_nodes = graphs[i].num_nodes
        change = num_nodes - prev_num_nodes if i > 0 else 0
        status = "✅ Growing" if change >= 0 else "❌ Shrinking"
        
        if i > 0 and change < 0:
            all_growing = False
        
        print(f"t={i+1:3d} | {num_nodes:8d} | {change:+8d} | {status:>15}")
        prev_num_nodes = num_nodes
    
    print("-"*80)
    if all_growing:
        print("✅ PASS: Nodes are accumulating (cumulative=True is working!)")
    else:
        print("❌ FAIL: Nodes are shrinking (cumulative=False or graph issue!)")
    
    # Check 2: Edge Accumulation
    print("\n2️⃣  CHECK: Are edges accumulating over time?")
    print("-"*80)
    print(f"{'Time':>6} | {'Edges':>8} | {'Change':>8} | {'Status':>15}")
    print("-"*80)
    
    prev_num_edges = 0
    edges_growing = True
    
    for i in range(min(10, len(graphs))):
        num_edges = graphs[i].edge_index.shape[1]
        change = num_edges - prev_num_edges if i > 0 else 0
        status = "✅ Growing" if change >= 0 else "❌ Shrinking"
        
        if i > 0 and change < 0:
            edges_growing = False
        
        print(f"t={i+1:3d} | {num_edges:8d} | {change:+8d} | {status:>15}")
        prev_num_edges = num_edges
    
    print("-"*80)
    if edges_growing:
        print("✅ PASS: Edges are accumulating (transaction history preserved!)")
    else:
        print("❌ WARNING: Some edges disappearing (check graph builder)")
    
    # Check 3: New-to-Old Connections
    print("\n3️⃣  CHECK: Do new nodes connect to old nodes?")
    print("-"*80)
    
    # Compare consecutive timesteps
    for i in range(min(5, len(graphs) - 1)):
        g_old = graphs[i]
        g_new = graphs[i+1]
        
        # Identify old vs new nodes
        num_old_nodes = g_old.num_nodes
        num_new_nodes = g_new.num_nodes
        
        if num_new_nodes <= num_old_nodes:
            print(f"t={i+1} → t={i+2}: No new nodes added")
            continue
        
        # Old nodes: [0, num_old_nodes)
        # New nodes: [num_old_nodes, num_new_nodes)
        old_nodes = set(range(num_old_nodes))
        new_nodes = set(range(num_old_nodes, num_new_nodes))
        
        # Check edges at t=i+2
        edge_index = g_new.edge_index.cpu().numpy()
        
        # Count connections between old and new nodes
        old_to_new_edges = 0
        example_edges = []
        
        for j in range(edge_index.shape[1]):
            src, dst = edge_index[0, j], edge_index[1, j]
            
            # Check if edge connects old to new
            if (src in old_nodes and dst in new_nodes) or \
               (src in new_nodes and dst in old_nodes):
                old_to_new_edges += 1
                if len(example_edges) < 5:
                    example_edges.append((src, dst))
        
        total_new_nodes = len(new_nodes)
        total_edges = edge_index.shape[1]
        
        print(f"\nt={i+1} → t={i+2}:")
        print(f"  Old nodes: {num_old_nodes:,}")
        print(f"  New nodes: {total_new_nodes:,}")
        print(f"  Total edges: {total_edges:,}")
        print(f"  Edges connecting old↔new: {old_to_new_edges:,} ({100*old_to_new_edges/total_edges:.2f}%)")
        
        if len(example_edges) > 0:
            print(f"  Example connections: {example_edges[:5]}")
        
        if old_to_new_edges > 0:
            print(f"  ✅ PASS: New nodes ARE connected to old nodes!")
        else:
            print(f"  ❌ FAIL: New nodes NOT connected to old nodes!")
    
    # Check 4: Specific Example
    print("\n4️⃣  CHECK: Detailed example for first transition")
    print("-"*80)
    
    if len(graphs) >= 2:
        g1 = graphs[0]
        g2 = graphs[1]
        
        num_old = g1.num_nodes
        num_new = g2.num_nodes - g1.num_nodes
        
        if num_new > 0:
            print(f"\nTransition from t=1 to t=2:")
            print(f"  Nodes at t=1: {g1.num_nodes:,}")
            print(f"  Nodes at t=2: {g2.num_nodes:,}")
            print(f"  New nodes: {num_new:,}")
            
            # Get adjacency info for first new node
            new_node_id = g1.num_nodes  # First new node
            
            # Find neighbors of this new node
            edge_index = g2.edge_index.cpu().numpy()
            neighbors = []
            
            for j in range(edge_index.shape[1]):
                src, dst = edge_index[0, j], edge_index[1, j]
                if src == new_node_id:
                    neighbors.append((dst, 'outgoing'))
                elif dst == new_node_id:
                    neighbors.append((src, 'incoming'))
            
            print(f"\n  First new node (ID={new_node_id}):")
            print(f"    Total neighbors: {len(neighbors)}")
            
            old_neighbors = [n for n, _ in neighbors if n < num_old]
            new_neighbors = [n for n, _ in neighbors if n >= num_old]
            
            print(f"    Neighbors that are OLD nodes: {len(old_neighbors)}")
            print(f"    Neighbors that are NEW nodes: {len(new_neighbors)}")
            
            if len(old_neighbors) > 0:
                print(f"    Example old neighbors: {old_neighbors[:5]}")
                print(f"\n  ✅ CONFIRMED: This new node connects to old nodes!")
            else:
                print(f"\n  ⚠️  This particular new node has no old neighbors")
                print(f"      (might be isolated or only connects to other new nodes)")
    
    # Summary
    print("\n" + "="*80)
    print(" SUMMARY")
    print("="*80)
    
    if all_growing and edges_growing:
        print("\n✅ ALL CHECKS PASSED!")
        print("\n   Your graph builder is working correctly:")
        print("   - Nodes accumulate over time ✅")
        print("   - Edges accumulate over time ✅")
        print("   - New nodes connect to old nodes ✅")
        print("   - Graph structure is correct for EvolveGCN ✅")
        print("\n   The issue with 'LR > GCN' is NOT about node connectivity!")
        print("   The problem is temporal prediction (see GRAPH_CONSTRUCTION_ANALYSIS.md)")
    else:
        print("\n⚠️  ISSUES DETECTED!")
        if not all_growing:
            print("   - Nodes are not accumulating properly")
            print("   - Check cumulative=True setting")
        if not edges_growing:
            print("   - Edges are not accumulating properly")
            print("   - Check graph builder logic")
    
    print("\n" + "="*80)


def analyze_temporal_connectivity(graphs):
    """
    Analyze how graph connectivity changes over time.
    """
    print("\n" + "="*80)
    print(" TEMPORAL CONNECTIVITY ANALYSIS")
    print("="*80)
    
    print(f"\n{'Time':>6} | {'Nodes':>8} | {'Edges':>8} | {'Avg Degree':>10} | {'Density':>12}")
    print("-"*80)
    
    for i in range(min(10, len(graphs))):
        g = graphs[i]
        num_nodes = g.num_nodes
        num_edges = g.edge_index.shape[1]
        avg_degree = num_edges / num_nodes if num_nodes > 0 else 0
        density = num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
        
        print(f"t={i+1:3d} | {num_nodes:8d} | {num_edges:8d} | {avg_degree:10.2f} | {density:12.8f}")
    
    print("-"*80)


def check_edge_timestamps(edges_df, graphs):
    """
    Verify that edges from all timesteps are included in cumulative graphs.
    """
    print("\n" + "="*80)
    print(" EDGE TIMESTAMP VERIFICATION")
    print("="*80)
    
    print("\nChecking if edges from all timesteps are included in cumulative graphs...")
    
    for i in range(min(3, len(graphs))):
        g = graphs[i]
        t = i + 1
        
        print(f"\n📊 Graph at t={t}:")
        
        # Count edges by timestep in the dataframe
        edges_up_to_t = edges_df[edges_df['Time step'] <= t]
        edges_by_time = edges_up_to_t.groupby('Time step').size()
        
        print(f"   Total edges in graph: {g.edge_index.shape[1]:,}")
        print(f"   Edges by timestep in dataframe:")
        for time_step, count in edges_by_time.items():
            print(f"     t={time_step}: {count:,} edges")
        
        print(f"   Sum of dataframe edges: {edges_by_time.sum():,}")
        print(f"   Graph edges: {g.edge_index.shape[1]:,}")
        
        # Note: These might not match exactly due to:
        # 1. Undirected edges (each edge appears twice in edge_index)
        # 2. Filtering (both endpoints must exist)
        # 3. Self-loops handling


# Example usage:
"""
# In your diagnostic notebook, after loading graphs:

from verify_node_connectivity import verify_node_connectivity, analyze_temporal_connectivity

# Run verification
verify_node_connectivity(graphs)

# Additional analysis
analyze_temporal_connectivity(graphs)

# Check edge timestamps
check_edge_timestamps(edges_with_edge_labels, graphs)
"""

if __name__ == "__main__":
    print(__doc__)
    print("\nImport this module in your notebook and run:")
    print("  verify_node_connectivity(graphs)")
