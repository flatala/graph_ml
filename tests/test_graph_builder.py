"""
Comprehensive test suite for graph_builder.py

Run from project root: python -m tests.test_graph_builder
Or with pytest: pytest tests/test_graph_builder.py -v
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from collections import Counter
from torch_geometric.data import Data

# Add project root to path (parent of tests/)
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from code_lib.graph_builder import build_emergence_graphs_for_time_range
from code_lib.utils import load_parts


class GraphBuilderTester:
    """Test suite for graph builder functionality"""
    
    def __init__(self, data_dir="../elliptic_dataset"):
        """Initialize tester with data directory"""
        self.data_dir = data_dir
        self.nodes = None
        self.node_labels = None
        self.edges = None
        self.nodes_with_labels = None
        self.test_results = []
        
    def load_data(self):
        """Load the Elliptic dataset"""
        print("\n" + "="*70)
        print("LOADING DATA")
        print("="*70)
        
        try:
            # Load data files
            wallets_features = os.path.join(self.data_dir, "wallets_features.csv")
            wallets_classes = os.path.join(self.data_dir, "wallets_classes.csv")
            edges_prefix = "AddrTxAddr_edgelist_part_"
            
            print(f"Loading from: {os.path.abspath(self.data_dir)}")
            
            self.nodes = pd.read_csv(wallets_features)
            self.node_labels = pd.read_csv(wallets_classes)
            self.edges = load_parts(self.data_dir, edges_prefix)
            self.nodes_with_labels = self.nodes.merge(self.node_labels, on='address', how='left')
            
            print(f"✅ Loaded {len(self.nodes)} nodes")
            print(f"✅ Loaded {len(self.edges)} edges")
            print(f"✅ Time steps: {sorted(self.edges['Time step'].unique())[:10]}...")
            
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def test_basic_sanity_checks(self, graphs, test_name="Basic Sanity"):
        """Test 1: Basic sanity checks on generated graphs"""
        print("\n" + "="*70)
        print(f"TEST 1: {test_name} Checks")
        print("="*70)
        
        passed = True
        
        # Check 1.1: Graph count
        try:
            expected_count = len(graphs)
            assert len(graphs) > 0, "No graphs generated"
            print(f"✅ Check 1.1: Generated {len(graphs)} graphs")
        except AssertionError as e:
            print(f"❌ Check 1.1 FAILED: {e}")
            passed = False
        
        # Check 1.2: All are valid Data objects
        try:
            for i, g in enumerate(graphs):
                assert isinstance(g, Data), f"Graph {i} is not a Data object"
                assert hasattr(g, 'x'), f"Graph {i} missing node features"
                assert hasattr(g, 'edge_index'), f"Graph {i} missing edge_index"
                assert hasattr(g, 'y'), f"Graph {i} missing labels"
            print(f"✅ Check 1.2: All graphs are valid Data objects")
        except AssertionError as e:
            print(f"❌ Check 1.2 FAILED: {e}")
            passed = False
        
        # Check 1.3: Cumulative growth
        try:
            for i in range(1, len(graphs)):
                assert graphs[i].num_nodes >= graphs[i-1].num_nodes, \
                    f"Graph {i} has fewer nodes ({graphs[i].num_nodes}) than graph {i-1} ({graphs[i-1].num_nodes})"
                assert graphs[i].edge_index.shape[1] >= graphs[i-1].edge_index.shape[1], \
                    f"Graph {i} has fewer edges than graph {i-1}"
            print(f"✅ Check 1.3: Graphs grow cumulatively over time")
        except AssertionError as e:
            print(f"❌ Check 1.3 FAILED: {e}")
            passed = False
        
        # Check 1.4: Feature validity
        try:
            # Dynamically determine expected features from first graph
            expected_features = graphs[0].x.shape[1]
            for i, graph in enumerate(graphs):
                assert graph.x.shape[1] == expected_features, \
                    f"Graph {i}: Expected {expected_features} features, got {graph.x.shape[1]}"
                assert not torch.isnan(graph.x).any(), f"Graph {i} contains NaN features"
                assert not torch.isinf(graph.x).any(), f"Graph {i} contains Inf features"
            print(f"✅ Check 1.4: Features valid (shape={expected_features}, no NaN/Inf)")
        except AssertionError as e:
            print(f"❌ Check 1.4 FAILED: {e}")
            passed = False
        
        # Check 1.5: Edge validity
        try:
            for i, graph in enumerate(graphs):
                edge_index = graph.edge_index
                assert edge_index.shape[0] == 2, f"Graph {i}: edge_index should have 2 rows"
                max_node = graph.num_nodes - 1
                assert edge_index.max() <= max_node, \
                    f"Graph {i}: Edge references node {edge_index.max()} but only {graph.num_nodes} nodes exist"
                assert edge_index.min() >= 0, f"Graph {i}: Negative node index found"
            print(f"✅ Check 1.5: Edge structure is valid")
        except AssertionError as e:
            print(f"❌ Check 1.5 FAILED: {e}")
            passed = False
        
        self.test_results.append(("Basic Sanity Checks", passed))
        return passed
    
    def test_label_consistency(self, binary_graphs, distance_graphs):
        """Test 2: Binary and distance labels should be consistent"""
        print("\n" + "="*70)
        print("TEST 2: Label Consistency (Binary vs Distance)")
        print("="*70)
        
        passed = True
        total_mismatches = 0
        
        try:
            assert len(binary_graphs) == len(distance_graphs), \
                f"Different number of graphs: binary={len(binary_graphs)}, distance={len(distance_graphs)}"
            
            for i in range(len(binary_graphs)):
                binary_labels = binary_graphs[i].y
                distance_labels = distance_graphs[i].y
                
                # Binary: 0 (no emergence) or 1 (emergence)
                # Distance: 0 (direct transactor), 1-k (distance to illicit), -1 (unreachable = no emergence)
                # 
                # Mapping: 
                #   binary==1 should match distance>=0 (includes 0, 1, 2, ...)
                #   binary==0 should match distance==-1 (unreachable = no emergence)
                binary_positive = (binary_labels == 1)
                distance_positive = (distance_labels >= 0)  # Fixed: -1 means no emergence, so only >=0 is positive
                
                matches = (binary_positive == distance_positive).sum().item()
                total = len(binary_labels)
                mismatches = total - matches
                total_mismatches += mismatches
                
                if matches == total:
                    print(f"✅ t={i+1}: Labels consistent ({matches}/{total})")
                else:
                    print(f"⚠️  t={i+1}: {mismatches} mismatches ({100*mismatches/total:.2f}%)")
                    print(f"     Binary pos: {binary_positive.sum().item()}, Distance pos: {distance_positive.sum().item()}")
                    passed = False
            
            if passed:
                print(f"\n✅ All labels consistent across {len(binary_graphs)} graphs")
            else:
                print(f"\n⚠️  Total mismatches: {total_mismatches}")
                
        except Exception as e:
            print(f"❌ TEST 2 FAILED: {e}")
            passed = False
        
        self.test_results.append(("Label Consistency", passed))
        return passed
    
    def test_temporal_logic(self):
        """Test 3: Temporal logic - different horizons should affect labels"""
        print("\n" + "="*70)
        print("TEST 3: Temporal Logic (Time Horizon Effects)")
        print("="*70)
        
        passed = True
        
        try:
            # Build with different time horizons
            print("Building graphs with time_horizon=1...")
            graphs_h1 = build_emergence_graphs_for_time_range(
                edges_with_labels_df=self.edges,
                nodes_with_classes_df=self.nodes_with_labels,
                first_time_step=1,
                last_time_step=5,
                max_walk_length=2,
                time_horizon=1,
                use_distance_labels=False,
                keep_class_labels_as_features=False,
                ignore_illict=True,
                ignore_previously_transacting_with_illicit=True
            )
            
            print("Building graphs with time_horizon=5...")
            graphs_h5 = build_emergence_graphs_for_time_range(
                edges_with_labels_df=self.edges,
                nodes_with_classes_df=self.nodes_with_labels,
                first_time_step=1,
                last_time_step=5,
                max_walk_length=2,
                time_horizon=5,
                use_distance_labels=False,
                keep_class_labels_as_features=False,
                ignore_illict=True,
                ignore_previously_transacting_with_illicit=True
            )
            
            # Compare positive label counts
            print("\nTime Horizon Comparison:")
            print(f"{'t':<5} {'h=1':<10} {'h=5':<10} {'Ratio':<10}")
            print("-" * 40)
            
            for i in range(len(graphs_h1)):
                pos_h1 = (graphs_h1[i].y == 1).sum().item()
                pos_h5 = (graphs_h5[i].y == 1).sum().item()
                ratio = pos_h5 / max(pos_h1, 1)
                print(f"{i+1:<5} {pos_h1:<10} {pos_h5:<10} {ratio:<10.2f}x")
            
            # Longer horizon should generally have >= positive labels
            total_h1 = sum((g.y == 1).sum().item() for g in graphs_h1)
            total_h5 = sum((g.y == 1).sum().item() for g in graphs_h5)
            
            print(f"\nTotal positive labels: h=1 has {total_h1}, h=5 has {total_h5}")
            
            if total_h5 >= total_h1:
                print("✅ Longer horizon has more/equal positive labels (expected)")
            else:
                print("⚠️  Shorter horizon has more positive labels (unusual but not necessarily wrong)")
            
            # This is a soft check - doesn't fail the test
            print("✅ Temporal logic test completed")
            
        except Exception as e:
            print(f"❌ TEST 3 FAILED: {e}")
            passed = False
        
        self.test_results.append(("Temporal Logic", passed))
        return passed
    
    def test_performance(self):
        """Test 4: Performance benchmarks"""
        print("\n" + "="*70)
        print("TEST 4: Performance Benchmarks")
        print("="*70)
        
        passed = True
        
        try:
            # Binary labels
            print("Benchmarking binary labels (10 graphs)...")
            start = time.time()
            binary_graphs = build_emergence_graphs_for_time_range(
                edges_with_labels_df=self.edges,
                nodes_with_classes_df=self.nodes_with_labels,
                first_time_step=1,
                last_time_step=10,
                max_walk_length=2,
                time_horizon=3,
                use_distance_labels=False,
                keep_class_labels_as_features=False,
                ignore_illict=True,
                ignore_previously_transacting_with_illicit=True
            )
            binary_time = time.time() - start
            
            # Distance labels
            print("Benchmarking distance labels (10 graphs)...")
            start = time.time()
            distance_graphs = build_emergence_graphs_for_time_range(
                edges_with_labels_df=self.edges,
                nodes_with_classes_df=self.nodes_with_labels,
                first_time_step=1,
                last_time_step=10,
                max_walk_length=2,
                time_horizon=3,
                use_distance_labels=True,
                keep_class_labels_as_features=False,
                ignore_illict=True,
                ignore_previously_transacting_with_illicit=True
            )
            distance_time = time.time() - start
            
            print(f"\nPerformance Results:")
            print(f"  Binary labels:   {binary_time:.2f}s ({binary_time/10:.2f}s per graph)")
            print(f"  Distance labels: {distance_time:.2f}s ({distance_time/10:.2f}s per graph)")
            print(f"  Distance slowdown: {distance_time/binary_time:.2f}x")
            
            # Expected: distance should be <5x slower (after optimization)
            if distance_time / binary_time < 5:
                print("✅ Performance is good (distance <5x slower)")
            else:
                print("⚠️  Distance labels are very slow (>5x slower than binary)")
                print("    This may indicate optimization issues")
            
            # Estimate full 46 graphs
            est_binary_46 = (binary_time / 10) * 46
            est_distance_46 = (distance_time / 10) * 46
            print(f"\nEstimated time for 46 graphs:")
            print(f"  Binary:   {est_binary_46:.1f}s (~{est_binary_46/60:.1f} min)")
            print(f"  Distance: {est_distance_46:.1f}s (~{est_distance_46/60:.1f} min)")
            
        except Exception as e:
            print(f"❌ TEST 4 FAILED: {e}")
            passed = False
        
        self.test_results.append(("Performance", passed))
        return passed
    
    def test_statistics(self, graphs):
        """Test 5: Statistical validation"""
        print("\n" + "="*70)
        print("TEST 5: Statistical Validation")
        print("="*70)
        
        passed = True
        
        try:
            stats = []
            for i, graph in enumerate(graphs):
                positive = (graph.y == 1).sum().item()
                negative = (graph.y == 0).sum().item()
                total = positive + negative
                
                stats.append({
                    't': i+1,
                    'nodes': graph.num_nodes,
                    'edges': graph.edge_index.shape[1] // 2,
                    'pos': positive,
                    'neg': negative,
                    'pos%': f"{100*positive/total:.1f}%"
                })
            
            df = pd.DataFrame(stats)
            print("\nGraph Statistics:")
            print(df.to_string(index=False))
            
            # Validation checks
            print(f"\n📊 Statistical Summary:")
            print(f"  Nodes range: {df['nodes'].min()} → {df['nodes'].max()}")
            print(f"  Edges range: {df['edges'].min()} → {df['edges'].max()}")
            
            avg_pos_ratio = df['pos'].sum() / (df['pos'].sum() + df['neg'].sum()) * 100
            print(f"  Average positive ratio: {avg_pos_ratio:.2f}%")
            
            # Check cumulative growth
            is_cumulative = (df['nodes'].diff().dropna() >= 0).all()
            if is_cumulative:
                print(f"  ✅ Graphs are cumulative (nodes always increase)")
            else:
                print(f"  ❌ Graphs are NOT cumulative (nodes decrease somewhere)")
                passed = False
            
            # Check label ratio is reasonable (1-30%)
            if 1 <= avg_pos_ratio <= 30:
                print(f"  ✅ Positive label ratio is reasonable ({avg_pos_ratio:.1f}%)")
            else:
                print(f"  ⚠️  Unusual positive label ratio ({avg_pos_ratio:.1f}%)")
                print(f"     Expected: 1-30%")
            
        except Exception as e:
            print(f"❌ TEST 5 FAILED: {e}")
            passed = False
        
        self.test_results.append(("Statistical Validation", passed))
        return passed
    
    def print_summary(self):
        """Print final test summary"""
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for _, passed in self.test_results if passed)
        
        for test_name, passed in self.test_results:
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"{test_name:<30} {status}")
        
        print("="*70)
        print(f"Results: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("\n🎉 ALL TESTS PASSED! Your graph builder is working correctly.")
            return True
        else:
            print(f"\n⚠️  {total_tests - passed_tests} test(s) failed. Please review the output above.")
            return False
    
    def run_all_tests(self):
        """Run the complete test suite"""
        print("\n" + "="*70)
        print("GRAPH BUILDER TEST SUITE")
        print("="*70)
        print(f"Test file: {__file__}")
        print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load data
        if not self.load_data():
            print("❌ Failed to load data. Exiting.")
            return False
        
        # Build test graphs
        print("\n" + "="*70)
        print("BUILDING TEST GRAPHS")
        print("="*70)
        
        print("Building binary label graphs (10 time steps)...")
        binary_graphs = build_emergence_graphs_for_time_range(
            edges_with_labels_df=self.edges,
            nodes_with_classes_df=self.nodes_with_labels,
            first_time_step=1,
            last_time_step=10,
            max_walk_length=2,
            time_horizon=3,
            use_distance_labels=False,
            keep_class_labels_as_features=False,
            ignore_illict=True,
            ignore_previously_transacting_with_illicit=True
        )
        
        print("Building distance label graphs (10 time steps)...")
        distance_graphs = build_emergence_graphs_for_time_range(
            edges_with_labels_df=self.edges,
            nodes_with_classes_df=self.nodes_with_labels,
            first_time_step=1,
            last_time_step=10,
            max_walk_length=2,
            time_horizon=3,
            use_distance_labels=True,
            keep_class_labels_as_features=False,
            ignore_illict=True,
            ignore_previously_transacting_with_illicit=True
        )
        
        # Run tests
        self.test_basic_sanity_checks(binary_graphs, "Binary Graphs")
        self.test_basic_sanity_checks(distance_graphs, "Distance Graphs")
        self.test_label_consistency(binary_graphs, distance_graphs)
        self.test_temporal_logic()
        self.test_performance()
        self.test_statistics(binary_graphs)
        
        # Print summary
        return self.print_summary()


def main():
    """Main entry point"""
    # Adjust data directory - we're now in tests/ subfolder
    # So elliptic_dataset is in parent directory
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "elliptic_dataset"
    
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        print("Please ensure elliptic_dataset/ exists in the project root.")
        return False
    
    # Create tester and run all tests
    tester = GraphBuilderTester(data_dir=str(data_dir))
    success = tester.run_all_tests()
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
