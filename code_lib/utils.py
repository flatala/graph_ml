import os, glob, re
import pandas as pd
import numpy as np
import torch
import json
from pathlib import Path
from typing import List, Union
from torch_geometric.data import Data


def load_parts(data_dir: str, base: str) -> pd.DataFrame:
    """
    Load and concatenate multiple CSV files matching a pattern.

    Args:
        data_dir: Directory containing the CSV files
        base: Base filename pattern (e.g., "AddrTxAddr_edgelist_part_")

    Returns:
        Concatenated DataFrame
    """
    paths = glob.glob(os.path.join(data_dir, f"{base}*.csv"))
    if not paths:
        raise FileNotFoundError(f"No files found for pattern {base}_part_*.csv in {data_dir}")

    paths.sort(key=lambda p: int(re.search(r'_part_(\d+)\.csv$', p).group(1)))
    return pd.concat((pd.read_csv(p) for p in paths), ignore_index=True)



def graph_to_dict(graph: Data) -> dict:
    """
    Convert a PyTorch Geometric Data object to a JSON-serializable dictionary.

    Args:
        graph: PyTorch Geometric Data object

    Returns:
        Dictionary with graph data
    """
    data_dict = {}

    # Convert all attributes to numpy/native Python types
    for key, value in graph:
        if isinstance(value, torch.Tensor):
            data_dict[key] = value.cpu().numpy().tolist()
        elif isinstance(value, (int, float, str, bool)):
            data_dict[key] = value
        elif isinstance(value, np.ndarray):
            data_dict[key] = value.tolist()
        else:
            # Try to convert to string as fallback
            data_dict[key] = str(value)

    return data_dict


def dict_to_graph(data_dict: dict) -> Data:
    """
    Convert a dictionary back to a PyTorch Geometric Data object.

    Args:
        data_dict: Dictionary with graph data

    Returns:
        PyTorch Geometric Data object
    """
    # Convert lists back to tensors
    converted_dict = {}

    for key, value in data_dict.items():
        if isinstance(value, list):
            # Convert to numpy first, then to tensor
            arr = np.array(value)

            # Determine appropriate dtype
            if key == 'edge_index':
                converted_dict[key] = torch.tensor(arr, dtype=torch.long)
            elif key in ['y', 'node_class']:
                converted_dict[key] = torch.tensor(arr, dtype=torch.long)
            elif key in ['num_nodes', 'time_step']:
                converted_dict[key] = int(value) if not isinstance(value, list) else int(value[0])
            else:
                converted_dict[key] = torch.tensor(arr, dtype=torch.float)
        else:
            converted_dict[key] = value

    return Data(**converted_dict)


def save_graph(graph: Data, filepath: Union[str, Path]) -> None:
    """
    Save a single PyTorch Geometric Data object to a JSON file.

    Args:
        graph: PyTorch Geometric Data object
        filepath: Path where to save the graph (will create parent directories if needed)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    graph_dict = graph_to_dict(graph)

    with open(filepath, 'w') as f:
        json.dump(graph_dict, f)

    print(f"Graph saved to {filepath}")


def load_graph(filepath: Union[str, Path]) -> Data:
    """
    Load a PyTorch Geometric Data object from a JSON file.

    Args:
        filepath: Path to the saved graph file

    Returns:
        PyTorch Geometric Data object
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Graph file not found: {filepath}")

    with open(filepath, 'r') as f:
        graph_dict = json.load(f)

    return dict_to_graph(graph_dict)


def save_graphs(graphs: List[Data], save_dir: Union[str, Path], prefix: str = "graph") -> None:
    """
    Save a list of graphs to a directory, one file per graph.
    Each graph is saved as {prefix}_t{timestep}.json or {prefix}_{index}.json

    Args:
        graphs: List of PyTorch Geometric Data objects
        save_dir: Directory where to save the graphs
        prefix: Prefix for filenames (default: "graph")
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving {len(graphs)} graphs to {save_dir}...")

    for idx, graph in enumerate(graphs):
        # Try to use timestep if available, otherwise use index
        if hasattr(graph, 'time_step') and graph.time_step is not None:
            filename = f"{prefix}_t{graph.time_step}.json"
        else:
            filename = f"{prefix}_{idx:04d}.json"

        filepath = save_dir / filename
        save_graph(graph, filepath)

    print(f"Successfully saved {len(graphs)} graphs to {save_dir}")


def load_graphs(load_dir: Union[str, Path], prefix: str = "graph") -> List[Data]:
    """
    Load all graphs from a directory that match the given prefix.
    Graphs are sorted by timestep (if available) or by filename.

    Args:
        load_dir: Directory containing the saved graphs
        prefix: Prefix for filenames to load (default: "graph")

    Returns:
        List of PyTorch Geometric Data objects sorted by timestep or filename
    """
    load_dir = Path(load_dir)

    if not load_dir.exists():
        raise FileNotFoundError(f"Directory not found: {load_dir}")

    # Find all matching files
    pattern = f"{prefix}_*.json"
    graph_files = sorted(load_dir.glob(pattern))

    if not graph_files:
        raise FileNotFoundError(f"No graph files found matching pattern {pattern} in {load_dir}")

    print(f"Loading {len(graph_files)} graphs from {load_dir}...")

    # Load all graphs
    graphs = []
    for filepath in graph_files:
        graph = load_graph(filepath)
        graphs.append(graph)

    # Sort by timestep if available
    if graphs and hasattr(graphs[0], 'time_step') and graphs[0].time_step is not None:
        graphs.sort(key=lambda g: g.time_step)

    print(f"Successfully loaded {len(graphs)} graphs")

    return graphs


def cache_graphs(
    graphs: List[Data],
    cache_dir: Union[str, Path],
    cache_name: str = "graphs_cache",
    prefix: str = "graph"
) -> None:
    """
    Cache a list of graphs to a specific directory with a given name.
    Creates a subdirectory within cache_dir with the cache_name.

    Args:
        graphs: List of PyTorch Geometric Data objects
        cache_dir: Base directory for caching
        cache_name: Name of the cache (creates subdirectory with this name)
        prefix: Prefix for individual graph filenames (default: "graph")

    Example:
        cache_graphs(graphs, "data/cached", "train_graphs_k2_h3")
        # Creates: data/cached/train_graphs_k2_h3/graph_t1.json, etc.
    """
    cache_path = Path(cache_dir) / cache_name
    save_graphs(graphs, cache_path, prefix=prefix)


def load_cached_graphs(
    cache_dir: Union[str, Path],
    cache_name: str = "graphs_cache",
    prefix: str = "graph"
) -> List[Data]:
    """
    Load cached graphs from a specific directory and cache name.

    Args:
        cache_dir: Base directory for caching
        cache_name: Name of the cache (subdirectory name)
        prefix: Prefix for individual graph filenames (default: "graph")

    Returns:
        List of PyTorch Geometric Data objects

    Example:
        graphs = load_cached_graphs("data/cached", "train_graphs_k2_h3")
    """
    cache_path = Path(cache_dir) / cache_name
    return load_graphs(cache_path, prefix=prefix)