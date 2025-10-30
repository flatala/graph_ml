"""
Graph caching utilities to save and load pre-built temporal graphs.
This avoids the need to rebuild graphs every time you run experiments.
"""

import torch
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Any
from torch_geometric.data import Data


def get_graph_cache_key(
    first_time_step: int,
    last_time_step: int,
    max_walk_length: int,
    time_horizon: int,
    use_distance_labels: bool,
    keep_class_labels_as_features: bool,
    ignore_illict: bool,
    ignore_previously_transacting_with_illicit: bool,
    cumulative: bool = False  # Default to False for backward compatibility
) -> str:
    """
    Generate a unique cache key based on graph building parameters.
    
    Returns:
        str: A hash string that uniquely identifies this configuration
    """
    config = {
        'first_time_step': first_time_step,
        'last_time_step': last_time_step,
        'max_walk_length': max_walk_length,
        'time_horizon': time_horizon,
        'use_distance_labels': use_distance_labels,
        'keep_class_labels_as_features': keep_class_labels_as_features,
        'ignore_illict': ignore_illict,
        'ignore_previously_transacting_with_illicit': ignore_previously_transacting_with_illicit,
        'cumulative': cumulative
    }
    
    # Create a stable hash from the configuration
    config_str = json.dumps(config, sort_keys=True)
    cache_key = hashlib.md5(config_str.encode()).hexdigest()
    
    return cache_key


def save_graphs(
    graphs: List[Data],
    cache_dir: str = "./graph_cache",
    first_time_step: int = None,
    last_time_step: int = None,
    max_walk_length: int = None,
    time_horizon: int = None,
    use_distance_labels: bool = None,
    keep_class_labels_as_features: bool = None,
    ignore_illict: bool = None,
    ignore_previously_transacting_with_illicit: bool = None,
    cumulative: bool = False  # Default to False for backward compatibility
) -> str:
    """
    Save pre-built graphs to disk for future use.
    
    Args:
        graphs: List of PyTorch Geometric Data objects
        cache_dir: Directory to store cached graphs
        **kwargs: Graph building parameters for generating cache key
    
    Returns:
        str: Path to the saved cache file
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    # Generate cache key from parameters
    cache_key = get_graph_cache_key(
        first_time_step=first_time_step,
        last_time_step=last_time_step,
        max_walk_length=max_walk_length,
        time_horizon=time_horizon,
        use_distance_labels=use_distance_labels,
        keep_class_labels_as_features=keep_class_labels_as_features,
        ignore_illict=ignore_illict,
        ignore_previously_transacting_with_illicit=ignore_previously_transacting_with_illicit,
        cumulative=cumulative
    )
    
    # Create descriptive filename
    filename = (f"graphs_t{first_time_step}-{last_time_step}_"
                f"walk{max_walk_length}_horizon{time_horizon}_"
                f"{cache_key[:8]}.pt")
    
    cache_file = cache_path / filename
    
    # Save metadata along with graphs
    cache_data = {
        'graphs': graphs,
        'metadata': {
            'first_time_step': first_time_step,
            'last_time_step': last_time_step,
            'max_walk_length': max_walk_length,
            'time_horizon': time_horizon,
            'use_distance_labels': use_distance_labels,
            'keep_class_labels_as_features': keep_class_labels_as_features,
            'ignore_illict': ignore_illict,
            'ignore_previously_transacting_with_illicit': ignore_previously_transacting_with_illicit,
            'cumulative': cumulative,
            'num_graphs': len(graphs),
            'cache_key': cache_key
        }
    }
    
    # Save using torch.save
    torch.save(cache_data, cache_file)
    
    print(f"✓ Saved {len(graphs)} graphs to: {cache_file}")
    print(f"  Cache key: {cache_key}")
    
    return str(cache_file)


def load_graphs(
    cache_dir: str = "./graph_cache",
    first_time_step: int = None,
    last_time_step: int = None,
    max_walk_length: int = None,
    time_horizon: int = None,
    use_distance_labels: bool = None,
    keep_class_labels_as_features: bool = None,
    ignore_illict: bool = None,
    ignore_previously_transacting_with_illicit: bool = None,
    cumulative: bool = False  # Default to False for backward compatibility
) -> tuple:
    """
    Load pre-built graphs from cache if they exist.
    
    Args:
        cache_dir: Directory where cached graphs are stored
        **kwargs: Graph building parameters for generating cache key
    
    Returns:
        tuple: (graphs, metadata) if cache exists, (None, None) otherwise
    """
    cache_path = Path(cache_dir)
    
    if not cache_path.exists():
        return None, None
    
    # Generate cache key from parameters (with cumulative)
    cache_key_new = get_graph_cache_key(
        first_time_step=first_time_step,
        last_time_step=last_time_step,
        max_walk_length=max_walk_length,
        time_horizon=time_horizon,
        use_distance_labels=use_distance_labels,
        keep_class_labels_as_features=keep_class_labels_as_features,
        ignore_illict=ignore_illict,
        ignore_previously_transacting_with_illicit=ignore_previously_transacting_with_illicit,
        cumulative=cumulative
    )
    
    # Generate OLD cache key (without cumulative parameter) for backward compatibility
    config_old = {
        'first_time_step': first_time_step,
        'last_time_step': last_time_step,
        'max_walk_length': max_walk_length,
        'time_horizon': time_horizon,
        'use_distance_labels': use_distance_labels,
        'keep_class_labels_as_features': keep_class_labels_as_features,
        'ignore_illict': ignore_illict,
        'ignore_previously_transacting_with_illicit': ignore_previously_transacting_with_illicit,
        # NOTE: 'cumulative' is NOT included here for backward compatibility
    }
    config_str_old = json.dumps(config_old, sort_keys=True)
    cache_key_old = hashlib.md5(config_str_old.encode()).hexdigest()
    
    # Try both cache keys (new format first, then old format for backward compatibility)
    cache_keys_to_try = [
        (cache_key_new, "new format (with cumulative parameter)"),
        (cache_key_old, "old format (without cumulative parameter)")
    ]
    
    for cache_key, key_description in cache_keys_to_try:
        # Look for matching cache file
        for cache_file in cache_path.glob("*.pt"):
            if cache_key[:8] in cache_file.name:
                try:
                    print(f"Found cache file: {cache_file} ({key_description})")
                    cache_data = torch.load(cache_file, weights_only=False)
                    
                    # Verify metadata matches (for new format)
                    metadata = cache_data['metadata']
                    
                    # For old cache files, the metadata won't have 'cache_key' matching new format
                    # But we can still load them if the parameters match
                    if 'cache_key' in metadata and metadata['cache_key'] == cache_key:
                        # Exact match (new format)
                        graphs = cache_data['graphs']
                        print(f"✓ Loaded {len(graphs)} graphs from cache")
                        print(f"  Cache key: {cache_key}")
                        return graphs, metadata
                    elif cache_key == cache_key_old:
                        # Old format - verify parameters match even if cache_key field doesn't exist
                        params_match = (
                            metadata.get('first_time_step') == first_time_step and
                            metadata.get('last_time_step') == last_time_step and
                            metadata.get('max_walk_length') == max_walk_length and
                            metadata.get('time_horizon') == time_horizon and
                            metadata.get('use_distance_labels') == use_distance_labels and
                            metadata.get('keep_class_labels_as_features') == keep_class_labels_as_features and
                            metadata.get('ignore_illict') == ignore_illict and
                            metadata.get('ignore_previously_transacting_with_illicit') == ignore_previously_transacting_with_illicit
                        )
                        
                        if params_match:
                            graphs = cache_data['graphs']
                            print(f"✓ Loaded {len(graphs)} graphs from OLD cache (backward compatible)")
                            print(f"  Cache key (old): {cache_key}")
                            # Add cumulative parameter to metadata for consistency
                            if 'cumulative' not in metadata:
                                metadata['cumulative'] = False  # Old caches assumed non-cumulative
                            return graphs, metadata
                        
                except Exception as e:
                    print(f"⚠ Warning: Could not load cache file {cache_file}: {e}")
                    continue
    
    return None, None


def list_cached_graphs(cache_dir: str = "./graph_cache") -> List[Dict[str, Any]]:
    """
    List all cached graph files with their metadata.
    
    Args:
        cache_dir: Directory where cached graphs are stored
    
    Returns:
        List of dictionaries containing cache file information
    """
    cache_path = Path(cache_dir)
    
    if not cache_path.exists():
        print(f"Cache directory does not exist: {cache_dir}")
        return []
    
    cached_files = []
    
    for cache_file in cache_path.glob("*.pt"):
        try:
            cache_data = torch.load(cache_file, weights_only=False)
            metadata = cache_data['metadata']
            
            file_info = {
                'filename': cache_file.name,
                'path': str(cache_file),
                'size_mb': cache_file.stat().st_size / (1024 * 1024),
                **metadata
            }
            cached_files.append(file_info)
            
        except Exception as e:
            print(f"⚠ Warning: Could not read {cache_file}: {e}")
            continue
    
    return cached_files


def delete_cached_graphs(
    cache_dir: str = "./graph_cache",
    first_time_step: int = None,
    last_time_step: int = None,
    max_walk_length: int = None,
    time_horizon: int = None,
    use_distance_labels: bool = None,
    keep_class_labels_as_features: bool = None,
    ignore_illict: bool = None,
    ignore_previously_transacting_with_illicit: bool = None,
    cumulative: bool = False  # Default to False for backward compatibility
) -> bool:
    """
    Delete cached graphs matching the specified parameters.
    
    Returns:
        bool: True if cache was deleted, False if not found
    """
    cache_path = Path(cache_dir)
    
    if not cache_path.exists():
        return False
    
    # Generate cache key from parameters
    cache_key = get_graph_cache_key(
        first_time_step=first_time_step,
        last_time_step=last_time_step,
        max_walk_length=max_walk_length,
        time_horizon=time_horizon,
        use_distance_labels=use_distance_labels,
        keep_class_labels_as_features=keep_class_labels_as_features,
        ignore_illict=ignore_illict,
        ignore_previously_transacting_with_illicit=ignore_previously_transacting_with_illicit,
        cumulative=cumulative
    )
    
    # Look for matching cache file
    for cache_file in cache_path.glob("*.pt"):
        if cache_key[:8] in cache_file.name:
            try:
                cache_file.unlink()
                print(f"✓ Deleted cache file: {cache_file}")
                return True
            except Exception as e:
                print(f"⚠ Warning: Could not delete {cache_file}: {e}")
                return False
    
    print("No cache file found matching the parameters")
    return False
