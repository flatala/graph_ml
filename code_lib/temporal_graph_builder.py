import numpy as np
from typing import List
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal
from torch_geometric.data import Data


def convert_to_temporal_snapshots(snapshots: List[Data]) -> DynamicGraphTemporalSignal:
    """
    Converts a given list of PyG data object to a temporal version for use in PyG Temporal library
    We do this for ease of use with the snapshot based models geometric temporal provides us with

    Args:
        snapshots: List[Data]
        A list of PyG data objects representing the different timesteps in graphs
    Returns:
        data_t: DynamicGraphTemporalSignal
        A PyG Temporal object that iterates between the Data objects given to it when training.
        

    """
    edge_indices = [d.edge_index.cpu().numpy() for d in snapshots]
    edge_weights = [ (d.edge_weight.cpu().numpy() if hasattr(d, "edge_weight")
                      else np.ones(d.edge_index.size(1), dtype=np.float32)) for d in snapshots]
    features = [d.x.cpu().numpy() for d in snapshots]
    targets  = [d.y.cpu().numpy() for d in snapshots]
    return DynamicGraphTemporalSignal(edge_indices, edge_weights, features, targets)

def convert_to_event_stream():
    return