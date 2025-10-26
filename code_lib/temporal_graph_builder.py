import numpy as np
from typing import List
import torch
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
    edge_weights = []

    for d in snapshots:
        W = getattr(d, "edge_weights", None)
        if W is None:
        # Some datasets use edge_attr instead. If that exists, reduce to 1D if needed.
            W = getattr(d, "edge_attr", None)
        if W is not None and W.dim() > 1:  # e.g., shape [E, F]
            # If you truly want scalar weights, pick a channel or reduce:
            W = W.mean(dim=1)  # or w[:, 0]
        if W is None:
        # No weights provided: default to ones on the same device
            num_edges = d.edge_index.size(1)
            W = torch.ones(num_edges, dtype=torch.float32, device=d.edge_index.device)

        edge_weights.append(W.detach().cpu().numpy())


    # edge_weights = [ (d.edge_weight.cpu().numpy() if hasattr(d, "edge_weight")
    #                     else np.ones(d.edge_index.size(1), dtype=np.float32)) for d in snapshots]
    

    features = [d.x.cpu().numpy() for d in snapshots]
    targets  = [d.y.cpu().numpy() for d in snapshots]
    return DynamicGraphTemporalSignal(edge_indices, edge_weights, features, targets)

def convert_to_event_stream():

    
    return