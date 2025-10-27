import numpy as np
from typing import List
import torch
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal
from torch_geometric.data import Data

from torch_geometric.data import TemporalData


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

    Args:
        snapshots: List[Data]
        A list of PyG data objects representing the different timesteps in graphs
        dense: boolean
        A flag that either builds a dense event stream where every edge is turned into an event stream
        or if false, only turns new or changed edges into an event for the event stream.
    Returns:
        data: TemporalData
        A TemporalData object that represents the event stream.
    """
    #initialization of event lists
    src_list, dst_list, t_list, msg_list, label_list = [], [], [], [], []
    

    #Need to check if we have previous mappings, since we'd like to keep track of updates and deletions of nodes and events
    #This we believe is good for the predicting of emergence
    prev_map = None
    prev_edge_attr = None

    #A list that keeps track of nodes being deleted or not.
    evt_type_list = []

    has_x = hasattr(snapshots[0], 'x') and snapshots[0].x is not None
    has_edge_attr = hasattr(snapshots[0], 'edge_attr') and getattr(snapshots[0], 'edge_attr') is not None
    
    for t, d in enumerate(snapshots):

        feature_dim = d.x.size(1)
        msg_dim = feature_dim*2

        ei = d.edge_index
        mapping = edge_map(ei)

        #After each loop, we have the event stream per graph:
        #   src_list: the source node of an event
        #   dst_list: the destination of the event
        #   t_list: at what timestep the event has taken place
        #   evt_type_list: the type of event (event appears: +1, event dissapears: -1, event changes: 0)
        if not dense and prev_map is not None:
            added_edges = mapping.keys() - prev_map.keys()
            deleted_edges = prev_map.keys() - mapping.keys()
            changed_edges = set()
            if has_edge_attr:
                curr_attr = getattr(d, 'edge_attr')
                for e in (mapping & prev_map.keys()):
                    i_cur = mapping[e]
                    i_prev = prev_map[e]
                    same = torch.equal(curr_attr[i_cur], prev_edge_attr[i_prev])
                    if not same:
                        changed_edges.add(e)
                idx = [mapping[e] for e in changed_edges]
        
            
        else:
            added_edges = mapping.keys()
            deleted_edges = set()
            changed_edges = set()

        
        #get the indexes for each added edge node
        idx  = [mapping[e] for e in added_edges]
        #change to tensor so it plays nice with edge_index
        idx_t = torch.tensor(idx, dtype=torch.long)
        #add node from and to to added list
        src_list.append(ei[0, idx_t])
        dst_list.append(ei[1, idx_t])
        #add current timestep for each event
        t_list.append(torch.full((len(idx),), t, dtype=torch.long))
        #add event type to list
        evt_type_list.append(torch.full((len(idx),), 1, dtype=torch.int8))


        label_values = (d.y[ei[1, idx_t]]).float()
        label_list.append(label_values.view(-1, 1))

        src_feat = d.x[ei[0, idx_t]]
        dst_feat = d.x[ei[1, idx_t]]

        msg_list.append(torch.cat(([src_feat, dst_feat]), dim=-1))



        
        if deleted_edges:
            # for deletions, we take endpoints from PREVIOUS snapshot
            # (endpoints are the same; we only need (u,v) and time t)
            u = torch.tensor([uv[0] for uv in deleted_edges], dtype=torch.long)
            v = torch.tensor([uv[1] for uv in deleted_edges], dtype=torch.long)
            src_list.append(u)
            dst_list.append(v)
            t_list.append(torch.full((u.numel(),), t, dtype=torch.long))
            evt_type_list.append(torch.full((u.numel(),), -1, dtype=torch.int8))
            
            deleted_msg = torch.zeros((len(deleted_edges), msg_dim), dtype=d.x.dtype)
            label_list.append(torch.zeros((len(deleted_edges), 1), dtype=torch.float32))
            msg_list.append(deleted_msg)

            

        if changed_edges:
            idx = [mapping[e] for e in changed_edges]
            idx_t = torch.tensor(idx, dtype=torch.long)
            src_list.append(ei[0, idx_t])
            dst_list.append(ei[1, idx_t])

            src_feat = d.x[ei[0, idx_t]]
            dst_feat = d.x[ei[1, idx_t]]
            msg_list.append(torch.cat([src_feat, dst_feat], dim=-1))

            t_list.append(torch.full((len(idx),), t, dtype=torch.long))
            
            label_values = d.y[ei[1, idx_t]].float()
            label_list.append(label_values.view(-1, 1))
            evt_type_list.append(torch.zeros(len(idx), dtype=torch.int8))


        prev_map = mapping
        prev_edge_attr = getattr(d, 'edge_attr') if has_edge_attr else None


    if src_list:
        src = torch.cat(src_list)
        dst = torch.cat(dst_list)
        tt  = torch.cat(t_list)
        msg = torch.cat(msg_list)
        evt = torch.cat(evt_type_list)
        labels = torch.cat(label_list).view(-1)

        data = TemporalData(src=src, dst=dst, t=tt, msg=msg)
        data.y = labels
        data.event_type = evt
    
    return data

def edge_map(edge_index: torch.Tensor) -> dict:
    """
    Creates an edge mapping from all nodes in a specific timestep, and stores them in a dict.
    Args:
        edge_index: Tensor
        The edge indexes of the graph the mapping is being performed for
    Returns:
        dict:
        A dictionary with all current edge mappings
    """
    #convert from and to to list
    u = edge_index[0].cpu().tolist()
    v = edge_index[1].cpu().tolist()


    return {(uu, vv): i for i, (uu,vv) in enumerate(zip(u,v))}