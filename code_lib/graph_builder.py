import scipy.sparse as sp
import pandas as pd
import numpy as np
import torch
import time

from torch_geometric.data import Data
from scipy.sparse import csr_matrix
from collections import Counter
from tqdm import tqdm


def extract_node_features(nodes_up_to_t, active_addresses, address_to_local_id, keep_class_labels_as_features: bool = True, add_staleness_feature: bool = False, current_time_step: int = None):
    """
    Extract latest features for each active node (OPTIMIZED VERSION).
    When a node appears multiple times, use the most recent feature values.

    Args:
        nodes_up_to_t: DataFrame with node data up to current time step
        active_addresses: array/list of addresses active (have emerged) up to current time step
        address_to_local_id: dict mapping address -> local node ID
        keep_class_labels_as_features: bool, whether to include class labels as features
        add_staleness_feature: bool, whether to add staleness feature (current_t - first_appearance_t)
        current_time_step: int, current time step (required if add_staleness_feature=True)

    Returns:
        torch.Tensor: node features [num_nodes, num_features]
    """

    # include labels at t as features (exploiting the knowledge, may be bad for "unobvious" emergence)
    if keep_class_labels_as_features:
        feature_cols = [col for col in nodes_up_to_t.columns
                    if col not in ['address', 'Time step']]

    # don't include labels at t as features (less bias towards emergence nera illict nodes)
    else:
        feature_cols = [col for col in nodes_up_to_t.columns
                    if col not in ['address', 'Time step', 'class']]

    # we sort by time to only include latest node features
    nodes_sorted = nodes_up_to_t.sort_values('Time step', ascending=True)

    # use groupby to get latest row per address
    latest_per_address = nodes_sorted.groupby('address', as_index=True)[feature_cols].last()

    # prepare an empty array (potentially with staleness feature)
    num_base_features = len(feature_cols)
    num_features = num_base_features + (1 if add_staleness_feature else 0)
    num_active_nodes = len(active_addresses)
    node_features = np.zeros((num_active_nodes, num_features))

    # if staleness feature is requested, compute first appearance time for each address
    if add_staleness_feature:
        if current_time_step is None:
            raise ValueError("current_time_step must be provided when add_staleness_feature=True")
        first_appearance = nodes_sorted.groupby('address')['Time step'].min()

    # populate features
    for addr in active_addresses:
        if addr in latest_per_address.index:
            local_id = address_to_local_id[addr]
            # add base features
            node_features[local_id, :num_base_features] = latest_per_address.loc[addr].values

            # add staleness feature if requested
            if add_staleness_feature:
                staleness = current_time_step - first_appearance.loc[addr]
                node_features[local_id, num_base_features] = staleness

    return torch.tensor(node_features, dtype=torch.float)


def extract_node_classes(active_addresses, address_to_local_id, node_labels_df):
    """
    Extract node class labels (1=illicit, 2=licit, 3=unknown).
    
    Args:
        active_addresses: array/list of addresses active at current time step
        address_to_local_id: dict mapping address -> local node ID
        node_labels_df: DataFrame with columns ['address', 'class']
    
    Returns:
        torch.Tensor: node classes [num_nodes] with values 1 (illicit), 2 (licit), or 3 (unknown)
    """
    num_nodes = len(active_addresses)
    node_classes = torch.full((num_nodes,), 3, dtype=torch.long)  # default: unknown
    
    # create address -> class mapping
    address_to_class = dict(zip(node_labels_df['address'], node_labels_df['class']))
    
    # assign classes
    for addr in active_addresses:
        if addr in address_to_class and addr in address_to_local_id:
            local_id = address_to_local_id[addr]
            node_classes[local_id] = address_to_class[addr]
    
    return node_classes


def build_edge_index(edges_up_to_t, address_id_map_until_t):
    """
    Build edge index including ALL edges
    
    Args:
        edges_up_to_t: DataFrame with edge data up to current time step
        address_id_map_until_t: dict mapping address that have emerged until time t (t included) -> local node ID
    
    Returns:
        torch.Tensor: edge_index [2, num_edges]
    """

    # filter edges where both endpoints exist (maybe unnecessary)
    valid_src = edges_up_to_t['input_address'].isin(address_id_map_until_t.keys())
    valid_dst = edges_up_to_t['output_address'].isin(address_id_map_until_t.keys())
    edges_valid = edges_up_to_t[valid_src & valid_dst]
    
    if len(edges_valid) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    
    # map addresses to local IDs
    src_ids = edges_valid['input_address'].map(address_id_map_until_t).values
    dst_ids = edges_valid['output_address'].map(address_id_map_until_t).values
    
    # stack into edge_index
    edge_index = np.stack([src_ids, dst_ids], axis=0)
    
    return torch.tensor(edge_index, dtype=torch.long)


def build_undirected_adjacency_matrix(edge_index: torch.Tensor, num_nodes: int):
    """
    Build symmetric sparse adjacency matrix from edge_index.
    
    Args:
        edge_index: torch.Tensor [2, num_edges], directed edge list
        num_nodes: int, total number of nodes in the graph 
    
    Returns:
        scipy.sparse.csr_matrix: symmetric adjacency matrix [num_nodes, num_nodes]
    """
    if edge_index.shape[1] == 0:
        return csr_matrix((num_nodes, num_nodes))
    
    # convert to numpy
    edge_index_np = edge_index.cpu().numpy()
    
    # create directed edges
    src = edge_index_np[0]
    dst = edge_index_np[1]
    
    # add reverse edges for undirected graph 
    all_src = np.concatenate([src, dst])
    all_dst = np.concatenate([dst, src])
    
    # create sparse matrix (values are 1s for adjacency)
    data = np.ones(len(all_src), dtype=np.float32)
    adjacency_matrix = csr_matrix((data, (all_src, all_dst)), shape=(num_nodes, num_nodes))
    
    return adjacency_matrix


def compute_reachability_matrix(adjacency_matrix, max_walk_length):
    """
    Compute k-hop reachability matrix using sparse matrix powers.
    
    Args:
        adjacency_matrix: scipy.sparse.csr_matrix, adjacency matrix
        walk_length: int, maximum walk length to reach the neighbours 
    
    Returns:
        scipy.sparse.csr_matrix: boolean matrix where [i,j]=1 if j is reachable from i in ≤k hops
    """
    num_nodes = adjacency_matrix.shape[0]
    
    # start with identity (0-hop: each node reaches itself)
    reachability = sp.eye(num_nodes, format='csr')
    
    # current power of adjacency matrix
    current_power = adjacency_matrix.copy()
    
    # add A + A^2 + ... + A^k
    # this will give us positive (i,j) values whenever we can reach node j from i in a k-hop walk
    for hop in range(1, max_walk_length + 1):
        if hop > 1:
            current_power = current_power @ adjacency_matrix
        reachability = reachability + current_power
    
    # convert to boolean matrix - positive value means coulkd be reached
    reachability = (reachability > 0).astype(np.float32)
    
    return reachability


def compute_reachability_to_targets(adjacency_matrix, target_nodes, max_walk_length):
    """
    Compute reachability from ALL nodes to a SUBSET of target nodes (much faster).
    
    This is significantly faster than computing the full reachability matrix when
    the number of target nodes is small compared to the total number of nodes.
    
    Uses reverse BFS from target nodes to find reachability efficiently.
    
    Args:
        adjacency_matrix: scipy.sparse.csr_matrix, adjacency matrix
        target_nodes: array-like, indices of target nodes
        max_walk_length: int, maximum walk length
    
    Returns:
        scipy.sparse.csr_matrix: boolean matrix [num_nodes, num_targets] where
        [i,j] = 1 if target j is reachable from node i within max_walk_length hops
    """
    num_nodes = adjacency_matrix.shape[0]
    num_targets = len(target_nodes)
    
    if num_targets == 0:
        return sp.csr_matrix((num_nodes, 0))
    
    # Convert target_nodes to numpy array for indexing
    target_nodes = np.asarray(target_nodes)
    
    # Transpose adjacency for reverse search (who can reach the targets)
    adj_T = adjacency_matrix.T.tocsr()
    
    # Build list of column data (more efficient than incremental sparse matrix updates)
    reachability_columns = []
    
    # For each target node, do a BFS to find who can reach it
    for target_idx, target_node in enumerate(target_nodes):
        # Track visited nodes
        visited = np.zeros(num_nodes, dtype=bool)
        
        # Start from the target node (using lil_matrix for efficient single element setting)
        current_frontier = sp.lil_matrix((num_nodes, 1))
        current_frontier[target_node, 0] = 1
        current_frontier = current_frontier.tocsr()
        visited[target_node] = True
        
        # BFS for up to max_walk_length hops
        for hop in range(1, max_walk_length + 1):
            # Find nodes that can reach current frontier in one hop
            # (reverse: who has edges TO current frontier)
            next_frontier = adj_T @ current_frontier
            
            # Only keep unvisited nodes
            next_frontier_array = next_frontier.toarray().flatten()
            newly_reached = (next_frontier_array > 0) & ~visited
            
            if not newly_reached.any():
                break
            
            # Mark as visited
            visited[newly_reached] = True
            
            # Update frontier for next iteration
            current_frontier = sp.csr_matrix(newly_reached.reshape(-1, 1).astype(float))
        
        # Store this column as a sparse vector (who can reach this target)
        reachability_columns.append(sp.csr_matrix(visited.reshape(-1, 1).astype(float)))
    
    # Horizontally stack all columns to create the final matrix
    if reachability_columns:
        reachability = sp.hstack(reachability_columns, format='csr')
    else:
        reachability = sp.csr_matrix((num_nodes, 0))
    
    return reachability


def compute_distance_matrix(adjacency_matrix, max_walk_length):
    """
    Compute a distance matrix from any node in the graph to any other node.
    
    Args:
        adjacency_matrix: scipy.sparse.csr_matrix, adjacency matrix
        k_hops: int, neighborhood radius
    
    Returns:
        scipy.sparse.csr_matrix: boolean matrix where [i,j]=distance if 
        j is reachable from i in ≤k hops. On the diagonal it returns zeros,
        as distance from i to i is zero, and off-diagonal zeros mean unreachable.
    """
    num_nodes = adjacency_matrix.shape[0]
    
    # start with identity (0-hop: each node reaches itself)
    visited = sp.eye(num_nodes, format='csr')
    distances = sp.csr_matrix((num_nodes, num_nodes))
    
    # current power of adjacency matrix
    current_power = adjacency_matrix.copy()
    
    # add A + A^2 + ... + A^k
    # this will give us edges whenever we can reach the other matrix in a k-hop walk
    for hop in range(1, max_walk_length + 1):
        if hop > 1:
            current_power = current_power @ adjacency_matrix

        # only keep the values for non-visited nodes
        reached_for_first_time = current_power - current_power.multiply(visited)

        # set distance matrix values to hop
        mask = reached_for_first_time.sign() * hop

        # update visited matrix
        visited = (visited + current_power).minimum(1)

        # update distances
        distances = distances + mask
    
    return distances


def compute_distances_to_targets(adjacency_matrix, target_nodes, max_walk_length):
    """
    Compute distances from ALL nodes to a SUBSET of target nodes (much faster).
    
    This is significantly faster than computing the full distance matrix when
    the number of target nodes is small compared to the total number of nodes.
    
    Uses reverse BFS from target nodes to find distances efficiently.
    
    Args:
        adjacency_matrix: scipy.sparse.csr_matrix, adjacency matrix
        target_nodes: array-like, indices of target nodes
        max_walk_length: int, maximum walk length
    
    Returns:
        scipy.sparse.csr_matrix: distance matrix [num_nodes, num_targets] where
        [i,j] = distance from node i to target j (0 means unreachable)
    """
    num_nodes = adjacency_matrix.shape[0]
    num_targets = len(target_nodes)
    
    if num_targets == 0:
        return sp.csr_matrix((num_nodes, 0))
    
    # Convert target_nodes to numpy array for indexing
    target_nodes = np.asarray(target_nodes)
    
    # Transpose adjacency for reverse search (who can reach the targets)
    adj_T = adjacency_matrix.T.tocsr()
    
    # Build list of column data (more efficient than incremental sparse matrix updates)
    distance_columns = []
    
    # For each target node, do a forward BFS to find who can reach it
    for target_idx, target_node in enumerate(target_nodes):
        # Track visited nodes and their distances
        visited = np.zeros(num_nodes, dtype=bool)
        current_dist = np.zeros(num_nodes, dtype=np.int32)
        
        # Start from the target node
        # Use lil_matrix for efficient single element setting
        current_frontier = sp.lil_matrix((num_nodes, 1))
        current_frontier[target_node, 0] = 1
        current_frontier = current_frontier.tocsr()
        visited[target_node] = True
        current_dist[target_node] = 0
        
        # BFS for up to max_walk_length hops
        for hop in range(1, max_walk_length + 1):
            # Find nodes that can reach current frontier in one hop
            # (reverse: who has edges TO current frontier)
            next_frontier = adj_T @ current_frontier
            
            # Only keep unvisited nodes
            next_frontier_array = next_frontier.toarray().flatten()
            newly_reached = (next_frontier_array > 0) & ~visited
            
            if not newly_reached.any():
                break
            
            # Mark as visited and record distance
            visited[newly_reached] = True
            current_dist[newly_reached] = hop
            
            # Update frontier for next iteration (convert boolean mask to sparse)
            current_frontier = sp.csr_matrix(newly_reached.reshape(-1, 1).astype(float))
        
        # Store this column as a sparse vector using COO format (efficient for construction)
        nonzero_indices = np.where(current_dist > 0)[0]
        if len(nonzero_indices) > 0:
            col_data = current_dist[nonzero_indices]
            col = sp.coo_matrix((col_data, (nonzero_indices, np.zeros(len(nonzero_indices), dtype=int))), 
                                shape=(num_nodes, 1))
            distance_columns.append(col.tocsr())
        else:
            distance_columns.append(sp.csr_matrix((num_nodes, 1)))
    
    # Horizontally stack all columns to create the final matrix
    if distance_columns:
        distances = sp.hstack(distance_columns, format='csr')
    else:
        distances = sp.csr_matrix((num_nodes, 0))
    
    return distances


def compute_distance_matrix_old(adjacency_matrix, max_walk_length):
    """
    Compute a distance matrix from any node in the graph to any other node.
    
    Args:
        adjacency_matrix: scipy.sparse.csr_matrix, adjacency matrix
        k_hops: int, neighborhood radius
    
    Returns:
        scipy.sparse.csr_matrix: boolean matrix where [i,j]=distance if 
        j is reachable from i in ≤k hops. On the diagonal it returns zeros,
        as distance from i to i is zero, and off-diagonal zeros mean unreachable.
    """
    num_nodes = adjacency_matrix.shape[0]
    
    # start with identity (0-hop: each node reaches itself)
    visited = sp.eye(num_nodes, format='csr')
    distances = sp.csr_matrix((num_nodes, num_nodes))
    
    # current power of adjacency matrix
    current_power = adjacency_matrix.copy()
    
    # add A + A^2 + ... + A^k
    # this will give us edges whenever we can reach the other matrix in a k-hop walk
    for hop in range(1, max_walk_length + 1):
        if hop > 1:
            current_power = current_power @ adjacency_matrix

        # only keep the values for non-visited nodes
        reached_for_first_time = current_power - current_power.multiply(visited)

        # set distance matrix values to hop
        mask = reached_for_first_time.sign() * hop

        # update visited matrix
        visited = (visited + current_power).minimum(1)

        # update distances
        distances = distances + mask
    
    return distances


def get_labels(
    current_time_step: int, 
    edges_df: pd.DataFrame,
    active_addresses: set,
    active_address_to_local_id,
    all_illicit_addresses: set,
    edge_index_at_t: torch.Tensor,
    edges_by_timestep: dict = None,
    use_distance_labels: bool = True,
    max_walk_length: int = 2,
    time_horizon: int = 3,
    ignore_illict: bool = True,
    ignore_previously_transacting_with_illicit: bool = True,
    profile: bool = False
):
    """
    Generate labels for illicit activity emergence prediction in a temporal transaction graph.
    
    This function predicts which nodes are at risk of exposure to NEW illicit activity by 
    computing either distance-based or binary labels. "NEW" illicit nodes are those that 
    don't exist at the current time step but emerge in the future time horizon.
    
    Args:
        current_time_step: int
            Current time step t for which to generate labels
        edges_df: pd.DataFrame
            DataFrame containing all transaction edges with columns:
            - 'Time step': temporal information
            - 'input_address': source node address
            - 'output_address': destination node address
        active_addresses: set
            Set of all addresses that have emerged (are active) up to and including time t
        active_address_to_local_id: dict
            Mapping from active addresses to local node IDs (0 to num_nodes-1)
        all_illicit_addresses: set
            Set of ALL illicit addresses in the entire graph (across all time steps)
        edge_index_at_t: torch.Tensor
            Edge index tensor [2, num_edges] representing the graph structure at time t
        edges_by_timestep: dict, optional
            Pre-grouped edges by time step for O(1) lookup (performance optimization)
        use_distance_labels: bool, default=True
            If True, labels are distances (0, 1, 2, ..., max_walk_length+1)
            If False, labels are binary (0 or 1)
        max_walk_length: int, default=2
            Maximum number of hops to consider in the neighborhood radius
        time_horizon: int, default=3
            Number of future time steps to look ahead (t+1, t+2, ..., t+time_horizon)
        ignore_illict: bool, default=True
            If True, nodes that are already illicit at time t receive default labels
            (max_walk_length+1 for distance, 0 for binary) and illicit nodes are excluded
            from the set of future illicit transactors
        ignore_previously_transacting_with_illicit: bool, default=True
            If True, nodes that have any transaction history with illicit nodes up to time t
            receive default labels and are excluded from the set of future illicit transactors
    
    Returns:
        torch.Tensor: labels [num_nodes] with dtype=torch.long
        
        If use_distance_labels=True:
            - 0: node will directly transact with NEW illicit nodes
            - 1, 2, ..., max_walk_length: distance to nearest neighbor that will transact with NEW illicit
            - max_walk_length + 1: no NEW illicit activity in k-hop neighborhood (default/no emergence)
        
        If use_distance_labels=False:
            - 0: no NEW illicit activity in k-hop neighborhood (default)
            - 1: at least one node in k-hop neighborhood will transact with NEW illicit nodes
    """

    label_timings = {}
    t_label_start = time.time()
    
    # get the number of nodes in the graph at time t
    num_nodes = len(active_addresses)

    # prepare reverse mapping for looking up addresses
    id_to_address = {idx: addr for addr, idx in active_address_to_local_id.items()}

    # build the adjacency matrix
    t0 = time.time()
    adjacency_matrix = build_undirected_adjacency_matrix(edge_index_at_t, num_nodes)
    label_timings['build_adjacency'] = time.time() - t0
    
    # Find illicit addresses that already exist at time t - we want
    # to exclude these, since we care about predicting emergence, not 
    # if a neighbour will have any edge with any illicit in the future
    t0 = time.time()
    existing_illicit_at_t = active_addresses & all_illicit_addresses
    
    # illicit nodes that don't exist at time t yet
    future_illicit_addresses = all_illicit_addresses - existing_illicit_at_t
    label_timings['identify_illicit'] = time.time() - t0
    
    # collect all nodes that will in future transact with 
    # illicit nodes that don't yet exist in the graph
    t0 = time.time()
    future_illicit_transactor_adresses = set()
    for future_t in range(current_time_step + 1, current_time_step + time_horizon + 1):
        # OPTIMIZATION: Use pre-grouped edges if available
        if edges_by_timestep is not None:
            edges_future = edges_by_timestep.get(future_t, pd.DataFrame())
        else:
            # Fallback to filtering (slower)
            edges_future = edges_df[edges_df['Time step'] == future_t]

        # continue if none are found in the timestep
        if edges_future.empty:
            continue
        
        # find transaction edges that go to new illicit adresses
        illicit_dst_mask = edges_future['output_address'].isin(future_illicit_addresses)
        src_to_illicit = set(edges_future.loc[illicit_dst_mask, 'input_address'].values)
        
        # find transaction edges that come from new illicit adresses
        illicit_src_mask = edges_future['input_address'].isin(future_illicit_addresses)
        dst_from_illicit = set(edges_future.loc[illicit_src_mask, 'output_address'].values)
        
        # collect all the adresses
        future_illicit_transactor_adresses.update(src_to_illicit | dst_from_illicit)
    label_timings['find_future_transactors'] = time.time() - t0

    # if it is specified so, ignore the transactors that are illict themselves
    # (it might be more interesting to look for illicit emergence where tehre is no illicit nodes)
    if ignore_illict:
        future_illicit_transactor_adresses = future_illicit_transactor_adresses - existing_illicit_at_t
    
    # if it is specified so, also ignore nodes that have
    # previously transacted with illicit nodes
    t0 = time.time()
    nodes_with_illicit_history = set()
    if ignore_previously_transacting_with_illicit:
        # OPTIMIZATION: Build illicit history from edge_index instead of re-filtering DataFrame
        if edge_index_at_t.shape[1] > 0:
            edge_index_np = edge_index_at_t.cpu().numpy()
            
            # Create reverse mapping from local ID to address
            id_to_address = {idx: addr for addr, idx in active_address_to_local_id.items()}
            
            # Get illicit node IDs at time t
            existing_illicit_ids = {active_address_to_local_id[addr] 
                                   for addr in existing_illicit_at_t 
                                   if addr in active_address_to_local_id}
            
            if existing_illicit_ids:
                # Find edges connected to illicit nodes
                src_ids = edge_index_np[0]
                dst_ids = edge_index_np[1]
                
                # Nodes that sent to illicit
                src_to_illicit_ids = src_ids[np.isin(dst_ids, list(existing_illicit_ids))]
                # Nodes that received from illicit
                dst_from_illicit_ids = dst_ids[np.isin(src_ids, list(existing_illicit_ids))]
                
                # Convert back to addresses
                nodes_with_illicit_history = {
                    id_to_address[node_id] 
                    for node_id in np.concatenate([src_to_illicit_ids, dst_from_illicit_ids])
                    if node_id in id_to_address
                }
        
        # remove these from future_illicit_transactor_adresses
        future_illicit_transactor_adresses = future_illicit_transactor_adresses - nodes_with_illicit_history
    label_timings['build_illicit_history'] = time.time() - t0

    # map adresses to ids for the nodes that are emerged at t
    # and will transact with future illict nodes in horizon
    t0 = time.time()
    future_illicit_transactor_ids = []
    for addr in future_illicit_transactor_adresses:
        if addr in active_address_to_local_id:
            future_illicit_transactor_ids.append(active_address_to_local_id[addr])

    # convert to numpy array
    future_illicit_transactor_ids = np.array(future_illicit_transactor_ids)

    # Create set for fast lookup
    future_illicit_transactor_id_set = set(future_illicit_transactor_ids)
    label_timings['map_transactor_ids'] = time.time() - t0
    
    # if we want labels to be the distance to the node's 
    # nearest neighbour (within the walk_length) that 
    # will make transactions with future illicit nodes
    if use_distance_labels:
        # Initialize all labels to -1 (unreachable within walk_length)
        labels = np.full(num_nodes, -1, dtype=int)

        # if no nodes emerged at t will transact with future illicit, return
        if len(future_illicit_transactor_adresses) == 0:
            return torch.tensor(labels, dtype=torch.long)

        # OPTIMIZATION: Compute distances only to target nodes (much faster!)
        t0 = time.time()
        distances_to_new_illicit = compute_distances_to_targets(
            adjacency_matrix, 
            future_illicit_transactor_ids, 
            max_walk_length
        )
        label_timings['compute_distances'] = time.time() - t0

        # Compute labels for each node
        t0 = time.time()
        for node_id in range(num_nodes):
            node_address = id_to_address[node_id]

            # if we dont want to label nodes with illicit transaction history
            if ignore_previously_transacting_with_illicit and node_address in nodes_with_illicit_history:
                continue

            # if it is specified to not label illicit nodes (that exist at time t)
            if ignore_illict and node_address in existing_illicit_at_t:
                continue

            # this node itself will transact with new illicit
            if node_id in future_illicit_transactor_id_set:
                labels[node_id] = 0
                continue
                
            # otherwise find minimum non-zero distance (keep sparse)
            row_distances = distances_to_new_illicit[node_id, :].toarray().flatten()

            # zero distances off-diagonal mean unreachable,
            # so we want to filter those out
            valid_distances = row_distances[row_distances > 0]

            # take the minimum distance to a neighbour
            # that will transact with a future illicit node,
            # if such a neighbour exists with max walk distance
            if len(valid_distances) > 0:
                labels[node_id] = int(valid_distances.min())
        label_timings['assign_distance_labels'] = time.time() - t0

        label_timings['total'] = time.time() - t_label_start
        
        if profile:
            print(f"    Label generation breakdown:")
            for key, val in sorted(label_timings.items(), key=lambda x: -x[1]):
                if key != 'total':
                    pct = 100 * val / label_timings['total']
                    print(f"      {key:30s}: {val:6.3f}s ({pct:5.1f}%)")

        return torch.tensor(labels, dtype=torch.long)

    # in the case it is specified to only use the binary labels
    else:
        labels = torch.zeros(num_nodes, dtype=torch.long)

        # if no nodes emerged at t will transact with future illicit, return all zeros
        if len(future_illicit_transactor_adresses) == 0:
            return labels

        # OPTIMIZATION: Compute reachability only to target nodes (much faster!)
        t0 = time.time()
        reachability_to_illicit = compute_reachability_to_targets(
            adjacency_matrix, 
            future_illicit_transactor_ids, 
            max_walk_length
        )
        label_timings['compute_reachability'] = time.time() - t0

        # check if a node can reach any of the target nodes (has illicit in neighborhood)
        has_new_illicit_in_neighborhood = (reachability_to_illicit.sum(axis=1) > 0).A1
        
        t0 = time.time()
        for node_id in range(num_nodes):
            # Get the address for this node
            node_address = id_to_address[node_id]

            # if we dont want to label nodes with illicit transaction history
            if ignore_previously_transacting_with_illicit and node_address in nodes_with_illicit_history:
                continue

            # if it is specified to not label illicit nodes (that exist at time t)
            if ignore_illict and node_address in existing_illicit_at_t:
                continue

            # this node itself will transact with new illicit
            if node_id in future_illicit_transactor_id_set:
                labels[node_id] = 1
                continue

            # neighbour within walk length will trasact
            if has_new_illicit_in_neighborhood[node_id]:
                labels[node_id] = 1
        label_timings['assign_binary_labels'] = time.time() - t0
        
        label_timings['total'] = time.time() - t_label_start
        
        if profile:
            print("    Label generation breakdown:")
            for key, val in sorted(label_timings.items(), key=lambda x: -x[1]):
                if key != 'total':
                    pct = 100 * val / label_timings['total']
                    print(f"      {key:30s}: {val:6.3f}s ({pct:5.1f}%)")
        
        return labels


def build_emergence_graph_at_timestep(
    current_time_step: int,
    nodes_with_classes_df: pd.DataFrame,
    edges_with_labels_df: pd.DataFrame,
    all_illicit_addresses: set = None,
    edges_by_timestep: dict = None,
    keep_class_labels_as_features: bool = False,
    add_staleness_feature: bool = False,
    use_distance_labels: bool = True,
    max_walk_length: int = 2,
    time_horizon: int = 3,
    ignore_illict: bool = True,
    ignore_previously_transacting_with_illicit: bool = True,
    profile: bool = False,
    cumulative: bool = False
) -> Data:
    """
    Build a temporal graph snapshot at a given time step for emergence prediction.

    This function creates a cumulative graph containing all nodes and edges that have
    appeared up to and including the current time step, along with emergence labels
    that predict future exposure to NEW illicit activity.

    Args:
        current_time_step: int
            Current time step t for which to build the graph
        nodes_df: pd.DataFrame
            DataFrame with node features and columns ['address', 'Time step', 'class', ...]
        edges_df: pd.DataFrame
            DataFrame with transaction edges and columns ['Time step', 'input_address', 'output_address']
        all_illicit_addresses: set, optional
            Pre-computed set of all illicit addresses (performance optimization)
        edges_by_timestep: dict, optional
            Pre-grouped edges by time step for O(1) lookup (performance optimization)
        keep_class_labels_as_features: bool, default=False
            If True, include node class labels as features (may introduce label leakage)
            If False, exclude class labels from features
        add_staleness_feature: bool, default=False
            If True, add a staleness feature (current_time_step - first_appearance_time) to node features
        use_distance_labels: bool, default=True
            If True, labels are distances (0, 1, 2, ..., max_walk_length+1)
            If False, labels are binary (0 or 1)
        max_walk_length: int, default=2
            Maximum number of hops to consider in the neighborhood
        time_horizon: int, default=3
            Number of future time steps to look ahead for emergence prediction
        ignore_illict: bool, default=True
            If True, nodes that are already illicit receive default labels
        ignore_previously_transacting_with_illicit: bool, default=True
            If True, nodes with illicit transaction history receive default labels
        profile: bool, default=False
            If True, print detailed timing information for each step
        cumulative: bool, default=False
            If True, builds a monotonically increasing grpah by iteratively adding the next timestep to the current graph.
            Otherwise, builds a seperate graph for each timestep
    
    Returns:
        Data: PyTorch Geometric Data object with attributes:
            - x: node features [num_nodes, num_features]
            - edge_index: graph structure [2, num_edges]
            - y: emergence labels [num_nodes] - PREDICTION TARGET
            - node_class: ground truth node classes [num_nodes] - for visualization/analysis
            - num_nodes: number of nodes in the graph
            - time_step: current time step (for tracking)
    """
    
    timings = {}
    t_start = time.time()
    
    # step 1: get all nodes and edges up to current time step
    if cumulative:
        t0 = time.time()
        nodes_up_to_t = nodes_with_classes_df[nodes_with_classes_df['Time step'] <= current_time_step]
        edges_up_to_t = edges_with_labels_df[edges_with_labels_df['Time step'] <= current_time_step]
        timings['filter_data'] = time.time() - t0
    else:
        t0 = time.time()
        nodes_up_to_t = nodes_with_classes_df[nodes_with_classes_df['Time step'] == current_time_step]
        edges_up_to_t = edges_with_labels_df[edges_with_labels_df['Time step'] == current_time_step]
        timings['filter_data'] = time.time() - t0
        
    
    # step 2: get active addresses (thise that have already emerged at t)
    t0 = time.time()
    active_addresses = nodes_up_to_t['address'].unique()
    timings['get_addresses'] = time.time() - t0
    
    # step 3: create local address-to-id mapping for this time step
    t0 = time.time()
    address_to_local_id = {addr: idx for idx, addr in enumerate(active_addresses)}
    timings['create_mapping'] = time.time() - t0
    
    # step 4: pre-compute set of all illicit addresses (for efficiency)
    t0 = time.time()
    if all_illicit_addresses is None:
        all_illicit_addresses = set(nodes_with_classes_df[nodes_with_classes_df['class'] == 1]['address'].values)
    timings['get_illicit'] = time.time() - t0
    
    # step 5: Extract node features
    t0 = time.time()
    node_features = extract_node_features(
        nodes_up_to_t,
        active_addresses,
        address_to_local_id,
        keep_class_labels_as_features=keep_class_labels_as_features,
        add_staleness_feature=add_staleness_feature,
        current_time_step=current_time_step
    )
    timings['extract_features'] = time.time() - t0
    
    # step 6: build edge index
    t0 = time.time()
    edge_index = build_edge_index(edges_up_to_t, address_to_local_id)
    timings['build_edge_index'] = time.time() - t0
    
    # step 7: generate emergence labels
    t0 = time.time()
    labels = get_labels(
        current_time_step=current_time_step,
        edges_df=edges_with_labels_df,
        active_addresses=set(active_addresses),
        active_address_to_local_id=address_to_local_id,
        all_illicit_addresses=all_illicit_addresses,
        edge_index_at_t=edge_index,
        edges_by_timestep=edges_by_timestep,
        use_distance_labels=use_distance_labels,
        max_walk_length=max_walk_length,
        time_horizon=time_horizon,
        ignore_illict=ignore_illict,
        ignore_previously_transacting_with_illicit=ignore_previously_transacting_with_illicit,
        profile=profile
    )
    timings['get_labels'] = time.time() - t0
    
    # step 8: extract node classes (this is modtly for visualization / analysis)
    t0 = time.time()
    node_classes = extract_node_classes(active_addresses, address_to_local_id, nodes_with_classes_df)
    timings['extract_classes'] = time.time() - t0
    
    # step 9: create PyTorch Geometric Data object
    t0 = time.time()
    data = Data(
        x=node_features,
        edge_index=edge_index,
        y=labels,
        node_class=node_classes,
        num_nodes=len(active_addresses),
        time_step=current_time_step
    )
    timings['create_data'] = time.time() - t0
    
    timings['total'] = time.time() - t_start
    
    if profile:
        print(f"\n  Timing breakdown for t={current_time_step}:")
        for key, val in sorted(timings.items(), key=lambda x: -x[1]):
            pct = 100 * val / timings['total']
            print(f"    {key:20s}: {val:6.3f}s ({pct:5.1f}%)")
    
    return data


def build_emergence_graphs_for_time_range(
    edges_with_labels_df: pd.DataFrame,
    nodes_with_classes_df: pd.DataFrame,
    first_time_step: int = 1,
    last_time_step: int = 49,
    max_walk_length: int = 2,
    time_horizon: int = 3,
    use_distance_labels: bool = False,
    keep_class_labels_as_features: bool = False,
    add_staleness_feature: bool = False,
    ignore_illict: bool = True,
    ignore_previously_transacting_with_illicit: bool = True,
    profile: bool = False,
    cumulative: bool = False
):
    """
    Build a sequence of emergence prediction graphs across multiple time steps.

    This function generates a series of temporal graph snapshots for a specified range of
    time steps, where each graph can be used to train or evaluate models for predicting
    the emergence of illicit activity in transaction networks.

    Args:
        edges_with_labels_df: pd.DataFrame
            DataFrame containing transaction edges with columns:
            - 'Time step': temporal information
            - 'input_address': source node address
            - 'output_address': destination node address
            - (optionally) edge features or labels
        nodes_with_classes_df: pd.DataFrame
            DataFrame with node features and class labels, containing columns:
            - 'address': unique node identifier
            - 'Time step': when the node first appeared
            - 'class': node class (1=illicit, 2=licit, 3=unknown)
            - (additional) node feature columns
        first_time_step: int, default=1
            Starting time step for graph generation (inclusive)
        last_time_step: int, default=49
            Ending time step for graph generation (inclusive)
        max_walk_length: int, default=2
            Maximum number of hops to consider when computing neighborhood-based labels
        time_horizon: int, default=3
            Number of future time steps to look ahead for emergence prediction
            (graphs will only be generated up to max_time_step - time_horizon)
        use_distance_labels: bool, default=False
            If True, labels are distances (0, 1, 2, ..., -1 for unreachable)
            If False, labels are binary (0=no emergence, 1=emergence in neighborhood)
        keep_class_labels_as_features: bool, default=False
            If True, include node class labels as features (may cause label leakage)
            If False, exclude class labels from node features
        add_staleness_feature: bool, default=False
            If True, add a staleness feature (current_time_step - first_appearance_time) to node features
        ignore_illict: bool, default=True
            If True, nodes that are already illicit at time t receive default labels
            and are excluded from the positive label set, which means that if they will be
            a neighbour to a future illicit, that wont be counting towards their current
            neihbours positive labelling
        ignore_previously_transacting_with_illicit: bool, default=True
            If True, nodes with any illicit transaction history up to time t
            receive default labels and are excluded from the positive label set,
            which means that if they will be a neighbour to a future illicit, that
            wont be counting towards their current neihbours positive labelling
        profile: bool, default=False
            If True, print detailed timing information for each step (useful for debugging performance)
        cumulative: bool, default=False
            If True, builds a monotonically increasing grpah by iteratively adding the next timestep to the current graph.
            Otherwise, builds a seperate graph for each timestep

    Returns:
        list[Data]: List of PyTorch Geometric Data objects, one per time step, where each contains:
            - x: node features [num_nodes, num_features]
            - edge_index: graph structure [2, num_edges]
            - y: emergence prediction labels [num_nodes]
            - node_class: ground truth node classes [num_nodes]
            - num_nodes: number of nodes in the graph
            - time_step: the time step for this graph
    """

    # get some overall stats and print them out
    time_steps = sorted(nodes_with_classes_df['Time step'].unique())
    all_addresses = nodes_with_classes_df['address'].unique()
    num_nodes_total = len(all_addresses)
    print(f"Total unique addresses across all time: {num_nodes_total}")
    print(f"Total time steps: {len(time_steps)}")

    # calculate valid time step range based on the input params
    max_time_step = time_steps[-1]
    valid_time_steps = [t for t in range(first_time_step, last_time_step + 1, 1) if t <= max_time_step - time_horizon]
    print(f"Generating {len(valid_time_steps)} graphs (time steps {valid_time_steps[0]} to {valid_time_steps[-1]})...\n")

    # OPTIMIZATION: Pre-compute illicit addresses once (never changes across time steps)
    all_illicit_addresses = set(nodes_with_classes_df[nodes_with_classes_df['class'] == 1]['address'].values)
    
    # OPTIMIZATION: Pre-group edges by time step for O(1) lookup instead of O(E) filtering
    print("Pre-processing edges by time step...")
    t_preprocess = time.time()
    edges_by_timestep = {}
    for t in tqdm(edges_with_labels_df['Time step'].unique(), desc="Grouping edges", disable=not profile):
        edges_by_timestep[t] = edges_with_labels_df[edges_with_labels_df['Time step'] == t]
    preprocess_time = time.time() - t_preprocess
    if profile:
        print(f"  Edge pre-processing took: {preprocess_time:.2f}s\n")
    
    # OPTIMIZATION: Sort DataFrames by Time step once for faster cumulative filtering
    nodes_sorted = nodes_with_classes_df.sort_values('Time step')
    edges_sorted = edges_with_labels_df.sort_values('Time step')

    # generate graphs
    print("\nBuilding graphs...")
    t_build_start = time.time()
    graphs = []
    for t in tqdm(valid_time_steps, desc="Time steps", disable=not profile):
        graph = build_emergence_graph_at_timestep(
            current_time_step=t,
            nodes_with_classes_df=nodes_sorted,
            edges_with_labels_df=edges_sorted,
            all_illicit_addresses=all_illicit_addresses,  # Pass pre-computed set
            edges_by_timestep=edges_by_timestep,  # Pass pre-grouped edges
            keep_class_labels_as_features=keep_class_labels_as_features,
            add_staleness_feature=add_staleness_feature,
            use_distance_labels=use_distance_labels,
            max_walk_length=max_walk_length,
            time_horizon=time_horizon,
            ignore_illict=ignore_illict,
            ignore_previously_transacting_with_illicit=ignore_previously_transacting_with_illicit,
            profile=profile,
            cumulative=cumulative
        )
        graphs.append(graph)
        
        # print stats for this time step
        label_counts = Counter(graph.y.numpy())
        if not profile:
            print(f"  t={t}: nodes={graph.num_nodes:6d}, edges={graph.edge_index.shape[1]:7d}, labels={dict(sorted(label_counts.items()))}")
    
    total_build_time = time.time() - t_build_start

    print(f"\nStored {len(graphs)} graphs")
    
    if profile:
        print("\n" + "="*60)
        print("OVERALL TIMING SUMMARY")
        print("="*60)
        print(f"Pre-processing: {preprocess_time:8.2f}s")
        print(f"Graph building: {total_build_time:8.2f}s ({total_build_time/len(graphs):.2f}s per graph)")
        print(f"Total time:     {preprocess_time + total_build_time:8.2f}s")
        print("="*60)

    return graphs