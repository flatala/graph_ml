"""
Model architectures for temporal graph neural networks.
"""

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import EvolveGCNO, EvolveGCNH


class TemporalGCNClassifierO(nn.Module):
    """
    EvolveGCN-O based classifier for temporal graph learning.
    
    EvolveGCN-O uses an RNN (LSTM) to evolve GCN parameters over time based on 
    the previous parameters (simpler and faster than EvolveGCN-H).
    
    Args:
        num_features: Number of input node features
        hidden_dim: Hidden dimension size (for future multi-layer versions)
        num_classes: Number of output classes
    """
    
    def __init__(self, num_features, hidden_dim, num_classes):
        super(TemporalGCNClassifierO, self).__init__()
        
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # EvolveGCNO layer - creates and evolves its own GCN layer
        self.recurrent = EvolveGCNO(in_channels=num_features)
        
        # Classifier head that maps from evolved features to classes
        self.classifier = nn.Linear(num_features, num_classes)
    
    def reset_hidden_state(self):
        """
        Reset the internal LSTM hidden state of EvolveGCNO.
        This should be called at the start of each epoch and before evaluation
        to ensure independence between training/validation phases.
        
        NOTE: This does NOT reset learned weights - it only resets the RNN's hidden state.
        """
        if hasattr(self.recurrent, '_h'):
            self.recurrent._h = None
        if hasattr(self.recurrent, '_c'):
            self.recurrent._c = None
        
    def forward(self, x, edge_index):
        """Forward pass through evolved GCN layer."""
        # EvolveGCNO internally updates weights based on temporal sequence
        h = self.recurrent(x, edge_index)
        
        # Apply activation and dropout
        h = F.relu(h)
        h = F.dropout(h, p=0.3, training=self.training)
        
        # Classification head
        out = self.classifier(h)
        return out


class TemporalGCNClassifierH(nn.Module):
    """
    EvolveGCN-H based classifier for temporal graph learning.
    
    EvolveGCN-H uses an RNN (LSTM) to evolve GCN parameters over time based on 
    node embeddings (more expressive but slower than EvolveGCN-O).
    
    NOTE: EvolveGCN-H requires knowing the number of nodes upfront because it uses
    TopK pooling internally. You must call set_num_nodes() before first forward pass.
    
    Args:
        num_features: Number of input node features
        hidden_dim: Hidden dimension size (for future multi-layer versions)
        num_classes: Number of output classes
        num_nodes: Number of nodes in the graph (required for EvolveGCN-H)
    """
    
    def __init__(self, num_features, hidden_dim, num_classes, num_nodes):
        super(TemporalGCNClassifierH, self).__init__()
        
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_nodes = num_nodes
        
        # EvolveGCNH layer - creates and evolves its own GCN layer
        # Requires num_of_nodes for internal TopK pooling
        self.recurrent = EvolveGCNH(
            in_channels=num_features,
            num_of_nodes=num_nodes,
            improved=False
        )
        
        # Classifier head that maps from evolved features to classes
        self.classifier = nn.Linear(num_features, num_classes)
    
    def reset_hidden_state(self):
        """
        Reset the internal LSTM hidden state of EvolveGCNH.
        This should be called at the start of each epoch and before evaluation
        to ensure independence between training/validation phases.
        
        NOTE: This does NOT reset learned weights - it only resets the RNN's hidden state.
        """
        if hasattr(self.recurrent, '_h'):
            self.recurrent._h = None
        if hasattr(self.recurrent, '_c'):
            self.recurrent._c = None
        
    def forward(self, x, edge_index):
        """Forward pass through evolved GCN layer."""
        # EvolveGCNH internally updates weights based on node embeddings
        h = self.recurrent(x, edge_index)
        
        # Apply activation and dropout
        h = F.relu(h)
        h = F.dropout(h, p=0.3, training=self.training)
        
        # Classification head
        out = self.classifier(h)
        return out
