"""
Loss functions for imbalanced classification in temporal GNN models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Focal Loss was proposed in "Focal Loss for Dense Object Detection" (Lin et al., 2017).
    It down-weights easy examples and focuses training on hard negatives.
    
    Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Where:
    - p_t is the model's estimated probability for the true class
    - alpha_t is a weighting factor for class imbalance (typically alpha=0.25 for rare class)
    - gamma is the focusing parameter (typically gamma=2.0)
    
    Args:
        alpha: Weighting factor in range [0,1] to balance positive/negative examples
               or a list of weights [alpha_neg, alpha_pos]
        gamma: Focusing parameter for modulating loss. gamma=0 is equivalent to CE loss
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: predictions from model (logits, not softmax!) [N, num_classes]
            targets: ground truth labels [N]
        
        Returns:
            focal_loss: computed focal loss
        """
        # Compute standard cross entropy loss (without reduction)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Get probabilities
        p = F.softmax(inputs, dim=1)
        
        # Get the probability of the true class for each sample
        p_t = p.gather(1, targets.view(-1, 1)).squeeze(1)
        
        # Compute focal loss
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = self.alpha * focal_weight * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class BinaryFocalLoss(nn.Module):
    """
    Binary Focal Loss - optimized version for binary classification.
    
    This version allows separate alpha values for positive and negative classes.
    
    Args:
        alpha_pos: Weight for positive class (default: 0.25)
        alpha_neg: Weight for negative class (default: 0.75)
        gamma: Focusing parameter (default: 2.0)
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(self, alpha_pos=0.25, alpha_neg=0.75, gamma=2.0, reduction='mean'):
        super(BinaryFocalLoss, self).__init__()
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: predictions from model (logits) [N, 2]
            targets: ground truth labels [N] with values 0 or 1
        """
        # Compute cross entropy loss (no reduction)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Get probabilities
        p = F.softmax(inputs, dim=1)
        p_t = p.gather(1, targets.view(-1, 1)).squeeze(1)
        
        # Compute focal term: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha weighting (different for pos/neg classes)
        alpha_t = torch.where(targets == 1, 
                              torch.tensor(self.alpha_pos, device=targets.device),
                              torch.tensor(self.alpha_neg, device=targets.device))
        
        # Compute focal loss
        focal_loss = alpha_t * focal_weight * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
