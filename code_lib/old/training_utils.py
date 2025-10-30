"""
Training utilities for temporal GNN models.
Contains shared training, evaluation, and metrics computation functions.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, average_precision_score,
    confusion_matrix
)


def train_epoch_temporal(model, graphs, optimizer, criterion, device):
    """
    Train one epoch with temporal sequence.
    Accumulates loss across all time steps before backward pass.
    
    Args:
        model: The temporal GNN model
        graphs: List of temporal graph snapshots
        optimizer: Optimizer
        criterion: Loss function (FocalLoss or weighted cross-entropy)
        device: Device to train on
    
    Returns:
        tuple: (avg_loss, avg_accuracy)
    """
    model.train()
    
    total_loss = 0
    total_correct = 0
    total_nodes = 0
    accumulated_loss = 0
    
    optimizer.zero_grad()
    
    # Process graphs in temporal order and accumulate loss
    for graph in graphs:
        graph = graph.to(device)
        
        # Forward pass
        out = model(graph.x, graph.edge_index)
        
        # Compute loss using the provided criterion
        loss = criterion(out, graph.y)
        
        # Accumulate loss for backward pass
        accumulated_loss += loss
        
        # Track metrics
        pred = out.argmax(dim=1)
        correct = (pred == graph.y).sum().item()
        total_loss += loss.item() * graph.num_nodes
        total_correct += correct
        total_nodes += graph.num_nodes
    
    # Single backward pass through entire temporal sequence
    accumulated_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
    
    avg_loss = total_loss / total_nodes
    avg_acc = total_correct / total_nodes
    return avg_loss, avg_acc


def evaluate_temporal(model, graphs, device, return_predictions=False, compute_metrics=True):
    """
    Evaluate model on temporal sequence with comprehensive metrics.
    
    Args:
        model: The temporal GNN model
        graphs: List of temporal graph snapshots
        device: Device to evaluate on
        return_predictions: Whether to return predictions and labels
        compute_metrics: Whether to compute comprehensive metrics
    
    Returns:
        If return_predictions=False: dict of metrics
        If return_predictions=True: (metrics dict, predictions, labels, probabilities)
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for graph in graphs:
            graph = graph.to(device)
            
            # Forward pass
            out = model(graph.x, graph.edge_index)
            
            probs = F.softmax(out, dim=1)
            preds = out.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(graph.y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    preds_arr = np.array(all_preds)
    labels_arr = np.array(all_labels)
    probs_arr = np.array(all_probs)
    
    # Compute comprehensive metrics
    if compute_metrics:
        metrics = compute_comprehensive_metrics(labels_arr, preds_arr, probs_arr)
    else:
        # Quick evaluation - just accuracy
        metrics = {'accuracy': (preds_arr == labels_arr).mean()}
    
    if return_predictions:
        return metrics, preds_arr, labels_arr, probs_arr
    return metrics


def compute_comprehensive_metrics(labels, preds, probs):
    """
    Compute comprehensive evaluation metrics for imbalanced classification.
    
    Args:
        labels: true labels (numpy array)
        preds: predicted labels (numpy array)
        probs: predicted probabilities [N, num_classes] (numpy array)
    
    Returns:
        dict: dictionary of metrics including accuracy, precision, recall, F1, ROC-AUC, AUPRC, etc.
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(labels, preds)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, average=None, zero_division=0
    )
    
    metrics['precision_neg'] = precision[0] if len(precision) > 0 else 0
    metrics['precision_pos'] = precision[1] if len(precision) > 1 else 0
    metrics['recall_neg'] = recall[0] if len(recall) > 0 else 0
    metrics['recall_pos'] = recall[1] if len(recall) > 1 else 0
    metrics['f1_neg'] = f1[0] if len(f1) > 0 else 0
    metrics['f1_pos'] = f1[1] if len(f1) > 1 else 0
    
    # Weighted/macro averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average='macro', zero_division=0
    )
    metrics['precision_macro'] = precision_macro
    metrics['recall_macro'] = recall_macro
    metrics['f1_macro'] = f1_macro
    
    # Binary metrics (for positive class)
    precision_bin, recall_bin, f1_bin, _ = precision_recall_fscore_support(
        labels, preds, average='binary', pos_label=1, zero_division=0
    )
    metrics['precision_binary'] = precision_bin
    metrics['recall_binary'] = recall_bin
    metrics['f1_binary'] = f1_bin
    
    # ROC-AUC and AUPRC (only if both classes present)
    if len(np.unique(labels)) == 2:
        try:
            metrics['roc_auc'] = roc_auc_score(labels, probs[:, 1])
            # AUPRC is more informative for imbalanced data
            metrics['auprc'] = average_precision_score(labels, probs[:, 1])
        except Exception:
            metrics['roc_auc'] = 0.0
            metrics['auprc'] = 0.0
    else:
        metrics['roc_auc'] = 0.0
        metrics['auprc'] = 0.0
    
    # Confusion matrix components
    if len(np.unique(labels)) == 2:
        cm = confusion_matrix(labels, preds)
        metrics['tn'] = int(cm[0, 0])
        metrics['fp'] = int(cm[0, 1])
        metrics['fn'] = int(cm[1, 0])
        metrics['tp'] = int(cm[1, 1])
        
        # False positive rate and False negative rate
        metrics['fpr'] = metrics['fp'] / (metrics['fp'] + metrics['tn']) if (metrics['fp'] + metrics['tn']) > 0 else 0
        metrics['fnr'] = metrics['fn'] / (metrics['fn'] + metrics['tp']) if (metrics['fn'] + metrics['tp']) > 0 else 0
    else:
        metrics['tn'] = metrics['fp'] = metrics['fn'] = metrics['tp'] = 0
        metrics['fpr'] = metrics['fnr'] = 0.0
    
    # Support (class distribution)
    metrics['support_neg'] = int(support[0]) if len(support) > 0 else 0
    metrics['support_pos'] = int(support[1]) if len(support) > 1 else 0
    metrics['total_samples'] = len(labels)
    metrics['pos_ratio'] = metrics['support_pos'] / len(labels) if len(labels) > 0 else 0
    
    return metrics


def print_metrics_summary(metrics, prefix=""):
    """Print formatted metrics summary."""
    print(f"\n{prefix}Performance Metrics:")
    print(f"{'='*70}")
    print(f"Overall Accuracy:     {metrics['accuracy']:.4f}")
    print(f"ROC-AUC Score:        {metrics['roc_auc']:.4f}")
    print(f"AUPRC Score:          {metrics['auprc']:.4f}  ← Better for imbalanced data")
    print(f"\n{'Per-Class Metrics':<25} {'Negative':>12} {'Positive':>12}")
    print(f"{'-'*50}")
    print(f"{'Precision':<25} {metrics['precision_neg']:>12.4f} {metrics['precision_pos']:>12.4f}")
    print(f"{'Recall':<25} {metrics['recall_neg']:>12.4f} {metrics['recall_pos']:>12.4f}")
    print(f"{'F1-Score':<25} {metrics['f1_neg']:>12.4f} {metrics['f1_pos']:>12.4f}")
    print(f"{'Support':<25} {metrics['support_neg']:>12} {metrics['support_pos']:>12}")
    print(f"\n{'Confusion Matrix Components:'}")
    print(f"  TN={metrics['tn']:6d}  FP={metrics['fp']:6d}  (FPR: {metrics['fpr']:.4f})")
    print(f"  FN={metrics['fn']:6d}  TP={metrics['tp']:6d}  (FNR: {metrics['fnr']:.4f})")
    print(f"{'='*70}")


def compute_temporal_stability(model, graphs, device):
    """
    Compute how stable predictions are across consecutive time steps.
    High stability = predictions don't fluctuate wildly.
    
    Args:
        model: The temporal GNN model
        graphs: List of temporal graph snapshots
        device: Device to evaluate on
    
    Returns:
        dict: stability metrics with keys 'prediction_stability', 'num_timesteps', 'num_comparisons'
    """
    model.eval()
    
    all_node_preds = []  # List of prediction arrays, one per time step
    
    with torch.no_grad():
        for graph in graphs:
            graph = graph.to(device)
            out = model(graph.x, graph.edge_index)
            preds = out.argmax(dim=1).cpu().numpy()
            all_node_preds.append(preds)
    
    if len(all_node_preds) < 2:
        return {'prediction_stability': 0.0, 'num_timesteps': len(all_node_preds)}
    
    # For nodes that exist in consecutive time steps, measure prediction changes
    stability_scores = []
    for t in range(len(all_node_preds) - 1):
        curr_preds = all_node_preds[t]
        next_preds = all_node_preds[t + 1]
        
        # Compare only up to the minimum size (simple approximation)
        min_size = min(len(curr_preds), len(next_preds))
        if min_size > 0:
            # What fraction of predictions stayed the same?
            stability = (curr_preds[:min_size] == next_preds[:min_size]).mean()
            stability_scores.append(stability)
    
    return {
        'prediction_stability': np.mean(stability_scores) if stability_scores else 0.0,
        'num_timesteps': len(all_node_preds),
        'num_comparisons': len(stability_scores)
    }


def compute_class_weights(graphs):
    """
    Compute class weights for handling imbalance.
    
    Args:
        graphs: List of temporal graph snapshots
    
    Returns:
        torch.Tensor: Class weights [weight_neg, weight_pos]
    """
    total_pos = sum((g.y == 1).sum().item() for g in graphs)
    total_neg = sum((g.y == 0).sum().item() for g in graphs)
    total = total_pos + total_neg
    
    # Inverse frequency weighting
    weight_pos = total / (2 * total_pos) if total_pos > 0 else 1.0
    weight_neg = total / (2 * total_neg) if total_neg > 0 else 1.0
    
    weights = torch.tensor([weight_neg, weight_pos], dtype=torch.float)
    print(f"Class weights: [0: {weight_neg:.3f}, 1: {weight_pos:.3f}]")
    print(f"Positive ratio: {100*total_pos/total:.2f}%")
    
    return weights


def analyze_class_distribution(graphs):
    """
    Analyze class distribution across all graphs to guide focal loss parameter selection.
    
    Args:
        graphs: List of temporal graph snapshots
    
    Returns:
        tuple: (pos_ratio, recommended_alpha_pos, recommended_alpha_neg, recommended_gamma)
    """
    total_pos = sum((g.y == 1).sum().item() for g in graphs)
    total_neg = sum((g.y == 0).sum().item() for g in graphs)
    total = total_pos + total_neg
    pos_ratio = 100 * total_pos / total if total > 0 else 0
    
    print(f"\n{'='*70}")
    print("CLASS DISTRIBUTION ANALYSIS")
    print(f"{'='*70}")
    print(f"Positive samples: {total_pos:7d} ({pos_ratio:.2f}%)")
    print(f"Negative samples: {total_neg:7d} ({100-pos_ratio:.2f}%)")
    print(f"Total samples:    {total:7d}")
    print(f"Imbalance ratio:  1:{total_neg/total_pos:.1f} (neg:pos)" if total_pos > 0 else "N/A")
    
    # Recommend focal loss parameters
    print("\nRECOMMENDED FOCAL LOSS PARAMETERS:")
    if pos_ratio < 5:
        alpha_pos = 0.15
        gamma = 2.5
        print("  Highly imbalanced (<5% positive)")
    elif pos_ratio < 10:
        alpha_pos = 0.20
        gamma = 2.0
        print("  Very imbalanced (5-10% positive)")
    elif pos_ratio < 20:
        alpha_pos = 0.25
        gamma = 2.0
        print("  Imbalanced (10-20% positive)")
    else:
        alpha_pos = 0.30
        gamma = 1.5
        print("  Moderately imbalanced (20%+ positive)")
    
    print(f"  → alpha_pos = {alpha_pos:.2f}")
    print(f"  → alpha_neg = {1-alpha_pos:.2f}")
    print(f"  → gamma = {gamma:.1f}")
    print(f"{'='*70}\n")
    
    return pos_ratio, alpha_pos, 1-alpha_pos, gamma
