import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_accuracy(predictions, targets):
    """Calculate accuracy for classification or segmentation"""
    if len(predictions.shape) > 1:
        # For segmentation: flatten point-wise predictions
        predictions = predictions.flatten()
        targets = targets.flatten()
    
    correct = (predictions == targets).sum()
    total = len(targets)
    
    return float(correct) / total * 100.0

def calculate_iou(predictions, targets, num_classes=None):
    """
    Calculate Intersection over Union (IoU) for segmentation
    
    Args:
        predictions: predicted labels (N,) or (B, N)
        targets: ground truth labels (N,) or (B, N)
        num_classes: number of classes (optional)
    
    Returns:
        mean_iou: mean IoU across all classes
        per_class_iou: IoU for each class
    """
    # Flatten if necessary
    if len(predictions.shape) > 1:
        predictions = predictions.flatten()
        targets = targets.flatten()
    
    # Remove invalid labels
    valid_mask = targets >= 0
    predictions = predictions[valid_mask]
    targets = targets[valid_mask]
    
    if num_classes is None:
        num_classes = max(int(targets.max()) + 1, int(predictions.max()) + 1)
    
    per_class_iou = []
    
    for class_id in range(num_classes):
        # True positives, false positives, false negatives
        pred_mask = predictions == class_id
        true_mask = targets == class_id
        
        intersection = np.logical_and(pred_mask, true_mask).sum()
        union = np.logical_or(pred_mask, true_mask).sum()
        
        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = intersection / union
        
        per_class_iou.append(iou)
    
    mean_iou = np.mean(per_class_iou)
    
    return mean_iou * 100.0, per_class_iou

def calculate_part_iou(predictions, targets, category_labels, num_categories):
    """
    Calculate part-level IoU for each category separately
    Used for ShapeNet Part segmentation evaluation
    """
    category_ious = {}
    
    for cat_id in range(num_categories):
        cat_mask = category_labels == cat_id
        if not cat_mask.any():
            continue
            
        cat_pred = predictions[cat_mask]
        cat_target = targets[cat_mask]
        
        # Calculate IoU for this category
        mean_iou, per_class_iou = calculate_iou(cat_pred, cat_target)
        category_ious[cat_id] = {
            'mean_iou': mean_iou,
            'per_class_iou': per_class_iou
        }
    
    # Overall mean IoU across all categories
    overall_mean_iou = np.mean([cat_data['mean_iou'] for cat_data in category_ious.values()])
    
    return overall_mean_iou, category_ious

def calculate_classification_metrics(predictions, targets, class_names=None):
    """
    Calculate comprehensive classification metrics
    
    Returns:
        dict: Dictionary containing accuracy, precision, recall, f1-score
    """
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.cpu().numpy()
    
    accuracy = calculate_accuracy(predictions, targets)
    
    # Confusion matrix and classification report
    cm = confusion_matrix(targets, predictions)
    
    if class_names is None:
        class_names = [f'Class_{i}' for i in range(len(np.unique(targets)))]
    
    report = classification_report(targets, predictions, 
                                 target_names=class_names, 
                                 output_dict=True, 
                                 zero_division=0)
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report,
        'per_class_precision': [report[cls]['precision'] for cls in class_names],
        'per_class_recall': [report[cls]['recall'] for cls in class_names],
        'per_class_f1': [report[cls]['f1-score'] for cls in class_names],
        'macro_precision': report['macro avg']['precision'],
        'macro_recall': report['macro avg']['recall'],
        'macro_f1': report['macro avg']['f1-score'],
        'weighted_precision': report['weighted avg']['precision'],
        'weighted_recall': report['weighted avg']['recall'],
        'weighted_f1': report['weighted avg']['f1-score']
    }

def calculate_segmentation_metrics(predictions, targets, category_labels=None, num_categories=None):
    """
    Calculate comprehensive segmentation metrics
    
    Returns:
        dict: Dictionary containing accuracy, mIoU, and per-class metrics
    """
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.cpu().numpy()
    if torch.is_tensor(category_labels):
        category_labels = category_labels.cpu().numpy()
    
    # Point-wise accuracy
    accuracy = calculate_accuracy(predictions, targets)
    
    # Overall IoU
    mean_iou, per_class_iou = calculate_iou(predictions, targets)
    
    metrics = {
        'accuracy': accuracy,
        'mean_iou': mean_iou,
        'per_class_iou': per_class_iou
    }
    
    # Category-specific IoU (for part segmentation)
    if category_labels is not None and num_categories is not None:
        overall_part_iou, category_ious = calculate_part_iou(
            predictions, targets, category_labels, num_categories
        )
        metrics.update({
            'overall_part_iou': overall_part_iou,
            'category_ious': category_ious
        })
    
    return metrics

def plot_confusion_matrix(confusion_matrix, class_names, save_path=None):
    """Plot confusion matrix heatmap"""
    plt.figure(figsize=(10, 8))
    
    # Normalize confusion matrix
    cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)
    
    sns.heatmap(cm_normalized, 
                annot=True, 
                fmt='.2f', 
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_per_class_metrics(metrics, class_names, metric_name='f1-score', save_path=None):
    """Plot per-class metrics (precision, recall, f1-score)"""
    plt.figure(figsize=(12, 6))
    
    if metric_name == 'precision':
        values = metrics['per_class_precision']
    elif metric_name == 'recall':
        values = metrics['per_class_recall']
    elif metric_name == 'f1-score':
        values = metrics['per_class_f1']
    elif metric_name == 'iou':
        values = metrics['per_class_iou']
    else:
        raise ValueError(f"Unknown metric: {metric_name}")
    
    bars = plt.bar(range(len(class_names)), values, alpha=0.8)
    plt.xlabel('Class')
    plt.ylabel(metric_name.title())
    plt.title(f'Per-Class {metric_name.title()}')
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    plt.ylim(0, 1.0 if metric_name != 'iou' else 100)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def print_metrics_summary(metrics, task='classification'):
    """Print a summary of metrics"""
    print("\n" + "="*50)
    print(f"{task.upper()} METRICS SUMMARY")
    print("="*50)
    
    print(f"Accuracy: {metrics['accuracy']:.2f}%")
    
    if task == 'classification':
        print(f"Macro Precision: {metrics['macro_precision']:.3f}")
        print(f"Macro Recall: {metrics['macro_recall']:.3f}")
        print(f"Macro F1-Score: {metrics['macro_f1']:.3f}")
        print(f"Weighted F1-Score: {metrics['weighted_f1']:.3f}")
        
    elif task == 'segmentation':
        print(f"Mean IoU: {metrics['mean_iou']:.2f}%")
        if 'overall_part_iou' in metrics:
            print(f"Overall Part IoU: {metrics['overall_part_iou']:.2f}%")
    
    print("="*50)