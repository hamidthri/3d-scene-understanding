import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import os

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def plot_training_curves(train_losses, train_accuracies, test_accuracies, save_path=None):
    """Plot training curves for classification"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Plot loss
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(epochs, train_accuracies, 'g-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, test_accuracies, 'r-', label='Test Accuracy', linewidth=2)
    ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")
    
    plt.show()

def plot_segmentation_curves(train_losses, train_accuracies, test_accuracies, test_ious, save_path=None):
    """Plot training curves for segmentation"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Plot loss
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(epochs, train_accuracies, 'g-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, test_accuracies, 'r-', label='Test Accuracy', linewidth=2)
    ax2.set_title('Point-wise Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot IoU
    ax3.plot(epochs, test_ious, 'purple', label='Test mIoU', linewidth=2)
    ax3.set_title('Mean Intersection over Union', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('mIoU (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Combined metrics
    ax4.plot(epochs, test_accuracies, 'r-', label='Test Accuracy', linewidth=2, alpha=0.8)
    ax4_twin = ax4.twinx()
    ax4_twin.plot(epochs, test_ious, 'purple', label='Test mIoU', linewidth=2, alpha=0.8)
    
    ax4.set_title('Combined Metrics', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy (%)', color='r')
    ax4_twin.set_ylabel('mIoU (%)', color='purple')
    
    # Legends
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")
    
    plt.show()

def plot_point_cloud(points, colors=None, title="Point Cloud", save_path=None, figsize=(10, 8)):
    """
    Plot 3D point cloud
    
    Args:
        points: numpy array of shape (N, 3) or (3, N)
        colors: colors for each point (optional)
        title: plot title
        save_path: path to save plot
        figsize: figure size
    """
    # Handle different input shapes
    if points.shape[0] == 3 and points.shape[1] != 3:
        points = points.T
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    if colors is not None:
        scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                           c=colors, cmap='tab10', s=20, alpha=0.8)
        plt.colorbar(scatter)
    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                  c='blue', s=20, alpha=0.8)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Make axes equal
    max_range = np.array([points[:, 0].max() - points[:, 0].min(),
                         points[:, 1].max() - points[:, 1].min(),
                         points[:, 2].max() - points[:, 2].min()]).max() / 2.0
    
    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Point cloud plot saved to: {save_path}")
    
    plt.show()

def plot_segmentation_results(points, true_labels, pred_labels, title="Segmentation Results", save_path=None):
    """
    Plot segmentation results with ground truth and predictions side by side
    
    Args:
        points: numpy array of shape (N, 3) or (3, N)
        true_labels: ground truth segmentation labels
        pred_labels: predicted segmentation labels
        title: plot title
        save_path: path to save plot
    """
    # Handle different input shapes
    if points.shape[0] == 3 and points.shape[1] != 3:
        points = points.T
    
    fig = plt.figure(figsize=(20, 8))
    
    # Ground Truth
    ax1 = fig.add_subplot(121, projection='3d')
    scatter1 = ax1.scatter(points[:, 0], points[:, 1], points[:, 2], 
                          c=true_labels, cmap='tab10', s=20, alpha=0.8)
    ax1.set_title('Ground Truth', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    plt.colorbar(scatter1, ax=ax1, shrink=0.5)
    
    # Predictions
    ax2 = fig.add_subplot(122, projection='3d')
    scatter2 = ax2.scatter(points[:, 0], points[:, 1], points[:, 2], 
                          c=pred_labels, cmap='tab10', s=20, alpha=0.8)
    ax2.set_title('Predictions', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    plt.colorbar(scatter2, ax=ax2, shrink=0.5)
    
    # Make axes equal for both subplots
    max_range = np.array([points[:, 0].max() - points[:, 0].min(),
                         points[:, 1].max() - points[:, 1].min(),
                         points[:, 2].max() - points[:, 2].min()]).max() / 2.0
    
    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
    
    for ax in [ax1, ax2]:
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Segmentation results saved to: {save_path}")
    
    plt.show()

def plot_classification_results(points_list, predictions, true_labels, class_names, 
                               title="Classification Results", save_path=None, max_samples=8):
    """
    Plot classification results for multiple samples
    
    Args:
        points_list: list of point clouds
        predictions: predicted class labels
        true_labels: ground truth class labels
        class_names: list of class names
        title: plot title
        save_path: path to save plot
        max_samples: maximum number of samples to plot
    """
    n_samples = min(len(points_list), max_samples)
    cols = 4
    rows = (n_samples + cols - 1) // cols
    
    fig = plt.figure(figsize=(20, 5 * rows))
    
    for i in range(n_samples):
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
        
        points = points_list[i]
        if points.shape[0] == 3:
            points = points.T
        
        pred_class = class_names[predictions[i]]
        true_class = class_names[true_labels[i]]
        
        # Color based on correctness
        color = 'green' if predictions[i] == true_labels[i] else 'red'
        
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                  c=color, s=10, alpha=0.6)
        
        ax.set_title(f'True: {true_class}\nPred: {pred_class}', 
                    fontsize=10, fontweight='bold',
                    color=color)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Remove tick labels to save space
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Classification results saved to: {save_path}")
    
    plt.show()

def plot_model_comparison(results_dict, metric='accuracy', title="Model Comparison", save_path=None):
    """
    Plot comparison of different models
    
    Args:
        results_dict: dict with model names as keys and metrics as values
        metric: metric to compare ('accuracy', 'mean_iou', etc.)
        title: plot title
        save_path: path to save plot
    """
    models = list(results_dict.keys())
    values = [results_dict[model][metric] for model in models]
    
    plt.figure(figsize=(12, 6))
    
    bars = plt.bar(models, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(models)])
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Model')
    plt.ylabel(f'{metric.replace("_", " ").title()}{"(%)" if "accuracy" in metric or "iou" in metric else ""}')
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, values)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{value:.2f}{"%" if "accuracy" in metric or "iou" in metric else ""}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison saved to: {save_path}")
    
    plt.show()

def plot_learning_curves_comparison(train_curves_dict, test_curves_dict, 
                                  metric='accuracy', title="Learning Curves Comparison", save_path=None):
    """
    Compare learning curves of multiple models
    
    Args:
        train_curves_dict: dict with model names as keys and training curves as values
        test_curves_dict: dict with model names as keys and test curves as values
        metric: metric name for y-axis label
        title: plot title
        save_path: path to save plot
    """
    plt.figure(figsize=(15, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, model in enumerate(train_curves_dict.keys()):
        train_curve = train_curves_dict[model]
        test_curve = test_curves_dict[model]
        epochs = range(1, len(train_curve) + 1)
        
        color = colors[i % len(colors)]
        
        plt.plot(epochs, train_curve, '--', color=color, alpha=0.7, 
                label=f'{model} (train)', linewidth=2)
        plt.plot(epochs, test_curve, '-', color=color, 
                label=f'{model} (test)', linewidth=2)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel(f'{metric.replace("_", " ").title()}{"(%)" if "accuracy" in metric or "iou" in metric else ""}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Learning curves comparison saved to: {save_path}")
    
    plt.show()

def create_training_gif(metric_history, metric_name='accuracy', save_path='training_progress.gif'):
    """
    Create animated GIF showing training progress
    
    Args:
        metric_history: list of metric values over epochs
        metric_name: name of the metric
        save_path: path to save GIF
    """
    try:
        from matplotlib.animation import PillowWriter
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        def animate(frame):
            ax.clear()
            epochs = range(1, frame + 2)
            values = metric_history[:frame + 1]
            
            ax.plot(epochs, values, 'b-', linewidth=2, marker='o')
            ax.set_title(f'Training Progress - Epoch {frame + 1}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(f'{metric_name.title()}{"(%)" if "accuracy" in metric_name or "iou" in metric_name else ""}')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(1, len(metric_history))
            ax.set_ylim(min(metric_history) - 5, max(metric_history) + 5)
        
        writer = PillowWriter(fps=2)
        
        import matplotlib.animation as animation
        ani = animation.FuncAnimation(fig, animate, frames=len(metric_history), repeat=False)
        ani.save(save_path, writer=writer)
        
        plt.close()
        print(f"Training progress GIF saved to: {save_path}")
        
    except ImportError:
        print("Warning: Pillow not available. Cannot create training GIF.")
    except Exception as e:
        print(f"Error creating training GIF: {e}")

def save_results_summary(results, save_path='results_summary.txt'):
    """
    Save results summary to text file
    
    Args:
        results: dictionary containing all results
        save_path: path to save summary
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    with open(save_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("TRAINING RESULTS SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        for key, value in results.items():
            if isinstance(value, (list, np.ndarray)):
                if len(value) > 0:
                    f.write(f"{key}:\n")
                    f.write(f"  Final: {value[-1]:.4f}\n")
                    f.write(f"  Best: {max(value) if 'loss' not in key.lower() else min(value):.4f}\n")
                    f.write(f"  Average: {np.mean(value):.4f}\n\n")
            else:
                f.write(f"{key}: {value}\n")
        
        f.write("=" * 60 + "\n")
    
    print(f"Results summary saved to: {save_path}")


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


def visualize_classification_predictions(point_clouds, true_labels, pred_labels, classes, model_name, save_path):
    n_samples = min(len(point_clouds), 10)
    fig = plt.figure(figsize=(15, 8))
    
    for i in range(n_samples):
        ax = fig.add_subplot(2, 5, i+1, projection='3d')
        
        points = point_clouds[i]
        if points.shape[0] == 3:
            points = points.T
        
        true_class = classes[true_labels[i]] if true_labels[i] < len(classes) else f"Class_{true_labels[i]}"
        pred_class = classes[pred_labels[i]] if pred_labels[i] < len(classes) else f"Class_{pred_labels[i]}"
        
        correct = true_labels[i] == pred_labels[i]
        color = 'green' if correct else 'red'
        
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                  c=points[:, 2], cmap='viridis', s=1, alpha=0.8)
        
        ax.set_title(f"True: {true_class}\nPred: {pred_class}", 
                    fontsize=8, color=color, weight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.tick_params(labelsize=6)
        
        ax.view_init(elev=20, azim=45)
        ax.set_box_aspect([1,1,1])
    
    plt.suptitle(f'{model_name.upper()} - Classification Predictions', fontsize=14, weight='bold')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Classification visualization saved to: {save_path}")


def visualize_segmentation_predictions(point_clouds, true_seg_labels, pred_seg_labels, model_name, save_path):
    n_samples = min(len(point_clouds), 5)
    fig = plt.figure(figsize=(18, 8))
    
    for i in range(n_samples):
        points = point_clouds[i]
        true_segs = true_seg_labels[i]
        pred_segs = pred_seg_labels[i]
        
        if points.shape[0] == 3:
            points = points.T
        
        ax1 = fig.add_subplot(2, n_samples, i+1, projection='3d')
        ax2 = fig.add_subplot(2, n_samples, i+1+n_samples, projection='3d')
        
        unique_true = np.unique(true_segs)
        unique_pred = np.unique(pred_segs)
        
        colors_true = plt.cm.Set1(np.linspace(0, 1, len(unique_true)))
        colors_pred = plt.cm.Set1(np.linspace(0, 1, len(unique_pred)))
        
        for j, part in enumerate(unique_true):
            mask = true_segs == part
            ax1.scatter(points[mask, 0], points[mask, 1], points[mask, 2], 
                       c=[colors_true[j]], s=2, alpha=0.8, label=f'Part {part}')
        
        for j, part in enumerate(unique_pred):
            mask = pred_segs == part
            ax2.scatter(points[mask, 0], points[mask, 1], points[mask, 2], 
                       c=[colors_pred[j]], s=2, alpha=0.8, label=f'Part {part}')
        
        accuracy = np.mean(true_segs == pred_segs)
        
        ax1.set_title(f"Ground Truth\nSample {i+1}", fontsize=9, weight='bold')
        ax2.set_title(f"Prediction\nAccuracy: {accuracy:.2f}", 
                     fontsize=9, weight='bold', 
                     color='green' if accuracy > 0.8 else 'orange' if accuracy > 0.6 else 'red')
        
        for ax in [ax1, ax2]:
            ax.set_xlabel('X', fontsize=7)
            ax.set_ylabel('Y', fontsize=7)
            ax.set_zlabel('Z', fontsize=7)
            ax.tick_params(labelsize=6)
            ax.view_init(elev=20, azim=45)
            ax.set_box_aspect([1,1,1])
    
    plt.suptitle(f'{model_name.upper()} - Segmentation Predictions', fontsize=14, weight='bold')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Segmentation visualization saved to: {save_path}")


def plot_training_curves(train_losses, train_accuracies, test_accuracies, save_path=None):
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.plot(epochs, train_accuracies, 'g-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, test_accuracies, 'r-', label='Test Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Test Accuracy')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()