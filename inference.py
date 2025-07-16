import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import DataLoader
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import random
from models.pointnet import PointNetClassifier
from datsets.modelnet10 import ModelNet10


def load_model(model_path='pointnet_final.pth', device='cpu', data_dir=None):
    checkpoint = torch.load(model_path, map_location=device)

    # Handle case where 'classes' are missing in intermediate checkpoint
    if 'classes' not in checkpoint:
        if data_dir is None:
            raise ValueError("Checkpoint is missing 'classes'. Provide data_dir to infer classes from dataset.")
        temp_dataset = ModelNet10Dataset(data_dir, split='test', num_points=checkpoint.get('num_points', 1024))
        class_list = temp_dataset.classes
    else:
        class_list = checkpoint['classes']

    model = PointNetClassifier(
        k=len(class_list),
        feature_transform=checkpoint.get('feature_transform', True)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, class_list, checkpoint.get('num_points', 1024)

def predict_single_sample(model, points, device='cpu'):
    model.eval()
    with torch.no_grad():
        points = points.unsqueeze(0).to(device)
        pred, _, _ = model(points)
        pred_class = pred.data.max(1)[1].item()
        confidence = torch.exp(pred).max().item()
    return pred_class, confidence

def visualize_point_cloud(points, title="Point Cloud", color='blue'):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    if len(points.shape) == 3:
        points = points.squeeze()

    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()

    if points.shape[0] == 3:
        points = points.T

    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
              c=color, s=1, alpha=0.6)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    max_range = np.array([points[:, 0].max() - points[:, 0].min(),
                         points[:, 1].max() - points[:, 1].min(),
                         points[:, 2].max() - points[:, 2].min()]).max() / 2.0

    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    return fig, ax

def evaluate_model(model_path='pointnet_final.pth',
                  data_dir='./modelnet10_data',
                  visualize_samples=True,
                  num_viz_samples=6):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model, classes, num_points = load_model(model_path, device, data_dir=data_dir)


    test_dataset = ModelNet10Dataset(data_dir, split='test', num_points=num_points)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    all_predictions = []
    all_labels = []
    all_points = []

    print("Evaluating model on test set...")

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            pred, _, _ = model(data)
            pred_choice = pred.data.max(1)[1]

            correct += pred_choice.eq(target.data).cpu().sum().item()
            total += target.size(0)

            all_predictions.extend(pred_choice.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            all_points.extend(data.cpu().numpy())

    accuracy = 100. * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    cm = confusion_matrix(all_labels, all_predictions)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix (Accuracy: {accuracy:.2f}%)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=classes))

    if visualize_samples:
        print(f"\nVisualizing {num_viz_samples} random samples...")

        sample_indices = random.sample(range(len(all_predictions)), num_viz_samples)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12), subplot_kw={'projection': '3d'})
        axes = axes.flatten()

        for i, idx in enumerate(sample_indices):
            points = all_points[idx]
            true_label = all_labels[idx]
            pred_label = all_predictions[idx]

            if points.shape[0] == 3:
                points = points.T

            color = 'green' if true_label == pred_label else 'red'

            axes[i].scatter(points[:, 0], points[:, 1], points[:, 2],
                           c=color, s=1, alpha=0.6)

            axes[i].set_title(f'True: {classes[true_label]}\nPred: {classes[pred_label]}',
                             color=color, fontweight='bold')
            axes[i].set_xlabel('X')
            axes[i].set_ylabel('Y')
            axes[i].set_zlabel('Z')

            max_range = np.array([points[:, 0].max() - points[:, 0].min(),
                                points[:, 1].max() - points[:, 1].min(),
                                points[:, 2].max() - points[:, 2].min()]).max() / 2.0

            mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
            mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
            mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5

            axes[i].set_xlim(mid_x - max_range, mid_x + max_range)
            axes[i].set_ylim(mid_y - max_range, mid_y + max_range)
            axes[i].set_zlim(mid_z - max_range, mid_z + max_range)

        plt.suptitle('Sample Predictions (Green=Correct, Red=Incorrect)', fontsize=16)
        plt.tight_layout()
        plt.savefig('sample_predictions.png', dpi=300, bbox_inches='tight')
        plt.show()

        correct_samples = [(i, p, l) for i, (p, l) in enumerate(zip(all_predictions, all_labels)) if p == l]
        incorrect_samples = [(i, p, l) for i, (p, l) in enumerate(zip(all_predictions, all_labels)) if p != l]

        print(f"\nCorrect predictions: {len(correct_samples)}")
        print(f"Incorrect predictions: {len(incorrect_samples)}")

        if incorrect_samples:
            print("\nSome incorrect predictions:")
            for i, (idx, pred, true) in enumerate(incorrect_samples[:5]):
                print(f"Sample {idx}: Predicted {classes[pred]}, Actually {classes[true]}")

def predict_from_file(model_path, point_cloud_file, classes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _, _ = load_model(model_path, device)

    points = np.loadtxt(point_cloud_file, delimiter=',').astype(np.float32)

    if len(points) >= 1024:
        choice = np.random.choice(len(points), 1024, replace=False)
    else:
        choice = np.random.choice(len(points), 1024, replace=True)

    points = points[choice, :]
    points = points - np.mean(points, axis=0)
    dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
    points = points / dist

    points = torch.from_numpy(points).transpose(1, 0)

    pred_class, confidence = predict_single_sample(model, points, device)

    print(f"Predicted class: {classes[pred_class]}")
    print(f"Confidence: {confidence:.4f}")

    visualize_point_cloud(points, f"Predicted: {classes[pred_class]} ({confidence:.2f})")
    plt.show()

    return pred_class, confidence

if __name__ == "__main__":
    evaluate_model(
        model_path='pointnet_epoch_10.pth',
        data_dir='./ModelNet10',
        visualize_samples=True,
        num_viz_samples=6
    )