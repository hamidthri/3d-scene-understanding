import torch
import torch.nn as nn
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import json
import os

from models.pointnetpp.classification import PointNetPlusPlusClassifier
from models.enhance_pp.classification import EnhancedPointNetPlusPlusClassifier
from models.pointnetpp.segmentation import PointNetPlusPlusSegmentation
from models.enhance_pp.segmentation import EnhancedPointNetPlusPlusSegmentation
from trainer import Trainer
from data_loader import get_data_loaders


class ExperimentRunner:
    def __init__(self, args):
        self.args = args
        self.results = {}
        
    def run_classification_experiment(self, train_loader, test_loader, classes):
        print("="*50)
        print("Running Classification Experiments")
        print("="*50)
        
        baseline_model = PointNetPlusPlusClassifier(k=len(classes))
        enhanced_model = EnhancedPointNetPlusPlusClassifier(k=len(classes), use_contrastive=True)
        
        print("\n1. Training Baseline PointNet++...")
        baseline_trainer = Trainer(baseline_model, train_loader, test_loader, self.args, classes, 'classification')
        baseline_results = baseline_trainer.train()
        
        print("\n2. Training Enhanced PointNet++ with Contrastive Sampling...")
        enhanced_trainer = Trainer(enhanced_model, train_loader, test_loader, self.args, classes, 'classification')
        enhanced_results = enhanced_trainer.train()
        
        self.results['classification'] = {
            'baseline': baseline_results,
            'enhanced': enhanced_results
        }
        
        self.plot_classification_results()
        
    def run_segmentation_experiment(self, train_loader, test_loader, num_classes):
        print("="*50)
        print("Running Segmentation Experiments")
        print("="*50)
        
        baseline_model = PointNetPlusPlusSegmentation(num_classes=num_classes)
        enhanced_model = EnhancedPointNetPlusPlusSegmentation(num_classes=num_classes, use_contrastive=True)
        
        print("\n1. Training Baseline PointNet++...")
        baseline_trainer = EnhancedTrainer(baseline_model, train_loader, test_loader, self.args, list(range(num_classes)), 'segmentation')
        baseline_results = baseline_trainer.train()
        
        print("\n2. Training Enhanced PointNet++ with Contrastive Sampling...")
        enhanced_trainer = EnhancedTrainer(enhanced_model, train_loader, test_loader, self.args, list(range(num_classes)), 'segmentation')
        enhanced_results = enhanced_trainer.train()
        
        self.results['segmentation'] = {
            'baseline': baseline_results,
            'enhanced': enhanced_results
        }
        
        self.plot_segmentation_results()
    
    def plot_classification_results(self):
        baseline = self.results['classification']['baseline']
        enhanced = self.results['classification']['enhanced']
        
        epochs = range(1, len(baseline['train_accuracies']) + 1)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        axes[0, 0].plot(epochs, baseline['train_accuracies'], label='Baseline Train', color='blue', linestyle='-')
        axes[0, 0].plot(epochs, enhanced['train_accuracies'], label='Enhanced Train', color='red', linestyle='-')
        axes[0, 0].plot(epochs, baseline['test_accuracies'], label='Baseline Test', color='blue', linestyle='--')
        axes[0, 0].plot(epochs, enhanced['test_accuracies'], label='Enhanced Test', color='red', linestyle='--')
        axes[0, 0].set_title('Classification Accuracy Comparison')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy (%)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(epochs, baseline['train_losses'], label='Baseline', color='blue')
        axes[0, 1].plot(epochs, enhanced['train_losses'], label='Enhanced', color='red')
        axes[0, 1].set_title('Training Loss Comparison')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        axes[1, 0].plot(epochs, enhanced['contrastive_losses'], label='Contrastive Loss', color='green')
        axes[1, 0].set_title('Contrastive Loss Over Time')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Contrastive Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        axes[1, 1].plot(epochs, enhanced['consistency_losses'], label='Consistency Loss', color='orange')
        axes[1, 1].set_title('Consistency Loss Over Time')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Consistency Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('classification_experiment_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        final_baseline_acc = baseline['test_accuracies'][-1]
        final_enhanced_acc = enhanced['test_accuracies'][-1]
        improvement = final_enhanced_acc - final_baseline_acc
        
        print(f"\nClassification Results Summary:")
        print(f"Baseline Final Test Accuracy: {final_baseline_acc:.2f}%")
        print(f"Enhanced Final Test Accuracy: {final_enhanced_acc:.2f}%")
        print(f"Improvement: {improvement:.2f}%")
        
    def plot_segmentation_results(self):
        baseline = self.results['segmentation']['baseline']
        enhanced = self.results['segmentation']['enhanced']
        
        epochs = range(1, len(baseline['train_accuracies']) + 1)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        axes[0, 0].plot(epochs, baseline['train_accuracies'], label='Baseline Train', color='blue', linestyle='-')
        axes[0, 0].plot(epochs, enhanced['train_accuracies'], label='Enhanced Train', color='red', linestyle='-')
        axes[0, 0].plot(epochs, baseline['test_accuracies'], label='Baseline Test', color='blue', linestyle='--')
        axes[0, 0].plot(epochs, enhanced['test_accuracies'], label='Enhanced Test', color='red', linestyle='--')
        axes[0, 0].set_title('Segmentation Accuracy Comparison')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy (%)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(epochs, baseline['train_losses'], label='Baseline', color='blue')
        axes[0, 1].plot(epochs, enhanced['train_losses'], label='Enhanced', color='red')
        axes[0, 1].set_title('Training Loss Comparison')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        axes[1, 0].plot(epochs, enhanced['contrastive_losses'], label='Contrastive Loss', color='green')
        axes[1, 0].set_title('Contrastive Loss Over Time')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Contrastive Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        axes[1, 1].plot(epochs, enhanced['consistency_losses'], label='Consistency Loss', color='orange')
        axes[1, 1].set_title('Consistency Loss Over Time')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Consistency Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('segmentation_experiment_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        final_baseline_acc = baseline['test_accuracies'][-1]
        final_enhanced_acc = enhanced['test_accuracies'][-1]
        improvement = final_enhanced_acc - final_baseline_acc
        
        print(f"\nSegmentation Results Summary:")
        print(f"Baseline Final Test Accuracy: {final_baseline_acc:.2f}%")
        print(f"Enhanced Final Test Accuracy: {final_enhanced_acc:.2f}%")
        print(f"Improvement: {improvement:.2f}%")
    
    def save_results(self):
        os.makedirs('experiment_results', exist_ok=True)
        
        with open('experiment_results/results.json', 'w') as f:
            json.dump(self.results, f, indent=4)
        
        print("Results saved to experiment_results/results.json")


def create_experiment_args():
    parser = argparse.ArgumentParser(description='PointNet++ Contrastive Sampling Experiments')
    
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_points', type=int, default=1024, help='Number of points in point cloud')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto/cuda/cpu)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--save_interval', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--contrastive_weight', type=float, default=0.1, help='Weight for contrastive loss')
    parser.add_argument('--consistency_weight', type=float, default=0.05, help='Weight for consistency loss')
    parser.add_argument('--feature_transform', action='store_true', help='Use feature transform')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker threads for DataLoader')
    parser.add_argument('--data_dir', type=str, default='ModelNet10', help='Path to dataset directory')  # ✅ FIX
    parser.add_argument('--task', type=str, choices=['classification', 'segmentation'], required=True, help='Task to run: classification or segmentation')

    
    return parser.parse_args()




if __name__ == "__main__":
    args = create_experiment_args()

    print("Contrastive Sampling with Semantic Consistency Experiments")
    print("=" * 60)

    experiment_runner = ExperimentRunner(args)

    # Load Data
    train_loader, test_loader, class_or_label_info = get_data_loaders(args)

    if args.task == 'classification':
        experiment_runner.run_classification_experiment(train_loader, test_loader, class_or_label_info)
    else:
        experiment_runner.run_segmentation_experiment(train_loader, test_loader, num_classes=len(class_or_label_info))

    experiment_runner.save_results()
