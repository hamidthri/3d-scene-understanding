#!/usr/bin/env python3

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

from config import get_args
from data_loader import get_data_loaders
from model_factory import create_model, get_model_info


def get_evaluation_args():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate saved checkpoint')
    
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the saved checkpoint (.pth file)')
    
    parser.add_argument('--task', type=str, default=None,
                        choices=['classification', 'segmentation', 'cls', 'seg'],
                        help='Task type (auto-detected from checkpoint if not provided)')
    parser.add_argument('--model', type=str, default=None,
                        choices=['pointnet', 'pointnet++', 'dgcnn', 'pointmlp'],
                        help='Model architecture (auto-detected from checkpoint if not provided)')
    
    parser.add_argument('--data_dir', type=str, default='./ModelNet10',
                        help='Path to dataset directory')
    parser.add_argument('--seg_data_dir', type=str, default='./ShapeNetPart',
                        help='Path to segmentation dataset directory')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers for data loading')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu', 'mps'],
                        help='Device to use')
    
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed results')
    parser.add_argument('--save_results', action='store_true',
                        help='Save results to CSV file')
    parser.add_argument('--visualize', action='store_true',
                        help='Create visualization plots')
    
    return parser.parse_args()


def setup_device(device_arg):
    if device_arg == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device_arg)
    
    return device


def load_checkpoint_and_extract_args(checkpoint_path, device, args):
    print(f"Loading checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'args' in checkpoint:
        if isinstance(checkpoint['args'], dict):
            class ArgsNamespace:
                def __init__(self, **kwargs):
                    for k, v in kwargs.items():
                        setattr(self, k, v)
            
            saved_args = ArgsNamespace(**checkpoint['args'])
        else:
            saved_args = checkpoint['args']
        
        if args.task is None:
            args.task = getattr(saved_args, 'task', 'classification')
        if args.model is None:
            args.model = getattr(saved_args, 'model', 'pointnet')
        
        args.num_points = getattr(saved_args, 'num_points', 1024)
        args.feature_transform = getattr(saved_args, 'feature_transform', False)
        args.normal = getattr(saved_args, 'normal', False)
        args.category = getattr(saved_args, 'category', None)
        
    else:
        print("[Warning] No 'args' found in checkpoint, using defaults/command-line values")
        if args.task is None:
            args.task = 'classification'
        if args.model is None:
            args.model = 'pointnet'
        args.num_points = 1024
        args.feature_transform = False
        args.normal = False
        args.category = None
    
    if args.task in ['cls']:
        args.task = 'classification'
    elif args.task in ['seg']:
        args.task = 'segmentation'
    
    print(f"Checkpoint from epoch: {checkpoint.get('epoch', 'Unknown')}")
    print(f"Task: {args.task}")
    print(f"Model: {args.model}")
    
    return checkpoint, args


def evaluate_classification_model(model, test_loader, device, classes, args, verbose=False):
    model.eval()
    all_predictions = []
    all_labels = []
    all_data = []
    correct = 0
    total = 0
    
    print("\nEvaluating classification model...")
    start_time = time.time()
    
    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc="Testing")
        for batch_idx, (data, labels) in enumerate(test_pbar):
            data, labels = data.to(device), labels.to(device)
            
            outputs = model(data)
            if isinstance(outputs, tuple):
                pred_logits = outputs[0]
            else:
                pred_logits = outputs
            
            predictions = pred_logits.max(1)[1]
            
            correct += predictions.eq(labels).sum().item()
            total += labels.size(0)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_data.extend(data.cpu().numpy())
            
            current_acc = 100.0 * correct / total
            test_pbar.set_postfix({'Acc': f'{current_acc:.2f}%'})
    
    eval_time = time.time() - start_time
    accuracy = 100.0 * correct / total
    
    print(f"\n{'='*60}")
    print(f"CLASSIFICATION RESULTS")
    print(f"{'='*60}")
    print(f"Test Accuracy: {accuracy:.4f}% ({correct}/{total})")
    print(f"Evaluation Time: {eval_time:.2f} seconds")
    print(f"{'='*60}")
    
    if verbose and len(classes) > 0:
        print("\nDetailed Classification Report:")
        try:
            target_names = [str(cls) for cls in classes]
            report = classification_report(all_labels, all_predictions, 
                                         target_names=target_names, 
                                         zero_division=0)
            print(report)
        except Exception as e:
            print(f"Could not generate detailed report: {e}")
    
    if args.visualize:
        from utils.visualization import visualize_classification_predictions
        visualize_classification_predictions(
            all_data[:10], all_labels[:10], all_predictions[:10], 
            classes, args.model, 
            save_path=os.path.join(os.path.dirname(args.checkpoint_path), 
                                 f'{args.model}_cls_predictions.png')
        )
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'eval_time': eval_time,
        'predictions': all_predictions,
        'labels': all_labels
    }


def evaluate_segmentation_model(model, test_loader, device, num_parts, args, verbose=False):
    model.eval()
    correct_points = 0
    total_points = 0
    total_per_cat_iou = []
    all_data = []
    all_seg_labels = []
    all_predictions = []
    
    print("\nEvaluating segmentation model...")
    start_time = time.time()
    
    def ensure_bn_layout(seg_pred, num_parts):
        if seg_pred.dim() != 3:
            raise RuntimeError(f"seg_pred should be 3D, got shape {tuple(seg_pred.shape)}")
        B, D1, D2 = seg_pred.shape
        if D1 == num_parts and D2 != num_parts:
            seg_pred = seg_pred.transpose(1, 2).contiguous()
        return seg_pred
    
    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc="Testing")
        for batch_idx, (data, seg_labels, cat_labels) in enumerate(test_pbar):
            data = data.to(device)
            seg_labels = seg_labels.to(device)
            cat_labels = cat_labels.to(device)
            
            outputs = model(data, cat_labels)
            if isinstance(outputs, tuple):
                seg_pred = outputs[0]
            else:
                seg_pred = outputs
            
            seg_pred = ensure_bn_layout(seg_pred, num_parts)
            pred_choice = seg_pred.argmax(dim=-1)
            
            correct_points += pred_choice.eq(seg_labels).sum().item()
            total_points += seg_labels.numel()
            
            if len(all_data) < 10:
                all_data.extend(data.cpu().numpy())
                all_seg_labels.extend(seg_labels.cpu().numpy())
                all_predictions.extend(pred_choice.cpu().numpy())
            
            B, N = pred_choice.shape
            for i in range(B):
                segp = pred_choice[i].detach().cpu().numpy()
                segl = seg_labels[i].detach().cpu().numpy()
                parts = np.unique(segl)
                if parts.size == 0:
                    continue
                
                ious = []
                for p in parts:
                    if p < 0:
                        continue
                    intersection = np.sum((segp == p) & (segl == p))
                    union = np.sum((segp == p) | (segl == p))
                    if union == 0:
                        ious.append(1.0)
                    else:
                        ious.append(intersection / union)
                
                if ious:
                    total_per_cat_iou.append(float(np.mean(ious)))
            
            current_acc = 100.0 * correct_points / total_points
            current_miou = (np.mean(total_per_cat_iou) * 100) if total_per_cat_iou else 0.0
            test_pbar.set_postfix({
                'Acc': f'{current_acc:.2f}%',
                'mIoU': f'{current_miou:.2f}%'
            })
    
    eval_time = time.time() - start_time
    accuracy = 100.0 * correct_points / total_points
    mean_iou = (np.mean(total_per_cat_iou) * 100) if total_per_cat_iou else 0.0
    
    print(f"\n{'='*60}")
    print(f"SEGMENTATION RESULTS")
    print(f"{'='*60}")
    print(f"Test Accuracy: {accuracy:.4f}% ({correct_points}/{total_points})")
    print(f"Mean IoU: {mean_iou:.4f}%")
    print(f"Evaluation Time: {eval_time:.2f} seconds")
    print(f"Number of shapes evaluated: {len(total_per_cat_iou)}")
    print(f"{'='*60}")
    
    if args.visualize:
        from utils.visualization import visualize_segmentation_predictions
        visualize_segmentation_predictions(
            all_data[:5], all_seg_labels[:5], all_predictions[:5],
            args.model,
            save_path=os.path.join(os.path.dirname(args.checkpoint_path), 
                                 f'{args.model}_seg_predictions.png')
        )
    
    return {
        'accuracy': accuracy,
        'mean_iou': mean_iou,
        'correct_points': correct_points,
        'total_points': total_points,
        'eval_time': eval_time,
        'shape_ious': total_per_cat_iou
    }


def save_results_to_csv(results, checkpoint_path, args):
    import csv
    from datetime import datetime
    
    results_file = os.path.join(os.path.dirname(checkpoint_path), 'evaluation_results.csv')
    
    file_exists = os.path.exists(results_file)
    
    with open(results_file, 'a', newline='') as csvfile:
        if args.task == 'classification':
            fieldnames = ['timestamp', 'checkpoint', 'model', 'task', 'accuracy', 'eval_time']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'checkpoint': os.path.basename(checkpoint_path),
                'model': args.model,
                'task': args.task,
                'accuracy': f"{results['accuracy']:.4f}",
                'eval_time': f"{results['eval_time']:.2f}"
            })
        else:
            fieldnames = ['timestamp', 'checkpoint', 'model', 'task', 'accuracy', 'mean_iou', 'eval_time']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'checkpoint': os.path.basename(checkpoint_path),
                'model': args.model,
                'task': args.task,
                'accuracy': f"{results['accuracy']:.4f}",
                'mean_iou': f"{results['mean_iou']:.4f}",
                'eval_time': f"{results['eval_time']:.2f}"
            })
    
    print(f"Results saved to: {results_file}")


def main():
    args = get_evaluation_args()
    device = setup_device(args.device)
    
    print(f"{'='*60}")
    print(f"CHECKPOINT EVALUATION")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint_path}")
    
    checkpoint, args = load_checkpoint_and_extract_args(args.checkpoint_path, device, args)
    
    if args.task == 'segmentation':
        args.data_dir = args.seg_data_dir
        args.in_channels = 6 if args.normal else 3
    else:
        args.in_channels = 3
    
    print(f"Data directory: {args.data_dir}")
    print(f"{'='*60}")
    
    print("Loading dataset...")
    try:
        args.epochs = 1
        args.save_interval = 1
        args.checkpoint_dir = os.path.dirname(args.checkpoint_path)
        
        _, test_loader, meta = get_data_loaders(args)
        print(f"Test samples: {len(test_loader.dataset)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    if args.task == 'segmentation':
        if 'num_parts' not in meta:
            raise ValueError(f"Segmentation requires 'num_parts' in meta, got: {meta}")
        k = meta['num_parts']
    else:
        if 'num_classes' not in meta:
            raise ValueError(f"Classification requires 'num_classes' in meta, got: {meta}")
        k = meta['num_classes']
    
    print(f"Number of classes/parts: {k}")
    
    print("Creating model...")
    try:
        model = create_model(args.model, k, args)
        model.to(device)
    except Exception as e:
        print(f"Error creating model: {e}")
        return
    
    print("Loading model weights...")
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model weights loaded successfully")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return
    
    if args.task == 'classification':
        classes = meta.get('classes', [])
        results = evaluate_classification_model(
            model, test_loader, device, classes, args, args.verbose
        )
    else:
        num_parts = meta['num_parts']
        results = evaluate_segmentation_model(
            model, test_loader, device, num_parts, args, args.verbose
        )
    
    if args.save_results:
        save_results_to_csv(results, args.checkpoint_path, args)
    
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()