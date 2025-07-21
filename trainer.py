import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
from models.enhance_pp.classification import ContrastiveLoss
from models.enhance_pp.segmentation import SegmentationContrastiveLoss


class Trainer:
    def __init__(self, model, train_loader, test_loader, args, classes, task_type='classification'):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.args = args
        self.classes = classes
        self.task_type = task_type
        
        self.device = self._get_device()
        self.model = self.model.to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)
        
        if task_type == 'classification':
            self.criterion = ContrastiveLoss(
                classification_weight=1.0,
                contrastive_weight=getattr(args, 'contrastive_weight', 0.1),
                consistency_weight=getattr(args, 'consistency_weight', 0.05)
            )
        else:
            self.criterion = SegmentationContrastiveLoss(
                classification_weight=1.0,
                contrastive_weight=getattr(args, 'contrastive_weight', 0.1),
                consistency_weight=getattr(args, 'consistency_weight', 0.05)
            )
        
        self.train_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        self.contrastive_losses = []
        self.consistency_losses = []
        
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    def _get_device(self):
        if self.args.device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(self.args.device)
    
    def train_epoch_classification(self, epoch):
        self.model.train()
        running_loss = 0.0
        running_contrastive_loss = 0.0
        running_consistency_loss = 0.0
        correct_train = 0
        total_train = 0
        
        train_pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs} - Training")
        
        for data, target in train_pbar:
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            if hasattr(self.model, 'use_contrastive') and self.model.use_contrastive:
                pred, contrastive_loss, consistency_loss = self.model(data, target)
                total_loss, classification_loss, cont_loss_val, cons_loss_val = self.criterion(
                    pred, target, contrastive_loss, consistency_loss
                )
            else:
                pred, _, _ = self.model(data)
                total_loss, classification_loss, cont_loss_val, cons_loss_val = self.criterion(pred, target)
            
            total_loss.backward()
            self.optimizer.step()
            
            running_loss += total_loss.item()
            if cont_loss_val is not None:
                running_contrastive_loss += cont_loss_val.item()
            if cons_loss_val is not None:
                running_consistency_loss += cons_loss_val.item()
            
            pred_choice = pred.max(1)[1]
            correct_train += pred_choice.eq(target).sum().item()
            total_train += target.size(0)
            
            train_pbar.set_postfix({
                'Loss': f'{total_loss.item():.4f}',
                'Acc': f'{100. * correct_train / total_train:.2f}%',
                'ContLoss': f'{cont_loss_val.item() if cont_loss_val is not None else 0:.4f}',
                'ConsLoss': f'{cons_loss_val.item() if cons_loss_val is not None else 0:.4f}'
            })
        
        train_loss = running_loss / len(self.train_loader)
        train_acc = 100. * correct_train / total_train
        avg_contrastive_loss = running_contrastive_loss / len(self.train_loader)
        avg_consistency_loss = running_consistency_loss / len(self.train_loader)
        
        return train_loss, train_acc, avg_contrastive_loss, avg_consistency_loss
    
    def train_epoch_segmentation(self, epoch):
        self.model.train()
        running_loss = 0.0
        running_contrastive_loss = 0.0
        running_consistency_loss = 0.0
        correct_train = 0
        total_train = 0
        
        train_pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs} - Training")
        
        for data, target in train_pbar:
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            if hasattr(self.model, 'use_contrastive') and self.model.use_contrastive:
                pred, contrastive_loss, consistency_loss = self.model(data, target)
                total_loss, classification_loss, cont_loss_val, cons_loss_val = self.criterion(
                    pred, target, contrastive_loss, consistency_loss
                )
            else:
                pred, _, _ = self.model(data)
                total_loss, classification_loss, cont_loss_val, cons_loss_val = self.criterion(pred, target)
            
            total_loss.backward()
            self.optimizer.step()
            
            running_loss += total_loss.item()
            if cont_loss_val is not None:
                running_contrastive_loss += cont_loss_val.item()
            if cons_loss_val is not None:
                running_consistency_loss += cons_loss_val.item()
            
            pred_choice = pred.contiguous().view(-1, pred.size(-1)).max(1)[1]
            target_flat = target.contiguous().view(-1)
            correct_train += pred_choice.eq(target_flat).sum().item()
            total_train += target_flat.size(0)
            
            train_pbar.set_postfix({
                'Loss': f'{total_loss.item():.4f}',
                'Acc': f'{100. * correct_train / total_train:.2f}%',
                'ContLoss': f'{cont_loss_val.item() if cont_loss_val is not None else 0:.4f}',
                'ConsLoss': f'{cons_loss_val.item() if cons_loss_val is not None else 0:.4f}'
            })
        
        train_loss = running_loss / len(self.train_loader)
        train_acc = 100. * correct_train / total_train
        avg_contrastive_loss = running_contrastive_loss / len(self.train_loader)
        avg_consistency_loss = running_consistency_loss / len(self.train_loader)
        
        return train_loss, train_acc, avg_contrastive_loss, avg_consistency_loss
    
    def test_epoch_classification(self, epoch):
        self.model.eval()
        correct_test = 0
        total_test = 0
        
        with torch.no_grad():
            test_pbar = tqdm(self.test_loader, desc=f'Epoch {epoch+1}/{self.args.epochs} - Testing')
            
            for data, target in test_pbar:
                data, target = data.to(self.device), target.to(self.device)
                pred, _, _ = self.model(data)
                pred_choice = pred.max(1)[1]
                correct_test += pred_choice.eq(target).sum().item()
                total_test += target.size(0)
                
                test_pbar.set_postfix({
                    'Acc': f'{100. * correct_test / total_test:.2f}%'
                })
        
        test_acc = 100. * correct_test / total_test
        return test_acc
    
    def test_epoch_segmentation(self, epoch):
        self.model.eval()
        correct_test = 0
        total_test = 0
        
        with torch.no_grad():
            test_pbar = tqdm(self.test_loader, desc=f'Epoch {epoch+1}/{self.args.epochs} - Testing')
            
            for data, target in test_pbar:
                data, target = data.to(self.device), target.to(self.device)
                pred, _, _ = self.model(data)
                
                pred_choice = pred.contiguous().view(-1, pred.size(-1)).max(1)[1]
                target_flat = target.contiguous().view(-1)
                correct_test += pred_choice.eq(target_flat).sum().item()
                total_test += target_flat.size(0)
                
                test_pbar.set_postfix({
                    'Acc': f'{100. * correct_test / total_test:.2f}%'
                })
        
        test_acc = 100. * correct_test / total_test
        return test_acc
    
    def save_checkpoint(self, epoch, train_loss, test_acc, contrastive_loss, consistency_loss):
        checkpoint_path = os.path.join(self.args.checkpoint_dir, f'enhanced_pointnet_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'test_acc': test_acc,
            'contrastive_loss': contrastive_loss,
            'consistency_loss': consistency_loss,
        }, checkpoint_path)
    
    def save_final_model(self):
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'classes': self.classes,
            'task_type': self.task_type,
        }
        
        if hasattr(self.args, 'num_points'):
            save_dict['num_points'] = self.args.num_points
        if hasattr(self.args, 'feature_transform'):
            save_dict['feature_transform'] = self.args.feature_transform
        if hasattr(self.model, 'use_contrastive'):
            save_dict['use_contrastive'] = self.model.use_contrastive
            
        torch.save(save_dict, f'enhanced_pointnet_{self.task_type}_final.pth')
    
    def train(self):
        print(f"Training on device: {self.device}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Test samples: {len(self.test_loader.dataset)}")
        print(f"Task type: {self.task_type}")
        
        if hasattr(self.model, 'use_contrastive'):
            print(f"Using contrastive sampling: {self.model.use_contrastive}")
        
        for epoch in range(self.args.epochs):
            if self.task_type == 'classification':
                train_loss, train_acc, contrastive_loss, consistency_loss = self.train_epoch_classification(epoch)
                test_acc = self.test_epoch_classification(epoch)
            else:
                train_loss, train_acc, contrastive_loss, consistency_loss = self.train_epoch_segmentation(epoch)
                test_acc = self.test_epoch_segmentation(epoch)
            
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.test_accuracies.append(test_acc)
            self.contrastive_losses.append(contrastive_loss)
            self.consistency_losses.append(consistency_loss)
            
            print(f'Epoch {epoch+1}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Test Acc: {test_acc:.2f}%')
            print(f'  Contrastive Loss: {contrastive_loss:.4f}')
            print(f'  Consistency Loss: {consistency_loss:.4f}')
            
            self.scheduler.step()
            
            if (epoch + 1) % self.args.save_interval == 0:
                self.save_checkpoint(epoch, train_loss, test_acc, contrastive_loss, consistency_loss)
        
        self.save_final_model()
        
        print(f"Final Test Accuracy: {test_acc:.2f}%")
        print("Training completed!")
        
        return {
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies, 
            'test_accuracies': self.test_accuracies,
            'contrastive_losses': self.contrastive_losses,
            'consistency_losses': self.consistency_losses
        }