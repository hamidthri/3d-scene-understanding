import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

class Trainer:
    def __init__(self, model, train_loader, test_loader, args, classes):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.args = args
        self.classes = classes
        
        self.device = self._get_device()
        self.model = self.model.to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)
        self.criterion = nn.NLLLoss()
        
        self.train_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    def _get_device(self):
        if self.args.device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(self.args.device)
    
    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        train_pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs} - Training")
        
        for data, target in train_pbar:
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            pred, trans, trans_feat = self.model(data)
            loss = self.criterion(pred, target)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            pred_choice = pred.max(1)[1]
            correct_train += pred_choice.eq(target).sum().item()
            total_train += target.size(0)
            
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * correct_train / total_train:.2f}%'
            })
        
        train_loss = running_loss / len(self.train_loader)
        train_acc = 100. * correct_train / total_train
        
        return train_loss, train_acc
    
    def test_epoch(self, epoch):
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
    
    def save_checkpoint(self, epoch, train_loss, test_acc):
        checkpoint_path = os.path.join(self.args.checkpoint_dir, f'pointnet_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'test_acc': test_acc,
        }, checkpoint_path)
    
    def save_final_model(self):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'classes': self.classes,
            'num_points': self.args.num_points,
            'feature_transform': self.args.feature_transform
        }, 'pointnet_final.pth')
    
    def train(self):
        print(f"Training on device: {self.device}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Test samples: {len(self.test_loader.dataset)}")
        
        for epoch in range(self.args.epochs):
            train_loss, train_acc = self.train_epoch(epoch)
            test_acc = self.test_epoch(epoch)
            
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.test_accuracies.append(test_acc)
            
            print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
            
            self.scheduler.step()
            
            if (epoch + 1) % self.args.save_interval == 0:
                self.save_checkpoint(epoch, train_loss, test_acc)
        
        self.save_final_model()
        
        print(f"Final Test Accuracy: {test_acc:.2f}%")
        print("Training completed!")
        
        return self.train_losses, self.train_accuracies, self.test_accuracies
