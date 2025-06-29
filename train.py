import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ModelNet10Dataset
from model import PointNetClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_pointnet(
    data_dir='./ModelNet10',
        batch_size=32,
        epochs=200,
        learning_rate=0.001,
        num_points=2048,
        feature_transform=True):
    
    train_dataset = ModelNet10Dataset(data_dir, split='train', num_points=num_points)
    test_dataset = ModelNet10Dataset(data_dir, split='test', num_points=num_points)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Initialize model and move to device
    model = PointNetClassifier(k=len(train_dataset.classes), feature_transform=feature_transform)
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = nn.NLLLoss()
    
    train_losses, train_accuracies, test_accuracies = [], [], []
        
    for epoch in range(epochs):
        model.train()
        running_loss=0.0
        correct_train = 0
        total_train = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epoch} - Training")
        for data, target in train_pbar:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            pred, trans, trans_feat = model(data)
            loss = criterion(pred, target)
            
            # if feature_transform and trans_feat is not None:
            #     loss += feature_transform_regularizer(trans_feat) * 0.001
                
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pred_choice = pred.max(1)[1]
            correct_train = pred_choice.eq(target).sum().item()
            total_train += target.size(0)
            
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * correct_train / total_train:.2f}%'
            })
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Test
        model.eval()
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            test_pbar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{epochs} - Testing')
            for data, target in test_pbar:
                data, target = data.to(device), target.to(device)
                pred, _, _ = model(data)
                pred_choice = pred.max(1)[1]
                correct_test += pred_choice.eq(target).sum().item()
                total_test += target.size(0)

                test_pbar.set_postfix({
                    'Acc': f'{100. * correct_test / total_test:.2f}%'
                })

        test_acc = 100. * correct_test / total_test
        test_accuracies.append(test_acc)

        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'test_acc': test_acc,
            }, f'pointnet_epoch_{epoch+1}.pth')

    # Save final model and metadata
    torch.save({
        'model_state_dict': model.state_dict(),
        'classes': train_dataset.classes,
        'num_points': num_points,
        'feature_transform': feature_transform
    }, 'pointnet_final.pth')
        

if __name__ == "__main__":
    model, classes = train_pointnet(
        data_dir='./ModelNet10',
        batch_size=32,
        epochs=200,
        learning_rate=0.001,
        num_points=2048,
        feature_transform=True
    )
