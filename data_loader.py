from torch.utils.data import DataLoader
from datasets.modelnet10 import ModelNet10

def get_data_loaders(args):
    train_dataset = ModelNet10(args.data_dir, split='train', num_points=args.num_points)
    test_dataset = ModelNet10(args.data_dir, split='test', num_points=args.num_points)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers
    )
    
    return train_loader, test_loader, train_dataset.classes