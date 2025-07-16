import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Point Cloud Classification Training')
    
    parser.add_argument('--model', type=str, default='pointnet', 
                       choices=['pointnet', 'pointnet++', 'dgcnn'],
                       help='Model architecture to use')
    parser.add_argument('--data_dir', type=str, default='./ModelNet10',
                       help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate for optimizer')
    parser.add_argument('--num_points', type=int, default=2048,
                       help='Number of points to sample from each point cloud')
    parser.add_argument('--feature_transform', action='store_true', default=True,
                       help='Use feature transform in PointNet')
    parser.add_argument('--checkpoint_dir', type=str, default='assets/checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--save_interval', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of workers for data loading')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use for training')
    
    return parser.parse_args()