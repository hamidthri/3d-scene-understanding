import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Point Cloud Training Pipeline')

    # Task selection
    parser.add_argument('--task', type=str, default='classification',
                        choices=['classification', 'segmentation', 'cls', 'seg'],
                        help='Task to perform: classification or segmentation')

    # Model architecture
    parser.add_argument('--model', type=str, default='pointnet',
                        choices=['pointnet', 'pointnet++', 'dgcnn', 'pointmlp'],
                        help='Model architecture to use')

    # Data paths
    parser.add_argument('--data_dir', type=str, default='./ModelNet10',
                        help='Path to dataset directory')
    parser.add_argument('--seg_data_dir', type=str, default='./ShapeNetPart',
                        help='Path to segmentation dataset directory')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for optimizer')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='Number of points to sample from each point cloud')

    # Model specific parameters
    parser.add_argument('--feature_transform', action='store_true', default=False,
                        help='Use feature transform in PointNet (set this flag to enable)')
    parser.add_argument('--num_classes', type=int, default=None,
                        help='Number of classes (auto-detected if not specified)')
    parser.add_argument('--num_parts', type=int, default=None,
                        help='Number of parts for segmentation (auto-detected if not specified)')

    # Checkpoint and saving
    parser.add_argument('--checkpoint_dir', type=str, default='assets/checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Save checkpoint every N epochs')

    # System parameters
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers for data loading')
    parser.add_argument('--device', type=str, default='auto',
                    choices=['auto', 'cuda', 'cpu', 'mps'],
                    help='Device to use for training')

    # Segmentation specific parameters
    parser.add_argument('--category', type=str, default=None,
                        help='Specific category for segmentation (name or synset), else all')
    parser.add_argument('--normal', action='store_true', default=False,
                        help='Use normal information for segmentation (xyz+normals -> 6 channels)')

    # Add --use_tensorboard
    parser.add_argument('--use_tensorboard', action='store_true', default=False,
                        help='Use TensorBoard for logging')

    args = parser.parse_args()

    # Normalize task names
    if args.task in ['cls']:
        args.task = 'classification'
    elif args.task in ['seg']:
        args.task = 'segmentation'

    return args
