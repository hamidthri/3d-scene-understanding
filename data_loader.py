from torch.utils.data import DataLoader
from datasets.modelnet10 import ModelNet10
from datasets.ShapeNetPart import ShapeNetPart
import torch

def _norm_task(task: str) -> str:
    t = str(task).lower()
    if t in ("cls", "classification"):
        return "cls"
    if t in ("seg", "segmentation"):
        return "seg"
    return t

def get_data_loaders(args):
    """
    Returns:
        train_loader, test_loader, meta
    Where meta is a dict:
        Classification: {'classes': [...], 'num_classes': int}
        Segmentation:   {'num_parts': int, 'label_names': dict (optional)}
    """
    task = _norm_task(args.task)

    if task == 'cls':
        train_dataset = ModelNet10(args.data_dir, split='train', num_points=args.num_points)
        test_dataset  = ModelNet10(args.data_dir, split='test',  num_points=args.num_points)
        meta = {'classes': train_dataset.classes, 'num_classes': len(train_dataset.classes)}

    elif task == 'seg':
        root = getattr(args, "seg_data_dir", None) or args.data_dir
        use_normals = getattr(args, "normal", False)
        category = getattr(args, "category", None)

        train_dataset = ShapeNetPart(root, split='train', num_points=args.num_points,
                                     normal=use_normals, category=category)
        test_dataset  = ShapeNetPart(root, split='test',  num_points=args.num_points,
                                     normal=use_normals, category=category)

        num_parts = getattr(train_dataset, "num_parts", None)
        if num_parts is None:
            raise ValueError("ShapeNetPart dataset must define `num_parts`.")
        meta = {
            'num_parts': num_parts,
            'label_names': getattr(train_dataset, "label_names", None)
        }

    else:
        raise ValueError(f"Unknown task: {args.task}")

    pin_mem = False
    try:
        pin_mem = torch.cuda.is_available()
    except Exception:
        pass

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=pin_mem,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
    )
    return train_loader, test_loader, meta
