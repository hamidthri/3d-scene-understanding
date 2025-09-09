# train.py
import torch
from config import get_args
from data_loader import get_data_loaders
from model_factory import create_model
from trainer import Trainer
from seg_trainer import SegmentationTrainer
from utils.visualization import plot_training_curves
from model_factory import get_model_info

def _norm_task(task: str) -> str:
    t = str(task).lower()
    if t in ("cls", "classification"):
        return "cls"
    if t in ("seg", "segmentation"):
        return "seg"
    return t


def _maybe_route_seg_data_dir(args):
    """
    If the user passed --seg_data_dir for segmentation, use it as data_dir.
    This keeps backward compatibility with older CLIs that use --data_dir.
    """
    task = _norm_task(args.task)
    if task == "seg" and hasattr(args, "seg_data_dir"):
        seg_dir = getattr(args, "seg_data_dir")
        if seg_dir:  # non-empty string
            args.data_dir = seg_dir


def select_available_device(args):
    """
    Normalize args.device to an actually available backend.
    Sets args.device to one of: 'cuda', 'mps', or 'cpu'.
    """
    requested = str(getattr(args, "device", "auto")).lower()
    mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    if requested in ("auto", ""):
        args.device = "cuda" if torch.cuda.is_available() else ("mps" if mps_ok else "cpu")
        return

    if requested == "cuda" and not torch.cuda.is_available():
        print("[Warn] CUDA requested but not available/compiled. Falling back to CPU.")
        args.device = "cpu"
        return

    if requested == "mps" and not mps_ok:
        print("[Warn] MPS requested but not available. Falling back to CPU.")
        args.device = "cpu"
        return
    args.device = requested


def main():
    args = get_args()
    task = _norm_task(args.task)

    # Prefer --seg_data_dir when doing segmentation
    _maybe_route_seg_data_dir(args)

    # Pick a usable device based on availability and user request
    select_available_device(args)

    print("=" * 60)
    print("3D Point Cloud Processing Pipeline")
    print("=" * 60)
    print(f"Task: {'Segmentation' if task=='seg' else 'Classification'}")
    try:
        info = get_model_info(args.model, 'segmentation' if task == 'seg' else 'classification')
        print(f"Model: {info}")
    except Exception:
        title = "PointNet for 3D Part Segmentation" if task == 'seg' else "PointNet for 3D Object Classification"
        print(f"Model: {title if args.model=='pointnet' else args.model}")
    print(f"Device: {args.device}")
    print("=" * 60)
    print("Loading dataset...")

    train_loader, test_loader, meta = get_data_loaders(args)

    # Determine k (num classes or num parts) from dataset metadata
    if task == "seg":
        if "num_parts" not in meta:
            raise ValueError(f"Segmentation task requires 'num_parts' in meta, got: {meta}")
        k = meta["num_parts"]
    else:
        if "num_classes" not in meta:
            raise ValueError(f"Classification task requires 'num_classes' in meta, got: {meta}")
        k = meta["num_classes"]

    if not isinstance(k, int) or k <= 0:
        raise ValueError(f"Invalid k detected (classes/parts): {k}. Meta: {meta}")

    # Input channels (xyz vs xyz+normals)
    if task == "seg":
        args.in_channels = 6 if getattr(args, "normal", False) else 3
    else:
        args.in_channels = 3

    print("=" * 60)
    print("Creating model...")
    model = create_model(args.model, k, args)

    # Train
    if task == "cls":
        classes = meta.get('classes', [])
        trainer = Trainer(model, train_loader, test_loader, args, classes)
        train_losses, train_accuracies, test_accuracies = trainer.train()
        try:
            plot_training_curves(train_losses, train_accuracies, test_accuracies)
        except Exception as e:
            print(f"[Warn] Plotting failed: {e}")
    else:
        num_parts = meta["num_parts"]
        part_names = meta.get("label_names", None)
        trainer = SegmentationTrainer(model, train_loader, test_loader, args, num_parts, part_names)
        # returns: train_losses, train_accs, test_accs, test_mious
        results = trainer.train()
        try:
            tr_loss, tr_acc, te_acc, te_miou = (
                results[0][-1], results[1][-1], results[2][-1], results[3][-1]
            )
            print("=" * 60)
            print(f"[Final] Train Loss: {tr_loss:.4f} | "
                  f"Train Acc: {tr_acc:.2f}% | "
                  f"Test Acc: {te_acc:.2f}% | Test mIoU: {te_miou:.3f}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
