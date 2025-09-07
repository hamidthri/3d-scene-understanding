from models.pointnet.classification import PointNetClassifier
from models.pointnetpp.classification import PointNetPlusPlusClassifier
from models.dgcnn.classification import DGCNNClassifier
from models.pointMLP.classification import PointMLPClassifier

from models.pointnet.segmentation import PointNetSegmentation
from models.pointnetpp.segmentation import PointNetPlusPlusSegmentation
from models.dgcnn.segmentation import DGCNNSegmentation
from models.pointMLP.segmentation import PointMLPSegmentation

def _norm_task(task: str) -> str:
    t = str(task).lower()
    if t in ("cls", "classification"):
        return "cls"
    if t in ("seg", "segmentation"):
        return "seg"
    return t

def create_model(model_name, num_classes, args):
    task = _norm_task(args.task)

    if task == 'cls':
        return create_classification_model(model_name, num_classes, args)
    elif task == 'seg':
        return create_segmentation_model(model_name, num_classes, args)
    else:
        raise ValueError(f"Unknown task: {args.task}")

def create_classification_model(model_name, num_classes, args):
    if model_name == 'pointnet':
        return PointNetClassifier(k=num_classes, feature_transform=args.feature_transform)
    elif model_name == 'pointnet++':
        return PointNetPlusPlusClassifier(k=num_classes)
    elif model_name == 'dgcnn':
        return DGCNNClassifier(k=num_classes)
    elif model_name == 'pointmlp':
        return PointMLPClassifier(k=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def create_segmentation_model(model_name: str, num_classes: int, args):
    name = str(model_name).lower()
    in_ch = getattr(args, "in_channels", 3)

    if name in ("pointnet",):
        return PointNetSegmentation(k=num_classes, in_channels=in_ch,
                                    feature_transform=getattr(args, "feature_transform", False))

    elif name in ("pointnet++", "pointnet2", "pointnetpp"):
        return PointNetPlusPlusSegmentation(num_classes=num_classes, in_channels=in_ch)

    elif name in ("dgcnn"):
        return DGCNNSegmentation(k=num_classes, in_channels=in_ch)
    elif name in ("pointmlp"):
        return PointMLPSegmentation(num_classes=num_classes, in_channels=in_ch)
    else:
        raise ValueError(f"Unknown segmentation model: {model_name}")

def get_model_info(model_name, task):
    """Get model information"""
    info = {
        'pointnet': {
            'classification': 'PointNet for 3D Object Classification',
            'segmentation': 'PointNet for 3D Part Segmentation'
        },
        'pointnet++': {
            'classification': 'PointNet++ for 3D Object Classification',
            'segmentation': 'PointNet++ for 3D Part Segmentation'
        },
        'dgcnn': {
            'classification': 'Dynamic Graph CNN for 3D Object Classification',
            'segmentation': 'Dynamic Graph CNN for 3D Part Segmentation'
        },
        'pointmlp': {
            'classification': 'PointMLP for 3D Object Classification',
            'segmentation': 'PointMLP for 3D Part Segmentation'
        }
    }
    
    return info.get(model_name, {}).get(task, f"{model_name} for {task}")