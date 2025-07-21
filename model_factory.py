from models.pointnet.classification import PointNetClassifier
from models.pointnetpp.classification import PointNetPlusPlusClassifier
from models.dgcnn.classification import DGCNNClassifier

def create_model(model_name, num_classes, args):
    if model_name == 'pointnet':
        return PointNetClassifier(k=num_classes, feature_transform=args.feature_transform)
    elif model_name == 'pointnet++':
        return PointNetPlusPlusClassifier(k=num_classes)
    elif model_name == 'dgcnn':
        return DGCNNClassifier(k=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")