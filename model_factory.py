from models.pointnet.classification import PointNetClassifier

def create_model(model_name, num_classes, args):
    if model_name == 'pointnet':
        return PointNetClassifier(k=num_classes, feature_transform=args.feature_transform)
    elif model_name == 'pointnet++':
        raise NotImplementedError("PointNet++ not implemented yet")
    elif model_name == 'dgcnn':
        raise NotImplementedError("DGCNN not implemented yet")
    else:
        raise ValueError(f"Unknown model: {model_name}")