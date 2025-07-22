import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import EnhancedPointNetSetAbstraction


class EnhancedPointNetPlusPlusClassifier(nn.Module):
    def __init__(
        self, 
        k=10, 
        use_contrastive=True, 
        contrastive_weight=0.1,
        layer_weights=(0.5, 0.5)  # weights for sa1 and sa2 losses
    ):
        super(EnhancedPointNetPlusPlusClassifier, self).__init__()
        self.use_contrastive = use_contrastive
        self.contrastive_weight = contrastive_weight
        self.layer_weights = layer_weights

        self.sa1 = EnhancedPointNetSetAbstraction(
            npoint=512, radius=0.2, nsample=32, in_channel=6, 
            mlp=[64, 64, 128], group_all=False, use_contrastive=use_contrastive
        )
        self.sa2 = EnhancedPointNetSetAbstraction(
            npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, 
            mlp=[128, 128, 256], group_all=False, use_contrastive=use_contrastive
        )
        self.sa3 = EnhancedPointNetSetAbstraction(
            npoint=None, radius=None, nsample=None, in_channel=256 + 3, 
            mlp=[256, 512, 1024], group_all=True, use_contrastive=False
        )

        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, k)

    def forward(self, xyz, labels=None, return_aux=False):
        B, _, _ = xyz.shape
        norm = xyz  # can normalize here if needed

        total_contrastive_loss = 0
        total_consistency_loss = 0

        if self.use_contrastive and self.training:
            l1_xyz, l1_points, cont_loss1, cons_loss1 = self.sa1(xyz, norm, labels)
            l2_xyz, l2_points, cont_loss2, cons_loss2 = self.sa2(l1_xyz, l1_points, labels)
            l3_xyz, l3_points, _, _ = self.sa3(l2_xyz, l2_points, labels)

            # Weighted loss combination from layers
            total_contrastive_loss = (
                self.layer_weights[0] * cont_loss1 +
                self.layer_weights[1] * cont_loss2
            )
            total_consistency_loss = (
                self.layer_weights[0] * cons_loss1 +
                self.layer_weights[1] * cons_loss2
            )
        else:
            l1_xyz, l1_points = self.sa1(xyz, norm)[:2]
            l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)[:2]
            l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)[:2]

        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)

        if self.use_contrastive and (self.training or return_aux):
            return x, total_contrastive_loss, total_consistency_loss
        else:
            return x, None, None


class ContrastiveLoss(nn.Module):
    def __init__(
        self,
        classification_weight=1.0,
        contrastive_weight=0.1,
        consistency_weight=0.05,
        warmup_epochs=10
    ):
        super(ContrastiveLoss, self).__init__()
        self.classification_weight = classification_weight
        self.contrastive_weight = contrastive_weight
        self.consistency_weight = consistency_weight
        self.warmup_epochs = warmup_epochs
        self.epoch = 0  # will be set externally in training loop
        self.nll_loss = nn.NLLLoss()

    def forward(self, predictions, targets, contrastive_loss=None, consistency_loss=None):
        cls_loss = self.nll_loss(predictions, targets)
        total_loss = self.classification_weight * cls_loss

        # Warm-up scaling
        warmup_scale = min(1.0, self.epoch / self.warmup_epochs)

        if contrastive_loss is not None:
            total_loss += (self.contrastive_weight * warmup_scale) * contrastive_loss

        if consistency_loss is not None:
            total_loss += (self.consistency_weight * warmup_scale) * consistency_loss

        return total_loss, cls_loss, contrastive_loss, consistency_loss

    def set_epoch(self, epoch):
        self.epoch = epoch
