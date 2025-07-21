import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import EnhancedPointNetSetAbstraction, PointNetFeaturePropagation


class EnhancedPointNetPlusPlusSegmentation(nn.Module):
    def __init__(self, num_classes, use_contrastive=False, contrastive_weight=0.1):
        super(EnhancedPointNetPlusPlusSegmentation, self).__init__()
        self.use_contrastive = use_contrastive
        self.contrastive_weight = contrastive_weight
        
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
        
        self.fp3 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128, mlp=[128, 128, 128])
        
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz, labels=None):
        norm = xyz
        
        total_contrastive_loss = 0
        total_consistency_loss = 0
        
        if self.use_contrastive and self.training:
            l1_xyz, l1_points, cont_loss1, cons_loss1 = self.sa1(xyz, norm, labels)
            l2_xyz, l2_points, cont_loss2, cons_loss2 = self.sa2(l1_xyz, l1_points, labels)
            l3_xyz, l3_points, _, _ = self.sa3(l2_xyz, l2_points, labels)
            
            total_contrastive_loss = cont_loss1 + cont_loss2
            total_consistency_loss = cons_loss1 + cons_loss2
        else:
            l1_xyz, l1_points = self.sa1(xyz, norm)[:2]
            l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)[:2]
            l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)[:2]

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        
        if self.use_contrastive and self.training:
            return x.permute(0, 2, 1), total_contrastive_loss, total_consistency_loss
        else:
            return x.permute(0, 2, 1), None, None


class SegmentationContrastiveLoss(nn.Module):
    def __init__(self, classification_weight=1.0, contrastive_weight=0.1, consistency_weight=0.05):
        super(SegmentationContrastiveLoss, self).__init__()
        self.classification_weight = classification_weight
        self.contrastive_weight = contrastive_weight
        self.consistency_weight = consistency_weight
        self.nll_loss = nn.NLLLoss()
    
    def forward(self, predictions, targets, contrastive_loss=None, consistency_loss=None):
        B, N, C = predictions.shape
        predictions = predictions.contiguous().view(-1, C)
        targets = targets.contiguous().view(-1)
        
        classification_loss = self.nll_loss(predictions, targets)
        
        total_loss = self.classification_weight * classification_loss
        
        if contrastive_loss is not None:
            total_loss += self.contrastive_weight * contrastive_loss
        
        if consistency_loss is not None:
            total_loss += self.consistency_weight * consistency_loss
        
        return total_loss, classification_loss, contrastive_loss, consistency_loss
