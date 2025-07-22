import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import GeometricAffine, ResidualPointBlock


class PointMLPSegmentation(nn.Module):
    def __init__(self, k=2):
        super(PointMLPSegmentation, self).__init__()
        self.k = k
        
        self.geometric_affine = GeometricAffine(3)
        
        self.stage1 = nn.Sequential(
            ResidualPointBlock(3, 32),
            ResidualPointBlock(32, 32),
            ResidualPointBlock(32, 64)
        )
        
        self.stage2 = nn.Sequential(
            ResidualPointBlock(64, 64),
            ResidualPointBlock(64, 128),
            ResidualPointBlock(128, 128)
        )
        
        self.stage3 = nn.Sequential(
            ResidualPointBlock(128, 128),
            ResidualPointBlock(128, 256),
            ResidualPointBlock(256, 256)
        )
        
        self.stage4 = nn.Sequential(
            ResidualPointBlock(256, 256),
            ResidualPointBlock(256, 512),
            ResidualPointBlock(512, 512)
        )
        
        self.global_conv = nn.Conv1d(512, 1024, 1)
        self.global_bn = nn.BatchNorm1d(1024)
        
        self.segmentation_head = nn.Sequential(
            nn.Conv1d(1024 + 512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, k, 1)
        )
        
    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        
        x = self.geometric_affine(x)
        
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        point_features = self.stage4(x)
        
        global_features = F.relu(self.global_bn(self.global_conv(point_features)))
        global_features = torch.max(global_features, 2, keepdim=True)[0]
        global_features = global_features.repeat(1, 1, n_pts)
        
        combined_features = torch.cat([point_features, global_features], 1)
        
        x = self.segmentation_head(combined_features)
        x = x.transpose(2, 1).contiguous()
        x = F.log_softmax(x.view(-1, self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        
        return x