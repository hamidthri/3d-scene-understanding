import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import PointNetSetAbstraction
from .modules import PointNetFeaturePropagation

class PointNetPlusPlusSegmentation(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 3):
        """
        Args:
            num_classes: number of part labels
            in_channels: 3 (xyz) or 6 (xyz+normals)
        """
        super().__init__()
        assert in_channels in (3, 6), f"in_channels must be 3 or 6, got {in_channels}"
        self.num_classes = int(num_classes)
        self.in_channels = int(in_channels)

        feat_dim = self.in_channels - 3  # 0 if only xyz, 3 if xyz+normals


        self.sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.2, nsample=32,
            in_channel=3 + feat_dim, mlp=[64, 64, 128], group_all=False
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=128, radius=0.4, nsample=64,
            in_channel=128 + 3, mlp=[128, 128, 256], group_all=False
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None,
            in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True
        )

        # FP blocks (follow the canonical PN++ seg head sizes)
        self.fp3 = PointNetFeaturePropagation(in_channel=1024 + 256, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=256 + 128,  mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128 + feat_dim, mlp=[128, 128, 128])

        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, self.num_classes, 1)

    def forward(self, x, cat_labels=None):
        """
        x: [B, C, N] where C=3 or 6. If C=6, last 3 channels are normals.
        Returns:
          seg_pred: [B, N, num_classes] (log-probs), trans=None, trans_feat=None
        """
        B, C, N = x.size()
        assert C == self.in_channels, f"Input channels ({C}) != configured ({self.in_channels})"

        xyz = x[:, :3, :]                 # [B,3,N]
        points = x[:, 3:, :] if C > 3 else None  # [B,feat,N] or None

        # Set Abstraction
        l1_xyz, l1_points = self.sa1(xyz, points)     # [B,3,512], [B,128,512]
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # [B,3,128], [B,256,128]
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  # [B,3,1],   [B,1024,1]

        # Feature Propagation
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)  # [B,256,128]
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)  # [B,128,512]
        l0_points = self.fp1(xyz,    l1_xyz, points,   l1_points)   # [B,128,N]

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))     # [B,128,N]
        x = self.conv2(x)                                           # [B,num_classes,N]
        x = F.log_softmax(x, dim=1)                                 # [B,num_classes,N]
        x = x.permute(0, 2, 1).contiguous()                         # [B,N,num_classes]
        return x, None, None
