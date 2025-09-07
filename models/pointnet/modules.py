import torch
import torch.nn as nn
import torch.nn.functional as F

class TNet(nn.Module):
    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k
        
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        
        self.dropout = nn.Dropout(p=0.3)
        
    def forward(self, x):
        batchsize = x.size()[0]
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        
        iden = torch.eye(self.k, dtype=x.dtype, device=x.device).view(1, self.k * self.k).repeat(batchsize, 1)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        """
        Args:
            global_feat (bool): if True, return [B,1024] global feature (cls); else per-point [B,1088,N] (seg)
            feature_transform (bool): enable 64x64 feature-space T-Net
            channel (int): input channels, 3 for xyz, 6 for xyz+normals
        """
        super(PointNetEncoder, self).__init__()
        assert channel in (3, 6), f"PointNetEncoder expects channel 3 or 6, got {channel}"
        self.input_channel = channel

        self.stn = TNet(k=3)
        self.conv1 = nn.Conv1d(self.input_channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)

        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = TNet(k=64)  # 64x64 feature transform

        self.conv3 = nn.Conv1d(64, 128, 1)
        self.conv4 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(1024)

        self.global_feat = global_feat

    def forward(self, x):
        """
        x: [B, C, N], C=3 (xyz) or 6 (xyz+normals)
        Returns:
          if global_feat: ([B,1024], trans, trans_feat)
          else:           ([B,1088,N], trans, trans_feat)  where 1088=1024(global)+64(pointfeat)
        """
        B, C, N = x.size()
        assert C == self.input_channel, f"Expected input channel {self.input_channel}, got {C}"

        trans_feat = None
        # --- Input transform on xyz only (3x3) ---
        xyz = x[:, :3, :]                            # [B,3,N]
        trans = self.stn(xyz)                        # [B,3,3]
        xyz = torch.bmm(xyz.transpose(2, 1), trans).transpose(2, 1)  # [B,3,N]

        if C > 3:
            rest = x[:, 3:, :]                       # [B,C-3,N] (e.g., normals)
            x = torch.cat([xyz, rest], dim=1)        # [B,C,N]
        else:
            x = xyz                                   # [B,3,N]

        # --- Local feature extraction ---
        x = F.relu(self.bn1(self.conv1(x)))          # [B,64,N]
        x = F.relu(self.bn2(self.conv2(x)))          # [B,64,N]

        if self.feature_transform:
            trans_feat = self.fstn(x)                # [B,64,64]
            x = torch.bmm(x.transpose(2, 1), trans_feat).transpose(2, 1)  # [B,64,N]

        pointfeat = x                                 # [B,64,N]  (kept for seg concat)

        x = F.relu(self.bn3(self.conv3(x)))          # [B,128,N]
        x = self.bn4(self.conv4(x))                  # [B,1024,N]
        x = torch.max(x, 2, keepdim=True)[0]         # [B,1024,1]
        x = x.view(B, 1024)                          # [B,1024]

        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(B, 1024, 1).repeat(1, 1, N)   # [B,1024,N]
            out = torch.cat([x, pointfeat], 1)       # [B,1024+64=1088,N]
            return out, trans, trans_feat
