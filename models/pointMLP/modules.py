import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometricAffine(nn.Module):
    def __init__(self, in_channels):
        super(GeometricAffine, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, in_channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, in_channels, 1))
        self.eps = 1e-6
        
    def forward(self, x):
        mu = torch.mean(x, dim=-1, keepdim=True)
        sigma = torch.sqrt(torch.var(x, dim=-1, keepdim=True, unbiased=False) + self.eps)
        
        normalized = (x - mu) / (sigma + self.eps)
        
        return self.alpha * normalized + self.beta


class ResidualPointBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualPointBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += residual
        out = F.relu(out)
        
        return out


class PointMLPEncoder(nn.Module):
    def __init__(self):
        super(PointMLPEncoder, self).__init__()
        
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
        
        self.stage5 = nn.Sequential(
            ResidualPointBlock(512, 512),
            ResidualPointBlock(512, 1024),
            ResidualPointBlock(1024, 1024)
        )
        
    def forward(self, x):
        x = self.geometric_affine(x)
        
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        return x
