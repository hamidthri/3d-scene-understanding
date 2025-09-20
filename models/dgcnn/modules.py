import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    # x: [B, C, N]
    xt = x.transpose(2, 1).contiguous()          # [B, N, C]
    inner = -2 * torch.matmul(xt, xt.transpose(2, 1))  # [B, N, N]
    xx = torch.sum(xt ** 2, dim=2, keepdim=True)       # [B, N, 1]
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    _, idx = pairwise_distance.topk(k=k, dim=-1)       # [B, N, k] (long)
    return idx


def get_graph_feature(x, k=20, idx=None):
    # x: [B, C, N]
    B, C, N = x.size()
    device = x.device

    if idx is None:
        idx = knn(x, k=k)                     # already on x.device
    else:
        idx = idx.to(device, non_blocking=True)

    # build base indices on the SAME device
    idx_base = torch.arange(B, device=device).view(-1, 1, 1) * N  # [B,1,1]
    idx = (idx + idx_base).reshape(-1)        # [(B*N*k)]

    # gather neighbor features
    x_t = x.transpose(2, 1).contiguous()      # [B, N, C]
    x_flat = x_t.reshape(B * N, C)            # [B*N, C]
    feature = x_flat[idx, :].view(B, N, k, C) # [B, N, k, C]

    x_central = x_t.view(B, N, 1, C).expand(-1, -1, k, -1)  # [B, N, k, C]
    feature = torch.cat((feature - x_central, x_central), dim=3)  # [B, N, k, 2C]
    return feature.permute(0, 3, 1, 2).contiguous()          # [B, 2C, N, k]


class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, k=20):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        x = get_graph_feature(x, k=self.k)     # [B, 2C, N, k]
        x = self.conv(x)                       # [B, out, N, k]
        x = x.max(dim=-1, keepdim=False)[0]    # [B, out, N]
        return x


class Transform(nn.Module):
    def __init__(self, k=3):
        super().__init__()
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

    def forward(self, x):
        B = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0].view(B, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        # identity on SAME device as current tensor
        ident = torch.eye(self.k, device=x.device).view(1, self.k * self.k).repeat(B, 1)
        x = (x + ident).view(B, self.k, self.k)
        return x
