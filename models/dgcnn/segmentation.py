import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import EdgeConv, Transform

class DGCNNSegmentation(nn.Module):
    def __init__(self, k=13, in_channels=3, k_nn=20, dropout=0.5):
        super(DGCNNSegmentation, self).__init__()
        assert in_channels in (3, 6), f"in_channels must be 3 or 6, got {in_channels}"
        self.k = k
        self.in_channels = in_channels
        self.dropout = dropout
        self.transform = Transform(k=3)

        self.conv1 = EdgeConv(in_channels, 64, k=k_nn)
        self.conv2 = EdgeConv(64, 64, k=k_nn)
        self.conv3 = EdgeConv(64, 128, k=k_nn)
        self.conv4 = EdgeConv(128, 256, k=k_nn)

        # After concatenating x1,x2,x3,x4 → 64+64+128+256 = 512
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2)
        )

        # concat(x_cat [512], x_global [1024]) → 1536 (FIX: it was 1600)
        self.conv6 = nn.Sequential(
            nn.Conv1d(1536, 512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.conv7 = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.conv8 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.conv9 = nn.Sequential(
            nn.Conv1d(128, k, kernel_size=1, bias=False),
        )

        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, x, cat_labels=None):
        """
        x: [B, C, N]  (C = 3 or 6). cat_labels is accepted for API compatibility; not used.
        Returns:
            log_probs [B, k, N]  (trainer will transpose if needed),
            trans [B,3,3],
            trans_feat=None (DGCNN has no feature transform by default)
        """
        B, C, N = x.size()
        assert C == self.in_channels, f"Expected input channels {self.in_channels}, got {C}"

        # --- Align XYZ only with 3x3 transform ---
        xyz = x[:, :3, :]                            # [B,3,N]
        trans = self.transform(xyz)                  # [B,3,3]
        xyz = torch.bmm(trans, xyz)                  # [B,3,N]
        if C > 3:
            rest = x[:, 3:, :]                       # e.g., normals
            x = torch.cat([xyz, rest], dim=1)        # [B,C,N]
        else:
            x = xyz                                   # [B,3,N]

        # --- DGCNN EdgeConv backbone ---
        x1 = self.conv1(x)                           # [B, 64, N]
        x2 = self.conv2(x1)                          # [B, 64, N]
        x3 = self.conv3(x2)                          # [B,128, N]
        x4 = self.conv4(x3)                          # [B,256, N]

        x_cat = torch.cat((x1, x2, x3, x4), dim=1)   # [B,512,N]
        x5 = self.conv5(x_cat)                       # [B,1024,N]

        x_global = F.adaptive_max_pool1d(x5, 1).repeat(1, 1, N)  # [B,1024,N]
        x = torch.cat((x_cat, x_global), dim=1)      # [B,1536,N]

        x = self.conv6(x)                            # [B,512,N]
        x = self.dropout_layer(x)

        x = self.conv7(x)                            # [B,256,N]
        x = self.dropout_layer(x)

        x = self.conv8(x)                            # [B,128,N]
        x = self.dropout_layer(x)

        x = self.conv9(x)                            # [B,k,N]
        x = F.log_softmax(x, dim=1)                  # log-probs over classes (channel dim)

        return x, trans, None
