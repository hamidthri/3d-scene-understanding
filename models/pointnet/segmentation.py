import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import PointNetEncoder


class PointNetSegmentation(nn.Module):
    """
    PointNet for part segmentation.

    Args:
        k (int): number of part classes (num_parts).
        in_channels (int): 3 for xyz, 6 for xyz+normals.
        feature_transform (bool): enable feature transform (T-Net) regularizer.
    Returns (forward):
        x: [B, N, k] log-probabilities (use NLLLoss)
        trans, trans_feat: transforms for optional regularization
    """
    def __init__(self, k: int = 2, in_channels: int = 3, feature_transform: bool = False):
        super().__init__()
        self.k = int(k)
        self.in_channels = int(in_channels)
        self.feature_transform = bool(feature_transform)

        # Many PointNetEncoder impls accept `channel` (a.k.a. in_channels). If yours doesn't,
        # we gracefully fallback but require in_channels == 3.
        try:
            self.feat = PointNetEncoder(
                global_feat=False,
                feature_transform=self.feature_transform,
                channel=self.in_channels,        # standard kwarg name in common repos
            )
        except TypeError:
            # Older encoder without `channel` kwarg
            self.feat = PointNetEncoder(
                global_feat=False,
                feature_transform=self.feature_transform,
            )
            if self.in_channels != 3:
                raise ValueError(
                    "Your PointNetEncoder doesn't accept `channel=` and only supports 3D inputs. "
                    "Either pass --normal off (in_channels=3) or update the encoder to accept 6."
                )

        # Decoder head (standard PointNet seg head expects encoder output 1088 channels: 64+128+1024)
        self.conv1 = nn.Conv1d(1088, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, self.k, 1)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x: torch.Tensor, cat_labels: torch.Tensor = None):
        """
        x: [B, C, N] with C == in_channels (3 or 6). cat_labels is optional and ignored.
        Returns:
            log_probs: [B, N, k]
            trans, trans_feat: for regularization
        """
        assert x.dim() == 3, f"Expected input [B,C,N], got {tuple(x.shape)}"
        B, C, N = x.shape
        if C != self.in_channels:
            raise ValueError(f"Input channels ({C}) != configured in_channels ({self.in_channels}).")

        # Encoder produces per-point features of size 1088: [B, 1088, N]
        x, trans, trans_feat = self.feat(x)

        # MLP head
        x = F.relu(self.bn1(self.conv1(x)))   # [B, 512, N]
        x = F.relu(self.bn2(self.conv2(x)))   # [B, 256, N]
        x = F.relu(self.bn3(self.conv3(x)))   # [B, 128, N]
        x = self.conv4(x)                     # [B, k, N]

        # Return [B, N, k] log-probabilities (trainer uses NLLLoss)
        x = x.transpose(2, 1).contiguous()    # [B, N, k]
        x = F.log_softmax(x, dim=-1)
        return x, trans, trans_feat
