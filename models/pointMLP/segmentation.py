import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import GeometricAffine, ResidualPointBlock


class PointMLPSegmentation(nn.Module):
    """
    PointMLP segmentation.

    Args:
        k (int): number of classes.
        in_channels (int): 3 (XYZ) or 6 (e.g., XYZ+normals). Only 3 or 6 are allowed.
        apply_log_softmax (bool): if True returns log-probs (for NLLLoss),
                                  else returns logits (for CrossEntropyLoss).
    Returns from forward:
        seg_pred: (B, N, k)
        trans: None  (for compatibility with trainers expecting 3 outputs)
        trans_feat: None
    """
    def __init__(self, k: int = 2, in_channels: int = 3, apply_log_softmax: bool = True):
        super().__init__()
        if in_channels not in (3, 6):
            raise ValueError(f"in_channels must be 3 or 6, got {in_channels}")
        self.k = k
        self.apply_log_softmax = apply_log_softmax

        # Input normalization/adaptation for C_in points
        self.geometric_affine = GeometricAffine(in_channels)

        # Feature extractor stages; start from in_channels
        self.stage1 = nn.Sequential(
            ResidualPointBlock(in_channels, 32),
            ResidualPointBlock(32, 32),
            ResidualPointBlock(32, 64),
        )
        self.stage2 = nn.Sequential(
            ResidualPointBlock(64, 64),
            ResidualPointBlock(64, 128),
            ResidualPointBlock(128, 128),
        )
        self.stage3 = nn.Sequential(
            ResidualPointBlock(128, 128),
            ResidualPointBlock(128, 256),
            ResidualPointBlock(256, 256),
        )
        self.stage4 = nn.Sequential(
            ResidualPointBlock(256, 256),
            ResidualPointBlock(256, 512),
            ResidualPointBlock(512, 512),
        )

        self.global_conv = nn.Conv1d(512, 1024, 1)
        self.global_bn = nn.BatchNorm1d(1024)

        # point_features(512) + global(1024) -> 1536
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
            nn.Conv1d(128, k, 1),
        )

    def forward(self, x: torch.Tensor, cat_labels: torch.Tensor = None):
        """
        x: (B, in_channels, N)
        cat_labels: (B,) optional category ids — ignored here but accepted for trainer compatibility.

        Returns:
            seg_pred: (B, N, k) [log-probs if apply_log_softmax else logits]
            trans: None
            trans_feat: None
        """
        B, _, N = x.size()

        x = self.geometric_affine(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        point_features = self.stage4(x)  # (B, 512, N)

        global_features = F.relu(self.global_bn(self.global_conv(point_features)))  # (B, 1024, N)
        global_features = torch.max(global_features, dim=2, keepdim=True)[0]        # (B, 1024, 1)
        global_features = global_features.repeat(1, 1, N)                           # (B, 1024, N)

        combined = torch.cat([point_features, global_features], dim=1)              # (B, 1536, N)
        logits = self.segmentation_head(combined)                                   # (B, k, N)
        logits = logits.transpose(2, 1).contiguous()                                # (B, N, k)

        if self.apply_log_softmax:
            seg_pred = F.log_softmax(logits, dim=-1)
        else:
            seg_pred = logits

        # Return Nones for (trans, trans_feat) to match PointNet-style trainer signatures
        return seg_pred, None, None
