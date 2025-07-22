import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import GeometricAffine, ResidualPointBlock, PointMLPEncoder



class PointMLPClassifier(nn.Module):
    def __init__(self, k=10):
        super(PointMLPClassifier, self).__init__()
        self.encoder = PointMLPEncoder()
        
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, k)
        )
        
    def forward(self, x):
        features = self.encoder(x)
        x = self.classifier(features)
        return F.log_softmax(x, dim=1)
