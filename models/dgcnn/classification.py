import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import EdgeConv, Transform


class DGCNNClassifier(nn.Module):
    def __init__(self, k=10, dropout=0.5):
        super(DGCNNClassifier, self).__init__()
        self.k = k
        self.dropout = dropout
        
        self.transform = Transform(k=3)
        
        self.conv1 = EdgeConv(3, 64, k=20)
        self.conv2 = EdgeConv(64, 64, k=20)
        self.conv3 = EdgeConv(64, 128, k=20)
        self.conv4 = EdgeConv(128, 256, k=20)
        
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=dropout),
            nn.Linear(256, k)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        trans = self.transform(x)
        x = torch.bmm(trans, x)
        
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)
        
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        
        x = self.classifier(x)
        x = F.log_softmax(x, dim=1)
        
        return x, trans, None
