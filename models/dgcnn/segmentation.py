import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import EdgeConv, Transform


class DGCNNSegmentation(nn.Module):
    def __init__(self, k=13, dropout=0.5):
        super(DGCNNSegmentation, self).__init__()
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
        
        self.conv6 = nn.Sequential(
            nn.Conv1d(1600, 512, kernel_size=1, bias=False),
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
        
    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)
        
        trans = self.transform(x)
        x = torch.bmm(trans, x)
        
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        
        x_cat = torch.cat((x1, x2, x3, x4), dim=1)
        x5 = self.conv5(x_cat)
        
        x_global = F.adaptive_max_pool1d(x5, 1).repeat(1, 1, num_points)
        
        x = torch.cat((x_cat, x_global), dim=1)
        
        x = self.conv6(x)
        x = self.dropout_layer(x)
        
        x = self.conv7(x)
        x = self.dropout_layer(x)
        
        x = self.conv8(x)
        x = self.dropout_layer(x)
        
        x = self.conv9(x)
        x = F.log_softmax(x, dim=1)
        
        return x, trans, None
