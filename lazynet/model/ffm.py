import torch
from torch import nn
import torch.nn.functional as F
from model.cbrelu import CBRelu
class FeatureFusion(nn.Module):
    def __init__(self, shape, reduction=1):
        """feature fusion module"""
        super().__init__()
        batch, channel, width, height = shape
        self.conv1 = nn.Sequential(
            CBRelu(channel*2, channel//8, 1, 1, 0),
            CBRelu(channel//8, channel, 1, 1, 0))

        self.resize = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x1, x2):
        """forward"""
        x1 = self.resize(x1)
        x2 = self.resize(x2)
        x = torch.cat([x1, x2], dim=1)
        mid = self.conv1(x)
        return mid
