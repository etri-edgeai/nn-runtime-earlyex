import torch
import torch.nn as nn
import torch.nn.functional as F


class CBRelu(nn.Module):
    """Block module class"""
    def __init__(
        self, 
        in_channels=256, 
        out_channels=256, 
        kernel_size=3,
        stride=1, padding=0, dilation=1, groups=1):
        """init function"""
        super(CBRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=0.5)
        
    def forward(self, x):
        """forward function"""
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.drop(x)
        return x