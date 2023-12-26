import os
import random
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .cbrelu import CBRelu

class FeatureFusionModule(nn.Module):
    def __init__(self, cfg, in_channels=1280, out_channels=256):
        # E3RF RGBD and Detection Fusion 
        super(FeatureFusionModule, self).__init__()
        self.cfg = cfg
        self.conv1 = CBRelu(in_channels, out_channels//4, 1, 1, 0)
        # Attention 설정
        self.atten = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            CBRelu(out_channels // 4, out_channels // 8, 1, 1, 0),
            CBRelu(out_channels // 8, out_channels // 4, 1, 1, 0),
        )
        self.conv2 = CBRelu(out_channels//4, 4, 1, 1, 0)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.conv1(x)
        att = self.atten(x)
        out = x + x * att
        out = self.conv2(out)
        return out

class MaskFusionEncoder(nn.Module):
    def __init__(self, cfg, in_channels=5, out_channels=64):
        # E3RF RGBD and Detection Fusion 
        super(MaskFusionEncoder, self).__init__()
        self.cfg = cfg

        self.conv1 = CBRelu(
            in_channels=in_channels, out_channels=out_channels, \
                kernel_size=1, stride=1, padding=0, \
                dilation=1, groups=1)

        # Attention 설정
        self.atten = nn.Sequential(
            CBRelu(out_channels, out_channels // 2, 1, 1, 0),
            CBRelu(out_channels // 2, out_channels, 1, 1, 0),
        )

        self.conv2 = CBRelu(
            in_channels=out_channels, out_channels=out_channels, \
                kernel_size=1, stride=1, padding=0, \
                dilation=1, groups=1)

        # Encoder 설정
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*64*64, 512),
            nn.ReLU(),
            nn.Linear(512, 2048),
        )

    def forward(self, x1, x2):
        # Feature Fusion 모듈 추론
        # x1 = x1.view(-1, in_channels)

        x1 = F.interpolate(x1, size=(64, 64), mode='bilinear', align_corners=True)
        x2 = F.interpolate(x2, size=(64, 64), mode='bilinear', align_corners=True)

        # x1 = x1.unsqueeze(0).unsqueeze(0)
        # x2 = x2.unsqueeze(0).unsqueeze(0)
        x = torch.cat([x1, x2], dim=1)
        # Attention 적용
        x = self.conv1(x)
        att = self.atten(x)
        x = x + x * att
        x = self.conv2(x)
        out = self.encoder(x)
        out = out.view(-1, 2048)
        return out


class FeatureFusionEncoder(nn.Module):
    def __init__(self, cfg, in_channels=1280, out_channels=64):
        # E3RF RGBD and Detection Fusion 
        super(FeatureFusionEncoder, self).__init__()
        self.cfg = cfg

        self.conv1 = CBRelu(
            in_channels=in_channels, out_channels=out_channels, \
                kernel_size=1, stride=1, padding=0, \
                dilation=1, groups=1)

        # Attention 설정
        self.atten = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            CBRelu(out_channels, out_channels // 2, 1, 1, 0),
            CBRelu(out_channels // 2, out_channels, 1, 1, 0),
        )

        self.conv2 = CBRelu(
            in_channels=out_channels, out_channels=out_channels, \
                kernel_size=1, stride=1, padding=0, \
                dilation=1, groups=1)

        # Encoder 설정
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*32*64, 128),
            nn.ReLU(),
            nn.Linear(128, 2048),
        )

    def forward(self, x1, x2):
        # Feature Fusion 모듈 추론
        x1 = F.interpolate(x1, size=(32, 32), mode='bilinear', align_corners=True)
        x2 = F.interpolate(x2, size=(32, 32), mode='bilinear', align_corners=True)

        x = torch.cat([x1, x2], dim=1)
        # Attention 적용
        x = self.conv1(x)
        att = self.atten(x)
        x = x + x * att
        x = self.conv2(x)
        out = self.encoder(x)
        out = out.view(-1, 2048)
        return out

if __name__ == "__main__":
    with open("./config/e3rf.yml", 'r') as f:
        cfg = yaml.safe_load(f)
    rgbd = FeatureFusionEncoder(cfg, 64)
    x1 = torch.randn(2, 256, 32, 32)
    x2 = torch.randn(2, 256, 32, 32)
        
    output = rgbd.forward(x1, x2)
    print("output_shape: ", output.shape)