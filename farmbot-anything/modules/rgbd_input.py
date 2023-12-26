import os
import sys
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
cwd = os.getcwd()
sys.path.append(cwd)
from .cbrelu import CBRelu

random.seed(0)
torch.manual_seed(0)

class RGBDInput(nn.Module):
    def __init__(self, cfg):
        super(RGBDInput, self).__init__()
        self.cfg = cfg
        
        # RGB pathway
        self.rgb_conv1 = CBRelu(3, 32, 7, 2, 3)
        self.rgb_conv2 = CBRelu(32, 64, 5, 2, 2)
        self.rgb_residual = CBRelu(64, 64, 3, 1, 1)  # Residual block for RGB

        # Depth pathway
        self.depth_conv1 = CBRelu(1, 32, 7, 2, 3)
        self.depth_conv2 = CBRelu(32, 64, 5, 2, 2)
        self.depth_residual = CBRelu(64, 64, 3, 1, 1)
        # Combined pathway
        self.combined_conv = CBRelu(128, 128, 3, 2, 1)

    def forward(self, img, depth):
        # RGB pathway
        rgb = self.rgb_conv1(img)
        rgb = self.rgb_conv2(rgb)
        rgb_residual = self.rgb_residual(rgb)
        
        # Add the residual (shortcut connection)
        rgb = rgb + rgb_residual

        # Depth pathway
        depth = self.depth_conv1(depth)
        depth = self.depth_conv2(depth)
        depth_residual = self.depth_residual(depth)
        depth = depth + depth_residual

        # Concatenate the RGB and depth pathways
        comb = torch.cat([rgb, depth], dim=1)
        comb = self.combined_conv(comb)
        return comb

class RGBDInputModule(nn.Module):
    def __init__(self, cfg):
        super(RGBDInputModule, self).__init__()
        self.cfg = cfg
    
        self.rgbd = nn.Sequential(
            CBRelu(4, 64, 7, 2, 3),
            CBRelu(64, 128, 5, 2, 2),
            CBRelu(128, 256, 3, 2, 1),
            )

    def forward(self, input):
        x = self.rgbd(input)
        return x

if __name__ == "__main__":
    with open("./config/e3rf.yml", 'r') as f:
        cfg = yaml.safe_load(f)
    rgbd = RGBDInput(cfg)
    input = torch.randn(1, 4, 256, 256)
    output = rgbd.forward(input)
    print("input_shape: ", input.shape)
    print("output_shape: ", output.shape)