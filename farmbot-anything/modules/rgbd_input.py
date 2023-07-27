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
    
        self.rgbd = nn.Sequential(
            CBRelu(4, 64, 7, 2, 3),
            CBRelu(64, 128, 5, 1, 2),
            CBRelu(128, 256, 3, 1, 2),
            )

    def forward(self, input):
        x = self.rgbd(input)
        return x


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