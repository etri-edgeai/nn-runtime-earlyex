import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision import datasets
from tqdm import tqdm
import numpy as np
from datetime import datetime
from metric_ex.utils import config, get_dataloader, get_dataset
from metric_ex.model.backbone import get_backbone
from metric_ex.trainer.ce_branch import CEBranchTrainer
from metric_ex.model import Model
from metric_ex.model.devour import DevourModel
import argparse 
import sys
from metric_ex.model.backbone import resnet18
from metric_ex.model.backbone.efficientnet import EfficientNet
from torchvision.models.mobilenetv2 import mobilenet_v2
from torchvision.models.vgg import vgg16_bn



def main():
    print("Branch Trainer v0.5")
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default = "./configs/train.yml")
    args = parser.parse_args()
    cfg = config(args.config)
    backbone = resnet18(cfg)

    model = DevourModel(backbone, cfg)

    model.hunt(keyword= "Sequential", out="Sequential")

if __name__ == "__main__":
    main()
