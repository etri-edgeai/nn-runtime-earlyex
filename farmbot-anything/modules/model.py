import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.loss import chamfer_distance
from utils import *

from backbone import resnet18
from fpn import FPN
from body import Body
from rgbd_input import RGBDInput
random.seed(0)
torch.manual_seed(0)



class FAMENet(nn.Module):
    """entire segmentation module"""
    def __init__(self, cfg, device, num_class=49):
        # FAMENet 분석 모듈 초기화 설정
        super(FAMENet, self).__init__()
        self.cfg            = cfg
        self.img_size       = cfg['0_img_size']
        self.batch_size     = cfg['4_batch_size']       
        self.num_class      = num_class
        self.training       = True

        self.backbone       = resnet18().to(device=device)
        self.fpn            = FPN().to(device=device)

        self.body           = Body(cfg, num_classes=num_class,
                                in_channels=128,
                                device=device,
                                seg_feat_channels=256,
                                stacked_convs=7,
                                strides=[8, 8, 16, 32, 32],
                                scale_ranges=((1, 96), (48, 192), (96, 384), \
                                            (192, 768), (384, 2048)),
                                num_grids=[40, 36, 24, 16, 12]).to(device=device)

        self.rgbd_input     = RGBDInput(cfg).to(device=device)
   

    def forward(self, img, depth, pcd=None, masks=None, bboxes=None, labels=None, mode='train'):
        rgbd_feat = self.rgbd_input(img, depth)
        x = self.backbone(img)
        x = self.fpn(x)
        ins_pred, cate_pred, pcd_pred= self.body(x, rgbd_feat)
        return ins_pred, cate_pred, pcd_pred
    

if __name__ == "__main__":
    import yaml
    torch.cuda.empty_cache()
    with open("./config/config.yml", 'r') as f:
        cfg = yaml.safe_load(f)
    model = FAMENet(cfg, device='cuda:0')
    model.eval()

    input = torch.randn(1, 3, 256, 256).cuda()
    depth  = torch.randn(1, 1, 256,256).cuda()
    pcd    = torch.randn(1, 3, 10000).cuda()
    masks  = torch.randn(1, 256, 256).cuda()
    bboxes = torch.randn(1, 4).cuda()
    labels = torch.randn(1, 1).cuda()
    model(input, depth, pcd, masks, bboxes, labels)
    torch.onnx.export(model,
                      (input, depth, pcd, masks, bboxes, labels),
                    "famenet.onnx",
                    opset_version=11,
                    verbose=False,)
                    