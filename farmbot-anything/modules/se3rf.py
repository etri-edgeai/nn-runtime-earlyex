import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from .autoencoder import PCDDecoder
from .autoencoder import PCDAutoEncoder
# from .mask_rcnn import maskrcnn_resnet50_fpn_v2
from .solov1 import SOLOV1
# from .rrgbd_input import RGBDInputModule
from .rrgbd_input import RGBDInput
from .ssfusion import FeatureFusionModule, MaskFusionEncoder
import pytorch3d
from functools import partial

random.seed(0)
torch.manual_seed(0)

class E3RFnet(nn.Module):
    """entire segmentation module"""
    def __init__(self, cfg, device, num_class=49):
        # E3RF 분석 모듈 초기화 설정
        super(E3RFnet, self).__init__()
        self.cfg            = cfg
        self.img_size       = cfg['0_img_size']
        self.batch_size     = cfg['4_batch_size']       
        self.num_class      = num_class
        self.training       = True

        # E3RF Backbone+FPN 모듈 선언
        self.detect = SOLOV1(device, num_classes=num_class)
        self.detect.train()
        self.detect.training = True        
        try:
            print("Loading SOLOV1 checkpoint...")
            self.detect.load_state_dict(
                torch.load(self.cfg['3_solo_checkpoints']))
        except:
            print("No SOLOV1 checkpoint found, training from scratch...")
        
        # self.rgbd_input     = RGBDInputModule(cfg)
        self.rgbd_input       = RGBDInput(cfg)
        # self.feat_fusion    = FeatureFusionModule(cfg)
        # self.feat_fusion   = FeatureFusionModule(cfg)
        # self.mask_fusion    = MaskFusionEncoder(cfg)
        # self.autoencoder    = PCDAutoEncoder(
        #     embedding_dims  = cfg['4_embedding_dims'], pcd_samples = cfg['0_pcd_num'])
        # self.autoencoder.train()
        # self.decoder        = PCDDecoder(
        #     embedding_dims  = cfg['4_embedding_dims'], pcd_samples = cfg['0_pcd_num'])

        # try:
        #     print("Loading Decoder checkpoint...")
        #     self.autoencoder.load_state_dict(
        #         torch.load(self.cfg['2_pcd_checkpoints']))
        # except:
        #     print("No Decoder checkpoint found, training from scratch...")
        
        for name, param in self.named_parameters():
            param.requires_grad = True

    def feat_fuse(self, rgbd_feat, ins_pred):
        result = rgbd_feat + ins_pred
        return result

    def forward_loss(self, img, depth, pcd, masks, bboxes, labels):
        batch, channels, height, width = img.shape
        img_feats = self.detect.extract_feat(img)
        
        ins_pred, cate_pred, pcd_pred = self.detect.bbox_head(img_feats)

        img_feats = torch.cat([F.interpolate(
            i, size=(32, 32), mode='bilinear', align_corners=True) \
                for i in img_feats],dim=1)
        depth   = F.interpolate(
            depth.unsqueeze(1), size=(height,width),mode='nearest')

        loss = self.detect.bbox_head.loss(
            ins_pred, cate_pred, pcd_pred, bboxes, labels, masks, pcd)

        return loss
    
    def forward_inference(self, img, depth):

        return pcd_preds


    def forward(self, img, depth, pcd=None, masks=None, bboxes=None, 
        labels=None, mode='train'):
        if mode == 'test' or mode == 'val':
            return self.forward_inference(img, depth)
        else:
            return self.forward_loss(img, depth, pcd,  masks, bboxes, labels)



def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
            map the multiple outputs of the ``func`` into different
            list. Each list contains the same type of outputs corresponding
            to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))

def print_nested_list_shape(lst, prefix=""):
    if isinstance(lst, list):
        print(f"{prefix}List of shape {len(lst)}")
        for item in lst:
            print_nested_list_shape(item, prefix + "  ")
    elif isinstance(lst, torch.Tensor):
        print(f"{prefix}Tensor of shape {lst.shape}")

def imrescale(img, scale):
    # Convert the image to a PyTorch tensor and add an extra batch dimension
    # img = torch.from_numpy(img).unsqueeze(0)

    # Compute the new size
    h, w = img.shape[-2:]
    new_size = (int(h * scale), int(w * scale))

    # Use F.interpolate to resize the image
    rescaled_img = F.interpolate(img.unsqueeze(0), size=new_size, mode='bilinear', align_corners=False)

    # Remove the extra batch dimension and convert the image back to a numpy array
    rescaled_img = rescaled_img.squeeze(0)

    return rescaled_img