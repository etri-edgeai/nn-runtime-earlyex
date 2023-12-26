import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from .autoencoder import PCDDecoder
from .mask_rcnn import maskrcnn_resnet50_fpn_v2, MaskRCNN
import pytorch3d
from pytorch3d.loss import chamfer_distance
from .rgbd_input import RGBDInputModule
from .fusion import FeatureFusionModule, MaskFusionEncoder
random.seed(0)
torch.manual_seed(0)
from typing import Dict, List, Optional, Tuple, Union
from .utils import postprocess


class E3RFnet(nn.Module):
    """entire segmentation module"""
    def __init__(self, cfg, num_class, device):
        # E3RF 분석 모듈 초기화 설정
        super(E3RFnet, self).__init__()
        self.cfg            = cfg
        self.device         = device
        self.img_size       = cfg['img_size']
        self.batch_size     = cfg['batch_size']       
        self.num_class      = 49
        self.merge_size     = 64

        # E3RF Backbone+FPN 모듈 선언
        self.maskrcnn       = maskrcnn_resnet50_fpn_v2(num_classes=self.num_class)
        self.maskrcnn.train()
        self.maskrcnn.training = True
        # backbone = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.DEFAULT).features
        # backbone.out_channels = 1280

        # anchor_generator = torchvision.models.detection.rpn.AnchorGenerator(
        #     sizes=((32, 64, 128, 256, 512),),
        #     aspect_ratios=((0.5, 1.0, 2.0),))
        
        # roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        #     featmap_names=['0'],
        #     output_size=7,
        #     sampling_ratio=2)
        
        # mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        #     featmap_names=['0'],
        #     output_size=14,
        #     sampling_ratio=2)
        
        # self.maskrcnn = MaskRCNN(backbone,
        #                       num_classes=49,
        #                       rpn_anchor_generator=anchor_generator,
        #                       box_roi_pool=roi_pooler,
        #                       mask_roi_pool=mask_roi_pooler)

        try:
            print("Loading MaskRCNN checkpoint...")
            self.maskrcnn.load_state_dict(torch.load(self.cfg['e3rf_checkpoints']))
        except:
            print("No MaskRCNN checkpoint found, training from scratch...")        
        
        self.rgbd_input = RGBDInputModule(cfg)
        self.feat_fusion    = FeatureFusionModule(cfg)
        self.mask_fusion    = MaskFusionEncoder(cfg)
        self.decoder    = PCDDecoder(
            embedding_dims= 2048, pcd_samples = 2048)

        try:
            print("Loading Decoder checkpoint...")
            self.decoder.load_state_dict(torch.load(self.cfg['pcd_checkpoints']))
        except:
            print("No Decoder checkpoint found, training from scratch...")

        self.training = True
        # E3RF Loss 모듈 선언
 
    def forward(self, inputs, targetss=None):
        """
        Args:
            rgb: (B, 3, H, W)
            depth: (B, 1, H, W)
        """
        # B, C, H, W = imgs.shape
        imgs = inputs['image'].to(self.device)
        B,C,H,W = imgs.shape
        depth = inputs['depth'].to(self.device)
        list_imgs = [img.squeeze(0) for img in torch.chunk(imgs, B, dim=0)]
        list_depth = torch.chunk(depth, B, dim=0)

        targets = []
        for mask, bbox, label, pcd in zip(
            targetss['masks'], targetss['boxes'], targetss['labels'], targetss['pcd']):
            target = {}
            target['masks'] = mask.unsqueeze(0).to(self.device)
            target['boxes'] = bbox.to(self.device)
            target['labels'] = label.to(self.device)
            target['pcd'] = pcd.to(self.device)
            targets.append(target)
        # print(targets)

        images, targets = self.maskrcnn.transform(images=list_imgs, targets=targets)
        rcnn_features = self.maskrcnn.backbone(images.tensors)
        proposals, proposal_losses = self.maskrcnn.rpn(images, rcnn_features, targets)
        # for p in proposals:
        #     print("p.shape: ", p.shape)
        detections, detector_losses = self.maskrcnn.roi_heads(
            rcnn_features, proposals, images.image_sizes, targets)
        
        # for pro in proposals:
        #     print("pro.shape: ", pro.shape)

        #List[Tuple[int, int]]
        original_image_sizes = [(H, W)] * B
        # print(detections)
        detections = postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        rgbd = torch.cat((imgs, depth.unsqueeze(1)), dim=1)
        rgbd_feats = self.rgbd_input.forward(rgbd)
        
        i_feats = []
        for key in ['0', '1', '2', '3']:
            itp_feat = F.interpolate(
                rcnn_features[key], 
                size=(64, 64), mode='bilinear', align_corners=False)
            i_feats.append(itp_feat)
        i_feats = torch.cat(i_feats, dim=1)
        rgbd_feats = F.interpolate(
            rgbd_feats, size=(64, 64), mode='bilinear', align_corners=False)

        fused_feats = self.feat_fusion.forward(rgbd_feats, i_feats)
        losses['pcd_loss'] = 0.0
        pcd_preds = None
        if len(detections)>0:
            pcd_preds =[]
            
            for m, f, p in zip(detections[0]["masks"], fused_feats, targetss['pcd']):
                print("m.shape: ", m.shape)
                print("f.shape: ", f.shape)
                m = F.interpolate(m.unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False)
                f = F.interpolate(f.unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False)
                print("m.shape: ", m.shape)
                print("f.shape: ", f.shape)
                encoded = self.mask_fusion.forward(m, f)
                pcd_pred = self.decoder.forward(encoded)
                pcd_preds.append(pcd_pred)
                losses['pcd_loss'] += chamfer_distance(pcd_pred.float, p.float())[0]
        
        return losses, detections, pcd_preds