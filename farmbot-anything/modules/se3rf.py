import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .autoencoder import PCDDecoder
# from .mask_rcnn import maskrcnn_resnet50_fpn_v2
from .solov1 import SOLOV1
from .rgbd_input import RGBDInputModule
from .sfusion import FeatureFusionModule, MaskFusionEncoder
import pytorch3d
from pytorch3d.loss import chamfer_distance

random.seed(0)
torch.manual_seed(0)



class E3RFnet(nn.Module):
    """entire segmentation module"""
    def __init__(self, cfg, device, num_class=49):
        # E3RF 분석 모듈 초기화 설정
        super(E3RFnet, self).__init__()
        self.cfg            = cfg
        self.img_size       = cfg['img_size']
        self.batch_size     = cfg['batch_size']       
        self.num_class      = 49
        self.training       = True

        # E3RF Backbone+FPN 모듈 선언
        self.detect = SOLOV1(device, num_classes=num_class)
        self.detect.train()
        self.detect.training = True        
        try:
            print("Loading SOLOV1 checkpoint...")
            self.detect.load_state_dict(
                torch.load(self.cfg['seg_checkpoints']))
        except:
            print("No SOLOV1 checkpoint found, training from scratch...")
        
        self.rgbd_input = RGBDInputModule(cfg)
        self.feat_fusion    = FeatureFusionModule(cfg)
        self.mask_fusion    = MaskFusionEncoder(cfg)
        self.decoder    = PCDDecoder(
            embedding_dims= 2048, pcd_samples = 2048)

        try:
            print("Loading Decoder checkpoint...")
            self.decoder.load_state_dict(
                torch.load(self.cfg['pcd_checkpoints']))
        except:
            print("No Decoder checkpoint found, training from scratch...")

    def forward_loss(self, img, depth, pcd, masks, bboxes, labels):
        batch, channels, height, width = img.shape
        img_feats = self.detect.extract_feat(img) # img_feats on 5 levels
        inst_pred, cate_pred = self.detect.bbox_head(img_feats) # 5 levels
        depth = F.interpolate(depth.unsqueeze(1), size=(height,width),mode='nearest')
        rgbd = torch.cat((img, depth), dim=1)
        rgbd_feats = self.rgbd_input(rgbd)
        
        img_feats = torch.cat([F.interpolate(
            i, size=(32, 32), mode='bilinear', align_corners=True) \
                for i in img_feats],dim=1)
        
        det_loss = self.detect.bbox_head.loss(
            inst_pred, cate_pred, bboxes, labels, masks)

        seg_masks = self.detect.bbox_head.get_seg(inst_pred, cate_pred, img)
        feats = self.feat_fusion(rgbd_feats, img_feats)

        # if len(seg_masks) == 0:
        #     for b in range(batch):
        #         seg_masks.append(torch.zeros((1, 1, 32, 32)).to(self.device))
        pcd_preds = []
        pcd_loss = []
        loss = {}
        loss['pcd_loss'] = 0.0

        if len(seg_masks) != 0:
            try:         
                for f, s, p in zip(feats, seg_masks, pcd):
                    f = f.unsqueeze(0)
                    idx = torch.argmax(s[2],dim=0).item()
                    # print(idx)
                    s_0 = s[0][idx,:,:].unsqueeze(0).unsqueeze(0).float()
                    # print(f.shape, s_0.shape, s[1].shape, s[2].shape)
                    
                    latent = self.mask_fusion(f, s_0)
                    # print(latent.shape)
                    pcd_pred = self.decoder(latent)
                    # print(pcd_pred.shape, p.shape)
                    p = p.unsqueeze(0)
                    # pcd_pred = pcd_pred[max_index].unsqueeze(0)
                    # print(pcd_pred.shape, p.shape)
                    loss['pcd_loss'] += chamfer_distance(pcd_pred.float(), p.float())[0]
                    # ds.append(pcd_pred)
            except TypeError:
                loss['pcd_loss'] = 0.0
        # pcd_preds = torch.stack(pcd_preds, dim=0).squeeze(1)
        loss['inst_loss'] = det_loss['loss_ins']
        loss['cate_loss'] = det_loss['loss_cate']
        return loss
    
    def forward_inference(self, img, depth):
        assert(img.shape[0] == 1) # batch size must be 1

        img_feats = self.detect.extract_feat(img)
        inst_pred, cate_pred = self.detect.bbox_head(img_feats)
        rgbd = torch.cat((img, depth.unsqueeze(1)), dim=1)
        rgbd_feats = self.rgbd_input(rgbd)
        feats = self.feat_fusion(rgbd_feats, img_feats)            
        seg_masks = self.bbox_head.get_seg(inst_pred, cate_pred, img)

        for seg_mask in seg_masks:
            latent = self.mask_fusion(feats, seg_mask)
            pcd_pred = self.decoder(latent)
        pcd_preds = torch.stack(pcd_preds, dim=0)
        return pcd_preds


    def forward(
        self, 
        img, 
        depth,  
        pcd=None, 
        masks=None, 
        bboxes=None, 
        labels=None, mode='train'):
        if mode == 'test' or mode == 'val':
            return self.forward_inference(img, depth)
        else:
            return self.forward_loss(img, depth, pcd,  masks, bboxes, labels)