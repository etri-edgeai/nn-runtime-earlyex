import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

class Body(nn.Module):
    def __init__(
            self,
            cfg,
            num_classes=50, #50 classes
            in_channels=256, #256 fpn outputs
            device='cuda:0',
            seg_feat_channels=384,
            stacked_convs=7,
            sigma=0.2,
            cate_down_pos=0,
            base_edge_list=(16, 32, 64, 128, 256),
            strides=[8, 8, 16, 32, 32],
            scale_ranges=((8, 32), (16, 64), (32, 128), (64, 256), (128, 512)),
            num_grids=[40, 36, 24, 16, 12]):
        super(Body, self).__init__()
        self.num_classes = num_classes
        self.seg_num_grids = num_grids
        self.cate_out_channels = self.num_classes - 1
        self.in_channels = in_channels
        self.seg_feat_channels = seg_feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.sigma = sigma
        self.cate_down_pos = cate_down_pos # 0
        self.base_edge_list = base_edge_list
        self.scale_ranges = scale_ranges
        self.loss_cate = FocalLoss( alpha=0.25, gamma=2.0)
        self.ins_loss_weight = 3.0  #loss_ins['loss_weight']  #3.0

        self.init_layers()
        self.init_weights()

    def init_layers(self):
        norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
        self.cate_convs = nn.ModuleList()
        self.ins_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels + 130 if i == 0 else self.seg_feat_channels
            self.ins_convs.append(
                nn.Sequential(
                    nn.Conv2d(chn, self.seg_feat_channels, 3, stride=1, padding=1, bias=norm_cfg is None),
                    nn.GroupNorm(num_channels=self.seg_feat_channels, num_groups=32), 
                    nn.ReLU(inplace=True)
                )
            )

            chn = self.in_channels if i == 0 else self.seg_feat_channels
            self.cate_convs.append(nn.Sequential(
                nn.Conv2d(chn, self.seg_feat_channels, 3, stride=1, padding=1, bias=norm_cfg is None),
                nn.GroupNorm(num_channels=self.seg_feat_channels, num_groups=32),
                nn.ReLU(inplace=True)
                )
            )


        self.solo_ins_list = nn.ModuleList()
        for seg_num_grid in self.seg_num_grids:
            self.solo_ins_list.append(
                nn.Conv2d(self.seg_feat_channels, seg_num_grid**2, 1)
            )
        self.solo_cate = nn.Conv2d(
            self.seg_feat_channels, self.cate_out_channels, 3, padding=1)

        self.solo_pcd_list = nn.ModuleList()
        for strides , seg_num_grid in zip(reversed(self.strides), self.seg_num_grids):
            self.solo_pcd_list.append(
                nn.Sequential(
                    nn.Conv2d(self.seg_feat_channels, seg_num_grid * seg_num_grid, 1),
                    nn.Flatten(start_dim=2),
                    Permute(0, 2, 1),
                    nn.Conv1d(strides * strides, 2048 * 3, 1),
                )
            )
    
    def init_weights(self):
        for m in self.ins_convs:
            if isinstance(m, nn.Sequential):
                for con in m:
                    if isinstance(con, nn.Conv2d):
                        nn.init.normal_(con.weight, std=0.01)

        for m in self.cate_convs:
            if isinstance(m, nn.Sequential):
                for con in m:
                    if isinstance(con, nn.Conv2d):
                        nn.init.normal_(con.weight, std=0.01)

        bias_ins = nn.init.constant_(torch.tensor(0.01), 0.01)

        for m in self.solo_ins_list: 
            nn.init.normal_(m.weight, std=0.01)
        bias_cate = nn.init.constant_(torch.tensor(0.01), 0.01)
        nn.init.normal_(self.solo_cate.weight, std=0.01)

    def split_feats(self, feats):
        f_0 = F.interpolate(feats[0], scale_factor=0.5, mode='bilinear', \
                            align_corners=False, recompute_scale_factor=True)
        f_4 = F.interpolate(feats[4], size=feats[3].shape[-2:], \
                            mode='bilinear', align_corners=False)
        return (f_0, feats[1], feats[2], feats[3], f_4)        

    def forward(self, feats, rgbd, eval=False):
        new_feats = self.split_feats(feats)
        featmap_sizes = [featmap.size()[-2:] for featmap in new_feats]
        upsampled_size = (featmap_sizes[0][0] * 2, featmap_sizes[0][1] * 2)
        rgbd_feats = [rgbd.clone() for _ in range(len(self.seg_num_grids))]
        ins_pred, cate_pred, pcd_pred = multi_apply(
            self.forward_single, new_feats, \
                list(range(len(self.seg_num_grids))), rgbd_feats, \
                    eval=eval, upsampled_size=upsampled_size)

        return ins_pred, cate_pred, pcd_pred
    

    def forward_single(self, x, idx, rgbd, eval=False, upsampled_size=None):
        # Set meshgrid
        ins_feat = x
        cate_feat = x
        x_range = torch.linspace(-1, 1, ins_feat.shape[-1], device=x.device)
        y_range = torch.linspace(-1, 1, ins_feat.shape[-2], device=x.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([ins_feat.shape[0], 1, -1, -1])
        x = x.expand([ins_feat.shape[0], 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)        

        # Merge RGBD + Coord
        rgbd_feat = F.interpolate(rgbd, size=ins_feat.shape[-2:], mode='bilinear')
        ins_feat = torch.cat([ins_feat, rgbd_feat], 1)
        ins_feat = torch.cat([ins_feat, coord_feat], 1)
        for i, ins_layer in enumerate(self.ins_convs):
            ins_feat = ins_layer(ins_feat)

        # PCD branch
        pcd_pred = self.solo_pcd_list[idx](ins_feat)

        # Instance branch
        ins_feat_ = F.interpolate(ins_feat, scale_factor=2, mode='bilinear')
        ins_pred = self.solo_ins_list[idx](ins_feat_)

        # Category branch
        for i, cate_layer in enumerate(self.cate_convs):
            if i == self.cate_down_pos:
                seg_num_grid = self.seg_num_grids[idx]
                cate_feat = F.interpolate(cate_feat, size=seg_num_grid, mode='bilinear')
            cate_feat = cate_layer(cate_feat)
        cate_pred = self.solo_cate(cate_feat)
        
        # Eval mode
        ins_pred = F.interpolate(
            ins_pred.sigmoid(), size=upsampled_size, mode='bilinear')
        cate_pred = points_nms(
            cate_pred.sigmoid(), kernel=2).permute(0, 2, 3, 1)
        pcd_pred = pcd_pred.view(pcd_pred.shape[0], -1, 3, 2048)               

        return ins_pred, cate_pred, pcd_pred