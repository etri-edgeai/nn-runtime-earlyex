import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import *
from pytorch3d.loss import chamfer_distance

class Body(nn.Module):
    def __init__(
            self,
            cfg,
            num_classes=50, #50 classes
            in_channels=128, #256 fpn outputs
            device='cuda:0',
            seg_feat_channels=256,
            stacked_convs=7,
            sigma=0.2,
            cate_down_pos=0,
            base_edge_list=(16, 32, 64, 128, 256),
            strides=[8, 8, 16, 32, 32],
            scale_ranges=((8, 32), (16, 64), (32, 128), (64, 256), (128, 512)),
            num_grids=[40, 36, 24, 16, 12]):
        super(Body, self).__init__()
        self.device         = device
        self.num_classes    = num_classes
        self.seg_num_grids  = num_grids
        self.stacked_convs  = stacked_convs
        self.strides        = strides
        self.sigma          = sigma
        self.cate_down_pos  = cate_down_pos # 0
        self.base_edge_list = base_edge_list
        self.scale_ranges   = scale_ranges
        self.loss_cate      = FocalLoss( alpha=0.25, gamma=2.0)
        self.in_channels    = in_channels
        self.ins_loss_weight = 3.0  #loss_ins['loss_weight']  #3.0
        self.cate_out_channels = self.num_classes - 1
        self.seg_feat_channels = seg_feat_channels

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

    def gt_target_single(self, gt_bboxes_raw, gt_labels_raw, gt_masks_raw, 
                           gt_pcd_raw, featmap_sizes=None):

        # gt_pcd_raw = gt_pcd_raw.unsqueeze(0)

        gt_areas = torch.sqrt((gt_bboxes_raw[:, 2] - gt_bboxes_raw[:, 0]) * (
            gt_bboxes_raw[:, 3] - gt_bboxes_raw[:, 1]))

        ins_label_list = []
        cate_label_list = []
        ins_ind_label_list = []
        ins_pcd_list = []
        
        for (lower_bound, upper_bound), stride, featmap_size, num_grid \
                in zip(self.scale_ranges, self.strides, featmap_sizes, self.seg_num_grids):
            ins_label = torch.zeros([
                num_grid ** 2, featmap_size[0], featmap_size[1]], \
                    dtype=torch.uint8, device=self.device)
            cate_label = torch.zeros([num_grid, num_grid], \
                                     dtype=torch.int64, device=self.device)
            ins_ind_label = torch.zeros([num_grid ** 2], \
                                        dtype=torch.bool, device=self.device)
            pcd_label = torch.zeros([
                num_grid ** 2, 3, 2048], \
                    dtype=torch.float32, device=self.device)
            
            hit_indices = ((gt_areas >= lower_bound) & (gt_areas <= upper_bound)).nonzero().flatten()
            if len(hit_indices) == 0:
                ins_label_list.append(ins_label)
                cate_label_list.append(cate_label)
                ins_ind_label_list.append(ins_ind_label)
                ins_pcd_list.append(pcd_label)
                continue
            
            gt_labels = gt_labels_raw[hit_indices]
            gt_masks = gt_masks_raw[hit_indices.cpu().numpy(), ...]
            gt_bboxes = gt_bboxes_raw[hit_indices]
            gt_pcds = gt_pcd_raw[hit_indices]

            half_ws = 0.5 * (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * self.sigma
            half_hs = 0.5 * (gt_bboxes[:, 3] - gt_bboxes[:, 1]) * self.sigma

            gt_masks_pt = gt_masks.to(device=self.device).view(-1,gt_masks.shape[-1], gt_masks.shape[-1])
            center_ws, center_hs = center_of_mass(gt_masks_pt)
            valid_mask_flags = gt_masks_pt.sum(dim=-1).sum(dim=-1) > 0
            output_stride = stride / 2

            for seg_mask, gt_label, gt_pcd, half_h, half_w, center_h, center_w, valid_mask_flag \
                in zip(gt_masks, gt_labels, gt_pcds, half_hs, half_ws, center_hs, center_ws, valid_mask_flags):
                if not valid_mask_flag:
                   continue

                upsampled_size = (featmap_sizes[0][0] * 4, featmap_sizes[0][1] * 4)
                coord_w = int((center_w / upsampled_size[1]) // (1. / num_grid))
                coord_h = int((center_h / upsampled_size[0]) // (1. / num_grid))

                # left, top, right, down
                top_box = max(0, int(((center_h - half_h) / upsampled_size[0]) // (1. / num_grid)))
                down_box = min(num_grid - 1, int(((center_h + half_h) / upsampled_size[0]) // (1. / num_grid)))
                left_box = max(0, int(((center_w - half_w) / upsampled_size[1]) // (1. / num_grid)))
                right_box = min(num_grid - 1, int(((center_w + half_w) / upsampled_size[1]) // (1. / num_grid)))
                top = max(top_box, coord_h-1)
                down = min(down_box, coord_h+1)
                left = max(coord_w-1, left_box)
                right = min(right_box, coord_w+1)

                # squared
                cate_label[top:(down+1), left:(right+1)] = gt_label
                seg_mask = imrescale(seg_mask, scale=1. / output_stride)
                for i in range(top, down+1):
                    for j in range(left, right+1):
                        label = int(i * num_grid + j)
                        ins_label[label, :seg_mask.shape[-2], :seg_mask.shape[-1]] = seg_mask
                        ins_ind_label[label] = 1.0

            ins_label_list.append(ins_label)
            cate_label_list.append(cate_label)
            ins_ind_label_list.append(ins_ind_label)
            ins_pcd_list.append(pcd_label)

        return ins_label_list, cate_label_list, ins_ind_label_list, ins_pcd_list

    def loss(self, ins_preds, cate_preds, pcd_preds, gt_bbox_list, 
             gt_label_list, gt_mask_list, gt_pcd_list):
        device = self.device
        featmap_sizes = [featmap.size()[-2:] for featmap in ins_preds]
        ins_label_list, cate_label_list, ins_ind_label_list, ins_pcd_list = multi_apply(
            self.gt_target_single, gt_bbox_list, gt_label_list,
            gt_mask_list, gt_pcd_list, featmap_sizes=featmap_sizes)

        ins_labels = [
            torch.cat([ins_labels_level_img[ins_ind_labels_level_img, ...] 
                       for ins_labels_level_img, ins_ind_labels_level_img 
                       in zip(ins_labels_level, ins_ind_labels_level)], 0)
                        for ins_labels_level, ins_ind_labels_level 
                        in zip(zip(*ins_label_list), zip(*ins_ind_label_list))]
        ins_preds = [
            torch.cat([ins_preds_level_img[ins_ind_labels_level_img, ...]
                       for ins_preds_level_img, ins_ind_labels_level_img 
                       in zip(ins_preds_level, ins_ind_labels_level)], 0)
                     for ins_preds_level, ins_ind_labels_level 
                     in zip(ins_preds, zip(*ins_ind_label_list))]
        ins_ind_labels = [
            torch.cat([ins_ind_labels_level_img.flatten()
                       for ins_ind_labels_level_img in ins_ind_labels_level])
            for ins_ind_labels_level in zip(*ins_ind_label_list)
        ]
        flatten_ins_ind_labels = torch.cat(ins_ind_labels)
        num_ins = flatten_ins_ind_labels.sum()

        pcd_labels = [
            torch.cat([pcd_labels_level_img[ins_ind_labels_level_img, ...]
                          for pcd_labels_level_img, ins_ind_labels_level_img
                            in zip(pcd_labels_level, ins_ind_labels_level)], 0)
            for pcd_labels_level, ins_ind_labels_level
            in zip(zip(*ins_pcd_list), zip(*ins_ind_label_list))
        ]
        
        pcd_preds = [
            torch.cat([pcd_preds_level_img[ins_ind_labels_level_img, ...]
                            for pcd_preds_level_img, ins_ind_labels_level_img
                                in zip(pcd_preds_level, ins_ind_labels_level)], 0)
            for pcd_preds_level, ins_ind_labels_level
            in zip(pcd_preds, zip(*ins_ind_label_list))
        ]

        # dice loss
        inst_loss = []
        for input, target in zip(ins_preds, ins_labels):
            if input.size()[0] == 0:
                continue
            # input = torch.sigmoid(input)
            inst_loss.append(dice_loss(input, target))
        if len(inst_loss) != 0:
            inst_loss = torch.cat(inst_loss).mean()
        else:
            inst_loss = torch.tensor([0.0]).to(device)
        inst_loss = inst_loss * self.ins_loss_weight

        # cate_loss
        cate_labels = [
            torch.cat([cate_labels_level_img.flatten()
                       for cate_labels_level_img in cate_labels_level])
            for cate_labels_level in zip(*cate_label_list)]
        flatten_cate_labels = torch.cat(cate_labels)
        cate_preds = [
            cate_pred.permute(0, 2, 3, 1).reshape(-1, self.cate_out_channels)
            for cate_pred in cate_preds]
        flatten_cate_preds = torch.cat(cate_preds)
        flatten_cate_labels = F.one_hot(
            flatten_cate_labels.long(), num_classes= flatten_cate_preds.shape[-1]).float()
        cate_loss = self.loss_cate(flatten_cate_preds, flatten_cate_labels)

        # pcd loss
        pcd_loss = []
        for input, target in zip(pcd_preds, pcd_labels):
            if input.size()[0] == 0:
                continue
            pcd_loss.append(chamfer_distance(input, target)[0])
        if len(pcd_loss) != 0:
            pcd_loss = torch.stack(pcd_loss).mean()
        else:
            pcd_loss = torch.tensor([0.0]).to(device)
        pcd_loss = pcd_loss * self.ins_loss_weight

        loss = {
            'inst_loss': inst_loss,
            'cate_loss': cate_loss,
            'pcd_loss': pcd_loss
        }
        return loss