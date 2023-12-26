import torch
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from .focal_loss import FocalLoss
from scipy import ndimage

INF = 1e8

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

class SOLOv1Head(nn.Module):
    def __init__(self,
                 num_classes,  #81 coco datashet
                 in_channels,  # 256 fpn outputs
                 device,
                 seg_feat_channels=256,   #seg feature channels 
                 stacked_convs=7,        #solov1 light set 2, 遵循论文和原版配置文件
                 strides=(8, 8, 16, 32, 32),
                 base_edge_list=(16, 32, 64, 128, 256),
                 scale_ranges=((8, 32), (16, 64), (32, 128), (64, 256), (128, 512)),
                 sigma=0.2,
                 num_grids=[40, 36, 24, 16, 12],
                 cate_down_pos=0,
                 with_deform=False,
                 ins_out_channels=64,  #128
                 ):
        super(SOLOv1Head, self).__init__()
        self.device = device
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
        self.with_deform = with_deform # False
        self.loss_cate = FocalLoss( alpha=0.25, gamma=2.0)
        self.ins_loss_weight = 3.0  #loss_ins['loss_weight']  #3.0
        self._init_layers()

    def _init_layers(self):
        norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
        self.cate_convs = nn.ModuleList()
        self.ins_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels + 2 if i == 0 else self.seg_feat_channels
            self.ins_convs.append(
                nn.Sequential(
                    nn.Conv2d(
                        chn,
                        self.seg_feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        bias=norm_cfg is None),

                    nn.GroupNorm(num_channels=self.seg_feat_channels,
                    num_groups=32),
                    
                    nn.ReLU(inplace=True)
                )
            )

            chn = self.in_channels if i == 0 else self.seg_feat_channels
            self.cate_convs.append(nn.Sequential(
                nn.Conv2d(
                    chn,
                    self.seg_feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    bias=norm_cfg is None),

                nn.GroupNorm(num_channels=self.seg_feat_channels,
                num_groups=32),

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

    def init_weights(self):
        for m in self.ins_convs:
            if isinstance(m, nn.Sequential):
                for con in m:
                    if isinstance(con, nn.Conv2d):
                        nn.init.normal_(con.weight, std=0.01)
                        # normal_init(con, std=0.01)
        for m in self.cate_convs:
            if isinstance(m, nn.Sequential):
                for con in m:
                    if isinstance(con, nn.Conv2d):
                        nn.init.normal_(con.weight, std=0.01)
                        # normal_init(con, std=0.01)

        # bias_ins = bias_init_with_prob(0.01)
        bias_ins = nn.init.constant_(torch.tensor(0.01), 0.01)

        for m in self.solo_ins_list: 
            nn.init.normal_(m.weight, std=0.01)
            # normal_init(m, std=0.01, bias=bias_ins)
        # bias_cate = bias_init_with_prob(0.01)
        bias_cate = nn.init.constant_(torch.tensor(0.01), 0.01)
        # normal_init(self.solo_cate, std=0.01, bias=bias_cate)
        nn.init.normal_(self.solo_cate.weight, std=0.01)

    def forward(self, feats, eval=False):
        new_feats = self.split_feats(feats)
        featmap_sizes = [featmap.size()[-2:] for featmap in new_feats]
        upsampled_size = (featmap_sizes[0][0] * 2, featmap_sizes[0][1] * 2)

        ins_pred, cate_pred = multi_apply(
            self.forward_single, new_feats, \
                list(range(len(self.seg_num_grids))), \
                    eval=eval, upsampled_size=upsampled_size)

        return ins_pred, cate_pred
    
    def split_feats(self, feats):
        f_0 = F.interpolate(feats[0], scale_factor=0.5, mode='bilinear', \
                            align_corners=False, recompute_scale_factor=True)
        f_4 = F.interpolate(feats[4], size=feats[3].shape[-2:], \
                            mode='bilinear', align_corners=False)
        return (f_0, feats[1], feats[2], feats[3], f_4)

    def forward_single(self, x, idx, eval=False, upsampled_size=None):
        ins_feat = x
        cate_feat = x

        # for x_i in x:
        #     print("x_i.shape: ", x_i.shape)
        #     print("x_i",x_i[0,0,:])

        # ins branch
        # concat coord
        x_range = torch.linspace(-1, 1, ins_feat.shape[-1], device=x.device)
        y_range = torch.linspace(-1, 1, ins_feat.shape[-2], device=x.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([ins_feat.shape[0], 1, -1, -1])
        x = x.expand([ins_feat.shape[0], 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        ins_feat = torch.cat([ins_feat, coord_feat], 1)

        for i, ins_layer in enumerate(self.ins_convs):
            ins_feat = ins_layer(ins_feat)

        ins_feat = F.interpolate(ins_feat, scale_factor=2, mode='bilinear')
        ins_pred = self.solo_ins_list[idx](ins_feat)

        # cate branch
        for i, cate_layer in enumerate(self.cate_convs):
            if i == self.cate_down_pos:
                seg_num_grid = self.seg_num_grids[idx]
                cate_feat = F.interpolate(cate_feat, size=seg_num_grid, mode='bilinear')
            cate_feat = cate_layer(cate_feat)

        cate_pred = self.solo_cate(cate_feat)
        
        ins_pred = F.interpolate(
            ins_pred.sigmoid(), size=upsampled_size, mode='bilinear')
        cate_pred = points_nms(
            cate_pred.sigmoid(), kernel=2).permute(0, 2, 3, 1)
        return ins_pred, cate_pred

    def loss(self,
             ins_preds,
             cate_preds,
             gt_bbox_list,
             gt_label_list,
             gt_mask_list):

        device = self.device
        featmap_sizes = [featmap.size()[-2:] for featmap in ins_preds]
        # print(gt_mask_list)
        ins_label_list, cate_label_list, ins_ind_label_list = multi_apply(
            self.solo_target_single, gt_bbox_list, gt_label_list,
            gt_mask_list, featmap_sizes=featmap_sizes)

        # for gt_bbox, gt_label in zip(gt_bbox_list, gt_label_list):
        #     print("gt_bbox: ", gt_bbox)
        #     print("gt_label: ", gt_label)
            
        # for n , i in enumerate(ins_label_list):
        #     for m, ii in enumerate(i):
        #         for mm, iii in enumerate(ii):
        #             print(f"{n},{m},{mm} ins_label_list.shape: ", iii.shape)
        #             print(f"{n},{m},{mm} ins_label_list: ", iii.unique())
	
        # ins
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

        # for ins in ins_labels:
        #     print("ins_lab.shape: ", ins.shape)
        #     if ins.shape[0] != 0:
        #         print("ins_lab ", ins.unique())

        # for ins in ins_preds:
        #     print("ins.shape: ", ins.shape)
        #     if ins.shape[0] != 0:
        #         print("ins ", ins.unique())
        

        ins_ind_labels = [
            torch.cat([ins_ind_labels_level_img.flatten()
                       for ins_ind_labels_level_img in ins_ind_labels_level])
            for ins_ind_labels_level in zip(*ins_ind_label_list)
        ]

        flatten_ins_ind_labels = torch.cat(ins_ind_labels)

        num_ins = flatten_ins_ind_labels.sum()

        # dice loss
        loss_ins = []
        for input, target in zip(ins_preds, ins_labels):
            # print(input.size(), target.size())
            if input.size()[0] == 0:
                continue
            # input = torch.sigmoid(input)
            loss_ins.append(dice_loss(input, target))
        loss_ins = torch.cat(loss_ins).mean()
        loss_ins = loss_ins * self.ins_loss_weight

        # cate
        cate_labels = [
            torch.cat([cate_labels_level_img.flatten()
                       for cate_labels_level_img in cate_labels_level])
            for cate_labels_level in zip(*cate_label_list)
        ]
        flatten_cate_labels = torch.cat(cate_labels)

        cate_preds = [
            cate_pred.permute(0, 2, 3, 1).reshape(-1, self.cate_out_channels)
            for cate_pred in cate_preds
        ]

        flatten_cate_preds = torch.cat(cate_preds)
        # flatten_cate_labels = flatten_cate_labels.long().view(-1,1)
        flatten_cate_labels = F.one_hot(
            flatten_cate_labels.long(), num_classes= flatten_cate_preds.shape[-1]).float()
        # print(flatten_cate_preds.shape, flatten_cate_labels.shape)

        loss_cate = self.loss_cate(flatten_cate_preds, flatten_cate_labels)
        # print("loss_cate: ", loss_cate)
        # print("ins_loss: ", loss_ins)
        return dict(loss_ins=loss_ins, loss_cate=loss_cate)

    def solo_target_single(self, gt_bboxes_raw, gt_labels_raw, gt_masks_raw, \
                           featmap_sizes=None):

        # gt_masks_raw=gt_masks_raw.unsqueeze(0)
        # device = gt_labels_raw.device
        # ins
        gt_areas = torch.sqrt((gt_bboxes_raw[:, 2] - gt_bboxes_raw[:, 0]) * (
            gt_bboxes_raw[:, 3] - gt_bboxes_raw[:, 1]))

        ins_label_list = []
        cate_label_list = []
        ins_ind_label_list = []
        for (lower_bound, upper_bound), stride, featmap_size, num_grid \
                in zip(self.scale_ranges, self.strides, featmap_sizes, self.seg_num_grids):
            ins_label = torch.zeros([
                num_grid ** 2, featmap_size[0], featmap_size[1]], \
                    dtype=torch.uint8, device=self.device)
            cate_label = torch.zeros([num_grid, num_grid], \
                                     dtype=torch.int64, device=self.device)
            ins_ind_label = torch.zeros([num_grid ** 2], \
                                        dtype=torch.bool, device=self.device)
            
            hit_indices = ((gt_areas >= lower_bound) & (gt_areas <= upper_bound)).nonzero().flatten()
            if len(hit_indices) == 0:
                ins_label_list.append(ins_label)
                cate_label_list.append(cate_label)
                ins_ind_label_list.append(ins_ind_label)
                continue
 
            gt_labels = gt_labels_raw[hit_indices]
            gt_masks = gt_masks_raw[hit_indices.cpu().numpy(), ...]
            gt_bboxes = gt_bboxes_raw[hit_indices]

            half_ws = 0.5 * (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * self.sigma
            half_hs = 0.5 * (gt_bboxes[:, 3] - gt_bboxes[:, 1]) * self.sigma

            # mass center
            # gt_masks_pt = torch.from_numpy(gt_masks).to(device=self.device)
            # print("gt_masks_pt.shape",gt_masks_pt.shape)
            # gt_masks_pt = torch.from_numpy(gt_masks).to(device=self.device)
            gt_masks_pt = gt_masks.to(device=self.device).view(-1,gt_masks.shape[-1], gt_masks.shape[-1])
            center_ws, center_hs = center_of_mass(gt_masks_pt)
            valid_mask_flags = gt_masks_pt.sum(dim=-1).sum(dim=-1) > 0
            output_stride = stride / 2
            for seg_mask, gt_label, half_h, half_w, center_h, \
                center_w, valid_mask_flag in zip(gt_masks, gt_labels, \
                    half_hs, half_ws, center_hs, center_ws, valid_mask_flags):

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
                # print("cate_label_",cate_label.unique())
                # ins
                # print("cate_label",cate_label.unique())
                seg_mask = imrescale(seg_mask, scale=1. / output_stride)
                # print("seg_mask",seg_mask.unique())
                # seg_mask = torch.from_numpy(seg_mask).to(device=device)
                for i in range(top, down+1):
                    for j in range(left, right+1):
                        label = int(i * num_grid + j)
                        ins_label[label, :seg_mask.shape[-2], :seg_mask.shape[-1]] = seg_mask
                        # print("ins_label_",ins_label.unique())
                        ins_ind_label[label] = 1.0
                        # print("ins_ind_label_",ins_ind_label.unique())
            ins_label_list.append(ins_label)

            cate_label_list.append(cate_label)
            ins_ind_label_list.append(ins_ind_label)
        return ins_label_list, cate_label_list, ins_ind_label_list

    def get_seg(self, seg_preds, cate_preds, img):
        assert len(seg_preds) == len(cate_preds)
        img_shape = img.shape

        num_levels = len(cate_preds)
        featmap_size = seg_preds[0].size()[-2:]

        result_list = []

        for img_id in range(img_shape[0]):
            cate_pred_list = [
                cate_preds[i][img_id].view(-1,self.cate_out_channels).detach()\
                    for i in range(num_levels)
                    ]
            seg_pred_list = [
                seg_preds[i][img_id].detach() for i in range(num_levels)]
            cate_pred_list = torch.cat(cate_pred_list, dim=0)
            seg_pred_list = torch.cat(seg_pred_list, dim=0)

            result = self.get_seg_single(
                cate_pred_list, seg_pred_list, featmap_size, img_shape)
            result_list.append(result)
        return result_list

    def get_seg_single(self, cate_preds, seg_preds, featmap_size, img_shape):
        assert len(cate_preds) == len(seg_preds)

        # overall info.
        b, c, h, w = img_shape
        upsampled_size_out = (featmap_size[0] * 4, featmap_size[1] * 4)

        # process.
        inds = (cate_preds > 0.1)
        # category scores.
        cate_scores = cate_preds[inds]
        if len(cate_scores) == 0:
            return None
        # category labels.
        inds = inds.nonzero()
        cate_labels = inds[:, 1]

        # strides.
        size_trans = cate_labels.new_tensor(self.seg_num_grids).pow(2).cumsum(0)
        strides = cate_scores.new_ones(size_trans[-1])
        n_stage = len(self.seg_num_grids)
        strides[:size_trans[0]] *= self.strides[0]
        for ind_ in range(1, n_stage):
            strides[size_trans[ind_ - 1]:size_trans[ind_]] *= self.strides[ind_]
        strides = strides[inds[:, 0]]

        # masks.
        seg_preds = seg_preds[inds[:, 0]]
        seg_masks = seg_preds > 0.6
        sum_masks = seg_masks.sum((1, 2)).float()

        # filter.
        keep = sum_masks > strides
        if keep.sum() == 0:
            return None

        seg_masks = seg_masks[keep, ...]
        seg_preds = seg_preds[keep, ...]
        sum_masks = sum_masks[keep]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # maskness
        seg_scores = (seg_preds * seg_masks.float()).sum((1, 2)) / sum_masks
        cate_scores *= seg_scores

        if len(cate_scores) == 0:
            return None

        # sort and keep top nms_pre
        sort_inds = torch.argsort(cate_scores, descending=True)

        if len(sort_inds) > 100:
            sort_inds = sort_inds[:100]
        seg_masks = seg_masks[sort_inds, :, :]
        seg_preds = seg_preds[sort_inds, :, :]
        sum_masks = sum_masks[sort_inds]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        # Matrix NMS
        cate_scores = matrix_nms(seg_masks, cate_labels, cate_scores,
                                 kernel='gaussian', sigma=2.0, sum_masks=sum_masks)

        keep = cate_scores >= 0.06
        if keep.sum() == 0:
            return None
        seg_preds = seg_preds[keep, :, :]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # sort and keep top_k
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > 100:
            sort_inds = sort_inds[:100]
        seg_preds = seg_preds[sort_inds, :, :]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        seg_preds = F.interpolate(seg_preds.unsqueeze(0),
                                  size=upsampled_size_out,
                                  mode='bilinear')[:, :, :h, :w]
        seg_masks = F.interpolate(seg_preds,
                                  size=img_shape[2:],
                                  mode='bilinear').squeeze(0)
        seg_masks = seg_masks > 0.6

        return seg_masks, cate_labels, cate_scores
    
def center_of_mass(bitmasks):
    _, h, w = bitmasks.size()
    ys = torch.arange(0, h, dtype=torch.float32, device=bitmasks.device)
    xs = torch.arange(0, w, dtype=torch.float32, device=bitmasks.device)

    m00 = bitmasks.sum(dim=-1).sum(dim=-1).clamp(min=1e-6)
    m10 = (bitmasks * xs).sum(dim=-1).sum(dim=-1)
    m01 = (bitmasks * ys[:, None]).sum(dim=-1).sum(dim=-1)
    center_x = m10 / m00
    center_y = m01 / m00
    return center_x, center_y

def points_nms(heat, kernel=2):
    # kernel must be 2
    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=1)
    keep = (hmax[:, :, :-1, :-1] == heat).float()
    return heat * keep

def dice_loss(input, target):
    input = input.contiguous().view(input.size()[0], -1)
    target = target.contiguous().view(target.size()[0], -1).float()

    a = torch.sum(input * target, 1)
    b = torch.sum(input * input, 1) + 0.001
    c = torch.sum(target * target, 1) + 0.001
    d = (2 * a) / (b + c)
    return 1-d

def matrix_nms(seg_masks, cate_labels, cate_scores, kernel='gaussian', sigma=2.0, sum_masks=None):
    """Matrix NMS for multi-class masks.

    Args:
        seg_masks (Tensor): shape (n, h, w)
        cate_labels (Tensor): shape (n), mask labels in descending order
        cate_scores (Tensor): shape (n), mask scores in descending order
        kernel (str):  'linear' or 'gauss' 
        sigma (float): std in gaussian method
        sum_masks (Tensor): The sum of seg_masks

    Returns:
        Tensor: cate_scores_update, tensors of shape (n)
    """
    n_samples = len(cate_labels)
    if n_samples == 0:
        return []
    if sum_masks is None:
        sum_masks = seg_masks.sum((1, 2)).float()
    seg_masks = seg_masks.reshape(n_samples, -1).float()
    # inter.
    inter_matrix = torch.mm(seg_masks, seg_masks.transpose(1, 0))
    # union.
    sum_masks_x = sum_masks.expand(n_samples, n_samples)
    # iou.
    iou_matrix = (inter_matrix / (sum_masks_x + sum_masks_x.transpose(1, 0) - inter_matrix)).triu(diagonal=1)
    # label_specific matrix.
    cate_labels_x = cate_labels.expand(n_samples, n_samples)
    label_matrix = (cate_labels_x == cate_labels_x.transpose(1, 0)).float().triu(diagonal=1)

    # IoU compensation
    compensate_iou, _ = (iou_matrix * label_matrix).max(0)
    compensate_iou = compensate_iou.expand(n_samples, n_samples).transpose(1, 0)

    # IoU decay 
    decay_iou = iou_matrix * label_matrix

    # matrix nms
    if kernel == 'gaussian':
        decay_matrix = torch.exp(-1 * sigma * (decay_iou ** 2))
        compensate_matrix = torch.exp(-1 * sigma * (compensate_iou ** 2))
        decay_coefficient, _ = (decay_matrix / compensate_matrix).min(0)
    elif kernel == 'linear':
        decay_matrix = (1-decay_iou)/(1-compensate_iou)
        decay_coefficient, _ = decay_matrix.min(0)
    else:
        raise NotImplementedError

    # update the score.
    cate_scores_update = cate_scores * decay_coefficient
    return cate_scores_update


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
