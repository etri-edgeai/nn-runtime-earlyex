import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import resnet18, resnet34, resnet50
from .solov1_head import SOLOv1Head
from .focal_loss import FocalLoss
from .fpn import FPN

class SOLOV1(nn.Module):
    def __init__(self,device, num_classes=50):
        super(SOLOV1, self).__init__()
        self.mode = "train"
        self.backbone = resnet18(pretrained=True)

        self.fpn = FPN(upsample_cfg=dict(mode='nearest'))

        self.bbox_head = SOLOv1Head(num_classes=num_classes,
                            in_channels=128,
                            device=device,
                            seg_feat_channels=256,
                            stacked_convs=7,
                            strides=[8, 8, 16, 32, 32],
                            scale_ranges=((1, 96), (48, 192), (96, 384), \
                                          (192, 768), (384, 2048)),
                            num_grids=[40, 36, 24, 16, 12])


    def extract_feat(self, imgs):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(imgs)
        x = self.fpn(x)
        return x
    
    def forward_dummy(self, img):
        """Used for computing network flops.
        See `mmdetection/tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward(self, img, **kwargs):
        if self.mode == 'train':
            return self.forward_train(img, **kwargs)
        else:
            return self.forward_test(img, **kwargs)

    def forward_train(self, img, gt_bboxes, gt_labels, gt_masks):
        x = self.extract_feat(img)
        ins_pred , cate_pred = self.bbox_head(x)
        # for ins in ins_pred:
        #     print("ins.shape:",ins.shape)
        # for cate in cate_pred:
        #     print("cate.shape:",cate.shape)
        losses = self.bbox_head.loss(ins_pred, cate_pred, gt_bboxes, \
                                     gt_labels, gt_masks)
        # seg_result = self.bbox_head.get_seg(ins_pred, cate_pred, img)
        # if len(seg_result) != 0:
        #     print(seg_result)
        return losses

    def forward_test(self, img):
        x = self.extract_feat(img)
        ins_pred , cate_pred = self.bbox_head(x)
        seg_result = self.bbox_head.get_seg(ins_pred, cate_pred, img)
        return seg_result
