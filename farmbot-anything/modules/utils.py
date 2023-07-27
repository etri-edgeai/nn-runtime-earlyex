from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch import Tensor
import torchvision
import torch.nn.functional as F
def resize_boxes(boxes: torch.Tensor, 
                    original_size: List[int], 
                    new_size: List[int]) -> torch.Tensor:
    ratios = [
        torch.tensor(s, dtype=torch.float32, device=boxes.device)
        / torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratio_height, ratio_width = ratios
    xmin, ymin, xmax, ymax = boxes.unbind(1)

    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)

def expand_masks_tracing_scale(M, padding):
    # type: (int, int) -> float
    return torch.tensor(M + 2 * padding).to(torch.float32) / torch.tensor(M).to(torch.float32)


def expand_masks(mask, padding):
    # type: (Tensor, int) -> Tuple[Tensor, float]
    M = mask.shape[-1]
    if torch._C._get_tracing_state():  # could not import is_tracing(), not sure why
        scale = expand_masks_tracing_scale(M, padding)
    else:
        scale = float(M + 2 * padding) / M
    padded_mask = F.pad(mask, (padding,) * 4)
    return padded_mask, scale

def expand_boxes(boxes, scale):
    # type: (Tensor, float) -> Tensor
    # if torchvision._is_tracing():
    #     return _onnx_expand_boxes(boxes, scale)
    w_half = (boxes[:, 2] - boxes[:, 0]) * 0.5
    h_half = (boxes[:, 3] - boxes[:, 1]) * 0.5
    x_c = (boxes[:, 2] + boxes[:, 0]) * 0.5
    y_c = (boxes[:, 3] + boxes[:, 1]) * 0.5

    w_half *= scale
    h_half *= scale

    boxes_exp = torch.zeros_like(boxes)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 2] = x_c + w_half
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 3] = y_c + h_half
    return boxes_exp

def paste_mask_in_image(mask, box, im_h, im_w):
    # type: (Tensor, Tensor, int, int) -> Tensor
    TO_REMOVE = 1
    w = int(box[2] - box[0] + TO_REMOVE)
    h = int(box[3] - box[1] + TO_REMOVE)
    w = max(w, 1)
    h = max(h, 1)

    # Set shape to [batchxCxHxW]
    mask = mask.expand((1, 1, -1, -1))

    # Resize mask
    mask = F.interpolate(mask, size=(h, w), mode="bilinear", align_corners=False)
    mask = mask[0][0]

    im_mask = torch.zeros((im_h, im_w), dtype=mask.dtype, device=mask.device)
    x_0 = max(box[0], 0)
    x_1 = min(box[2] + 1, im_w)
    y_0 = max(box[1], 0)
    y_1 = min(box[3] + 1, im_h)

    im_mask[y_0:y_1, x_0:x_1] = mask[(y_0 - box[1]) : (y_1 - box[1]), (x_0 - box[0]) : (x_1 - box[0])]
    return im_mask

def paste_masks_in_image(masks, boxes, img_shape, padding=1):
    masks, scale = expand_masks(masks, padding=padding)
    boxes = expand_boxes(boxes, scale).to(dtype=torch.int64)
    im_h, im_w = img_shape

    # if torchvision._is_tracing():
    #     return _onnx_paste_masks_in_image_loop(
    #         masks, boxes, torch.scalar_tensor(im_h, dtype=torch.int64), torch.scalar_tensor(im_w, dtype=torch.int64)
    #     )[:, None]
    res = [paste_mask_in_image(m[0], b, im_h, im_w) for m, b in zip(masks, boxes)]
    if len(res) > 0:
        ret = torch.stack(res, dim=0)[:, None]
    else:
        ret = masks.new_empty((0, 1, im_h, im_w))
    return ret

def postprocess(
    result: List[Dict[str, torch.Tensor]],
    image_shapes: List[Tuple[int, int]],
    original_image_sizes: List[Tuple[int, int]],
) -> List[Dict[str, torch.Tensor]]:
    for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
        boxes = pred["boxes"]
        boxes = resize_boxes(boxes, im_s, o_im_s)
        result[i]["boxes"] = boxes
        if "masks" in pred:
            masks = pred["masks"]
            masks = paste_masks_in_image(masks, boxes, o_im_s)
            result[i]["masks"] = masks
        # if "keypoints" in pred:
        #     keypoints = pred["keypoints"]
        #     keypoints = resize_keypoints(keypoints, im_s, o_im_s)
        #     result[i]["keypoints"] = keypoints
    return result
