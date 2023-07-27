import json
import os

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
import yaml
from albumentations.pytorch import ToTensorV2
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

def custom_collate_fn(batch):
    # Load Images and Depth Maps
    imgs = [target["image"].unsqueeze(0) for target in batch]
    imgs = [torch.cat(imgs,dim=0)]
    depth = [target["depth"] for target in batch]

    # Load Bounding Boxes, Masks, Point Clouds
    masks = [target["mask"] for target in batch]
    pcd = [target["pcd"] for target in batch]
    pcd_center = [target["pcd_center"] for target in batch]
    pcd_radius = [target["pcd_radius"] for target in batch]    
    bboxes_list = [target["bboxes"] for target in batch]
    bboxes = [[torch.FloatTensor(box[:4]) for box in boxes] \
            for boxes in bboxes_list]    
    label = [[torch.tensor(box[4], dtype=torch.int64)
            for box in boxes] for boxes in bboxes_list]

    # Return Images and Targets
    return imgs, {
        "boxes":      bboxes, 
        "masks":      masks, 
        "labels":     label,
        "pcd":        pcd,
        "pcd_center": pcd_center,
        "pcd_radius": pcd_radius,
        "depth":      depth
        }

    
class ShapeNetRenderedDataset(Dataset):
    # Load Dataset
    def __init__(self, cfg, dataset_type="train"):
        # Load Config
        self.cfg        = cfg
        self.img_size   = cfg['0_img_size']
        self.root       = cfg['0_dataset_dir']
        
        # Load Dataset
        path = cfg['0_test_json'] \
            if dataset_type == "test" or dataset_type == "val"\
                  else cfg['0_train_json']
        self.dataset = json.load(open(path, 'r'))    

    def __len__(self):
        # Return Dataset Length
        return len(self.dataset["images"])

    def __getitem__(self, idx):
        # Load Image
        images      = self.dataset["images"][idx]
        annotations = self.dataset["annotations"][idx]
        label       = self.dataset["annotations"][idx]["category_id"]

        # Load Paths
        rgb_path    = images["rgb_path"]
        depth_path  = images["depth_path"]
        seg_path    = images["seg_path"]
        pcd_path    = images["pcd_path"]
        
        # Load Point Cloud
        pcd         = torch.load(pcd_path).squeeze(0).transpose(1, 0)

        # Normalize Point Cloud
        centroid    = torch.mean(pcd, 0)
        pcd         = pcd - centroid
        radius      = torch.max(torch.norm(pcd, 1))
        pcd         = pcd / radius
        
        # Load Image and Segmentation Mask
        img = cv2.imread(rgb_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        seg = cv2.imread(seg_path)
        seg = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)
        _, seg = cv2.threshold(seg, 127, 1, cv2.THRESH_BINARY)

        # Load Bounding Box
        x = annotations["bbox"][0]
        y = annotations["bbox"][1]
        w = annotations["bbox"][2]
        h = annotations["bbox"][3]
        bbox = [[x, y, x+w-1, y+h-1, label]]

        # Transform Image and Mask       
        img_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(
                (self.img_size, self.img_size), antialias=True),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ])
        img = img_transforms(img)

        # Transform Segmentation Mask
        mask_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((self.img_size, self.img_size)
                                          ,antialias=True)])
        mask = mask_transforms(seg)
        min_value = mask.min().item()
        max_value = mask.max().item()
        mask = (mask - min_value) / (max_value - min_value)

        # Load Depth Map
        depth = torch.load(depth_path).squeeze(0).transpose(1, 0)       

        # Create Target Dictionary
        targets= {}
        targets["image"]    = img
        targets["mask"]     = [mask]
        targets["bboxes"]   = bbox
        targets["depth"]    = depth
        targets["mean"]     = np.array(img).mean(axis=(0, 1))
        targets["std"]      = np.array(img).std(axis=(0, 1))
        targets['pcd_center'] = centroid
        targets['pcd_radius'] = radius
        targets['pcd']      = pcd  
        return targets

if __name__ == '__main__':
    cfg_path = "./config/config.yml"
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    trainDataset = ShapeNetRenderedDataset(cfg, "train")
    trainLoader = DataLoader(
        trainDataset, batch_size=4, shuffle=True,
        num_workers=16, collate_fn=custom_collate_fn)
    
    for i, (targets) in enumerate(trainLoader):
        print(targets)
        break    
