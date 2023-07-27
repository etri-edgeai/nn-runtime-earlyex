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
    input, targets = zip(*batch)

    imgs = [imgss["image"].unsqueeze(0) for imgss in input]
    imgs = torch.cat(imgs,dim=0)
    depth = [imgss["depth"].unsqueeze(0) for imgss in input]
    depth = torch.cat(depth,dim=0)

    inputs= {}
    inputs["image"] = imgs
    inputs["depth"] = depth
    targetss = {}
    targetss["masks"] = [torch.cat(targetss["masks"]).squeeze(0) for targetss in targets]
    targetss["boxes"] = [torch.cat(targetss["boxes"]) for targetss in targets]
    targetss["labels"] = [torch.cat(targetss["labels"]) for targetss in targets]
    targetss['pcd'] = [torch.cat(targetss['pcd']) for targetss in targets]
    targetss['pcd_center'] = [torch.cat(targetss['pcd_center']) for targetss in targets]
    targetss['pcd_radius'] = [torch.cat(targetss['pcd_radius']) for targetss in targets]
    return inputs, targetss


    
class ShapeNetRenderedDataset(Dataset):
    def __init__(self, cfg, dataset_type="train"):
        # Load Config
        self.cfg = cfg
        self.img_size   = cfg['0_img_size']
        self.root       = cfg['0_dataset_dir']

        # Load Dataset
        path = cfg['0_test_json'] \
            if dataset_type == "test" or dataset_type == "val"\
                  else cfg['0_train_json']
        self.dataset = json.load(open(path, 'r'))    

    def __len__(self):
        return len(self.dataset["images"])

    def __getitem__(self, idx):
        # Load Images and Depth Maps
        images = self.dataset["images"][idx]
        annotations = self.dataset["annotations"][idx]
        label = self.dataset["annotations"][idx]["category_id"]

        rgb_path = images["rgb_path"]
        depth_path = images["depth_path"]
        seg_path = images["seg_path"]
        pcd_path = images["pcd_path"]
        
        # Load Point Cloud
        pcd = torch.load(pcd_path).squeeze(0).transpose(1, 0)
        centroid = torch.mean(pcd, 0)
        pcd = pcd - centroid
        radius = torch.max(torch.norm(pcd, 1))
        pcd = pcd / radius
        
        # Load Images and Masks
        img = cv2.imread(rgb_path)
        seg = cv2.imread(seg_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        seg = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)
        _, seg = cv2.threshold(seg, 127, 1, cv2.THRESH_BINARY)

        # Transform Images and Masks        
        img_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(
                (self.img_size, self.img_size),antialias=True),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])])
        mask_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(
            (self.img_size, self.img_size), antialias=True)])
        img = img_transforms(img)
        mask = mask_transforms(seg)
        min_value = mask.min().item()
        max_value = mask.max().item()
        mask = (mask - min_value) / (max_value - min_value)

        # Load Bounding Boxes
        x = annotations["bbox"][0]
        y = annotations["bbox"][1]
        w = annotations["bbox"][2]
        h = annotations["bbox"][3]
        bbox = [x, y, x+w, y+h]
        bbox = torch.tensor([bbox[0], bbox[1], bbox[2], bbox[3]]).float()
        bbox /= self.img_size

        # Load Depth Maps
        depth = torch.load(depth_path).squeeze(0).transpose(1, 0)

        input = {}
        input["image"] = img.float()
        input["depth"] = depth
        input["mean"] = np.array(img).mean(axis=(0, 1))
        input["std"] = np.array(img).std(axis=(0, 1))

        targets= {}
        targets["masks"]     = [mask.unsqueeze(0)]
        targets["boxes"]   = [bbox.clone().detach().unsqueeze(0)]
        targets["labels"]   = [torch.tensor(label).unsqueeze(0)]
        targets['pcd']        = [pcd.clone().detach().unsqueeze(0)]
        targets['pcd_center'] = [centroid.clone().detach().unsqueeze(0)]
        targets['pcd_radius'] = [radius.clone().detach().unsqueeze(0)]
        return input, targets

if __name__ == '__main__':
    cfg_path = "./config/rendered.yml"
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    trainDataset = ShapeNetRenderedDataset(cfg, "train")
    trainLoader = DataLoader(
        trainDataset, batch_size=4, shuffle=True,
        num_workers=16, collate_fn=custom_collate_fn)
    
    for i, (targets) in enumerate(trainLoader):
        print(targets)
        break    
