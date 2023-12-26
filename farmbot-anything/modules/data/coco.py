import os
import cv2
import yaml
import json
import torch
import numpy as np
import pandas as pd
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pycocotools.coco import COCO

def custom_collate_fn(batch):
    # targets = zip(*batch)
    imgs = [target["image"] for target in batch]
    segs = [target["mask"] for target in batch]
    bboxes_list = [target["bboxes"] for target in batch]
    bboxes = [[
        torch.FloatTensor(box[:4]) for box in boxes]
            for boxes in bboxes_list]
    label = [[
        torch.tensor(box[4], dtype=torch.int64)
            for box in boxes] for boxes in bboxes_list]
    return imgs, {"boxes": bboxes, "masks": segs, "labels":label}

    
class COCORenderedDataset(Dataset):
    def __init__(self, cfg, dataset_type="train"):
        self.cfg = cfg
        self.img_size = cfg['0_img_size']
        self.root = cfg['1_dataset_dir']
        self.transform = A.Compose([])
        self.mysize = 256
        if dataset_type == "train":
            path = cfg['1_train_json']
            
            self.transform = A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Normalize(),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='coco'), is_check_shapes=False)
        else:
            path = cfg['1_test_json']
            self.transform = A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.Normalize(),
            ToTensorV2()
            ], bbox_params=A.BboxParams(format='coco'), is_check_shapes=False)

        self.dataset = json.load(open(path, 'r'))    

    def __len__(self):
        # return len(self.image_ids)
        return len(self.dataset["images"])

    def __getitem__(self, idx):
        images = self.dataset["images"][idx]
        annotations = self.dataset["annotations"][idx]
        label = self.dataset["annotations"][idx]["category_id"]

        rgb_path = images["rgb_path"]
        seg_path = images["seg_path"]
        img = cv2.imread(rgb_path)
        seg = cv2.imread(seg_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        seg = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)
        _, seg = cv2.threshold(seg, 127, 1, cv2.THRESH_BINARY)
        x = annotations["bbox"][0]
        y = annotations["bbox"][1]
        w = annotations["bbox"][2]
        h = annotations["bbox"][3]
        bbox = [[x, y, w, h, label]]

        targets = self.transform(
            image=np.array(img), mask=np.array(seg), bboxes=bbox)
        
        img = targets["image"]
        seg = targets["mask"]

        bbox = targets["bboxes"]
        bbox = [[bbox[0][0], bbox[0][1], bbox[0][2]+bbox[0][0], bbox[0][3]+bbox[0][1], bbox[0][4]]]
        targets["bboxes"] = bbox
        img_mean = np.array(img).mean(axis=(0, 1))
        img_std = np.array(img).std(axis=(0, 1))
        targets["mean"] = img_mean
        targets["std"] = img_std
        
        return targets

if __name__ == '__main__':
    cfg_path = "./config/rendered.yml"
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    trainDataset = COCORenderedDataset(cfg, "train")
    trainLoader = DataLoader(
        trainDataset, batch_size=4, shuffle=True,
        num_workers=16, collate_fn=custom_collate_fn)
    
    for i, (targets) in enumerate(trainLoader):
        print(targets)
        break    