import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision import datasets
from tqdm import tqdm
import numpy as np
from datetime import datetime
from early_ex.utils import config, get_dataloader, get_dataset
from early_ex.model.backbone import get_backbone
from early_ex.trainer.backbone import BackboneTrainer
from early_ex.model import Model
import argparse 
import sys

def main():
    print("Test backbone ")
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default = "./configs/base.yml")
    args = parser.parse_args()
    cfg = config(args.config)
    # backbone = get_backbone(cfg)
    # model = Model(backbone, cfg)
    # trainer = BackboneTrainer(model, cfg)
    trainset, testset = get_dataset(cfg)

    test_loader = torch.utils.data.DataLoader(
        testset,  
        batch_size= 1, 
        shuffle=False, 
        num_workers=1,
        pin_memory=False)

    try:
        print("loading model for testing..../checkpoints/model_scripted.pt")
        model = torch.jit.load('./checkpoints/model_scripted.pt')
    except RuntimeError as e:
        print(e)
    except FileNotFoundError as e:
        print(e)
        print(" file not found! Maybe try training it first?")


    model.eval()
    tbar = tqdm(test_loader)
    acc = 0 
    total = 0
    device = cfg['test_device']
    model.to(device)

    print(model)

    with torch.no_grad():
        for (i, data) in enumerate(tbar):
            input = data[0].to(device)
            label = data[1].to(device)
            total += input.shape[0]
            pred = model.forward(input)
            _ , pred = torch.max(pred, 1)
            acc += pred.eq(label).sum().item()
            tbar.set_description("total: {}, correct:{}".format(total, acc))
        print(print("accuracy: ", acc/ total))

if __name__ == "__main__":

    main()