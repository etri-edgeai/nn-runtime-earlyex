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
from early_ex.trainer.ce_branch import CEBranchTrainer
from early_ex.model import Model
import argparse 
import sys

def main():
    print("Branch Trainer v0.5")
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default = "./early_ex/configs/base.yml")
    args = parser.parse_args()
    cfg = config(args.config)
    backbone = get_backbone(cfg)
    try:
        print("loading pre-trained backbone",cfg['backbone_path'])
        backbone.load_state_dict(
            torch.load(cfg['backbone_path']),strict=False)
    except RuntimeError as e:
        print(e)
    except FileNotFoundError as e:
        print(e)
        print("Backbone file not found! Maybe try training it first?")
    model = Model(backbone, cfg)
    trainer = CEBranchTrainer(model, cfg)
    trainer.trainset, trainer.testset = get_dataset(cfg)

    trainer.train_loader = torch.utils.data.DataLoader(
        trainer.trainset, 
        batch_size=cfg['batch_size'], 
        shuffle=True,  
        num_workers=cfg['workers'],
        pin_memory=True) 

    trainer.val_loader = torch.utils.data.DataLoader(
        trainer.testset, 
        batch_size=cfg['batch_size'], 
        shuffle=False,  
        num_workers=cfg['workers'],
        pin_memory=True) 

    try:
        print("loading previous model...")
        trainer.model.backbone.load_state_dict(
            torch.load(trainer.cfg['model_path']), strict=False
            )
    except RuntimeError as e:
        print(e)
    except FileNotFoundError as e:
        print(e)

    trainer.branch_init()

    try:
        for epoch in range(50):
            trainer.branch_train(epoch)
            trainer.scheduler.step()
            trainer.branch_valid(epoch)
    except KeyboardInterrupt:
        print("terminating backbone training")

    try:
        print("calibrating branch using temperature scale")
        trainer.branch_calibrate()

    except KeyboardInterrupt:
        print("terminating backbone training")

    print("saving model to: ",cfg['model_path'])
    torch.save(trainer.model.backbone.state_dict(), cfg['model_path'])


if __name__ == "__main__":
    main()
