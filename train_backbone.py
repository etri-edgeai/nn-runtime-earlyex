import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision import datasets
from tqdm import tqdm
import numpy as np
from datetime import datetime
from metric_ex.utils import config, get_dataloader, get_dataset
from metric_ex.model.backbone import get_backbone
from metric_ex.trainer.backbone import BackboneTrainer
from metric_ex.model import Model
import argparse 

def main():
    print("Backbone Trainer v 1.0")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', 
        type = str, 
        default = "./configs/cifar10/resnet18_ce.yml"
        )
    args = parser.parse_args()
    cfg = config(args.config)
    backbone = get_backbone(cfg)
    try:
        print(cfg['backbone_path'])
        backbone.load_state_dict(
            torch.load(cfg['backbone_path']), strict=False)
    except RuntimeError as e:
        print(e)
    except FileNotFoundError as e:
        print(e)
    model = Model(backbone, cfg)
    trainer = BackboneTrainer(model, cfg)
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
        for epoch in range(cfg['backbone_training']['epochs']):
            trainer.backbone_train(epoch)
            trainer.scheduler.step()
            trainer.backbone_valid(epoch)
    except KeyboardInterrupt:
        print("terminating backbone training")

    torch.save(trainer.model.backbone.state_dict(), cfg['backbone_path'])


if __name__ == "__main__":
    main()