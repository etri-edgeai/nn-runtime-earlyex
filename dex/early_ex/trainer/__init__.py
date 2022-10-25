import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml
import os


class Trainer():
    def __init__(self, model, cfg):
        self.cfg            = cfg
        self.batch_size     = cfg['batch_size']
        self.best           = 0
        self.device         = torch.device(cfg['device'])

        self.model = model
        
        # print('criterion: Cross entropy loss')
        # self.criterion = nn.CrossEntropyLoss()
        # print('optimizer: Adam, lr: ',cfg['lr'])
        # self.optimizer = torch.optim.Adam(self.model.backbone.parameters(), lr = self.cfg['lr'])
        # print('scheduler = multistep LR')
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.cfg['backbone_training']['milestone'], gamma=0.5)
