import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml
import os
from tqdm import tqdm
from . import Trainer
from early_ex.utils import *
class BackboneTrainer(Trainer):

    def __init__(self, model, cfg) -> None:
        super().__init__(model, cfg)        
        print('criterion: Cross entropy loss')
        self.criterion = nn.CrossEntropyLoss()
        print('optimizer: Adam, lr: ',cfg['lr'])
        self.optimizer = torch.optim.Adam(self.model.backbone.parameters(), lr = self.cfg['lr'])
        print('scheduler = multistep LR')
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.cfg['backbone_training']['milestone'], gamma=0.5)

        self.trainset, self.testset = get_dataset(cfg)

        self.train_loader = torch.utils.data.DataLoader(
            self.trainset, 
            batch_size = cfg['batch_size'], 
            shuffle=True,  
            num_workers = cfg['workers'],
            pin_memory = True) 

        self.val_loader = torch.utils.data.DataLoader(
            self.testset, 
            batch_size = cfg['batch_size'], 
            shuffle=False,  
            num_workers = cfg['workers'],
            pin_memory = True) 

    def backbone_train(self, epoch):
        print("Trainer backbone model...")
        self.model.backbone.to(self.device)
        print("Epoch " + str(epoch) + ':')
        self.model.backbone.train() 
        tbar = tqdm(self.train_loader)
        train_loss = 0.0
        for i, data in enumerate(tbar): 
            input = data[0].to(self.device)
            label = data[1].to(self.device)

            self.optimizer.zero_grad()
            pred = self.model.forward(input)
            loss = self.criterion(pred, label)
            _, pred = torch.max(pred, 1)
            loss.backward()
            self.optimizer.step()

    def backbone_valid(self, epoch=1):
        tbar = tqdm(self.val_loader, desc='\r')
        self.model.backbone.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        pred_correct = 0

        with torch.no_grad():
            for i, data in enumerate(tbar):
                input = data[0].to(self.device)
                label = data[1].to(self.device)
                total += label.size(0)
                pred = self.model.forward(input)
                loss = self.criterion(pred, label)
                _, pred = torch.max(pred, 1)
                val_loss += loss.item()
                correct += pred.eq(label).sum().item()
                tbar.set_description('Val loss: %.6f' % (val_loss / ((i + 1)*self.batch_size)))

            acc = 100.*correct/total
            print("epoch: ",epoch, " accuracy: ",acc)


    def open_time_test(self):
        self.model.backbone.eval()
        for i in range(self.model.ex_num):
            m = self.model.exactly[i]
            m.gate = True
            m.knn_gate=False
            m.cross= True

        tbar = tqdm(self.test_loader)
        acc = 0 
        total = 0
        for (i, data) in enumerate(tbar):
            input = data[0].to(self.device)
            label = data[1]
            total += input.shape[0]
            pred = self.model.forward(input)
            _ , pred = torch.max(pred, 1)
            acc += pred.eq(label).sum().item()
            tbar.set_description("total: {}, correct:{}".format(total, acc))
        print(print("accuracy: ", acc/ total))
