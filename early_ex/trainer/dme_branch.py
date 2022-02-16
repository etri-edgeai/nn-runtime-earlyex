import enum
import torch
from . import Trainer
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from ..model.branch import Branch
import torch.nn.functional as F
from early_ex import visualization
from early_ex.utils import *
from early_ex.loss import *
import pytorch_metric_learning
import pytorch_metric_learning.utils.logging_presets as logging_presets
from pytorch_metric_learning import losses, miners, samplers, testers, trainers
from pytorch_metric_learning.utils import common_functions
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

class DMEBranchTrainer(Trainer):
    def __init__(self, model, cfg) -> None:
        super().__init__(model, cfg)

        print('criterion: Triplet Margin loss')
        print('optimizer: Adam, lr: ', self.cfg['lr'])
        print('scheduler = multistep LR')
        for n, m in enumerate(self.model.exactly):
            m.cros_path = False
            m.proj_path = True
            m.near_path = False

        for n, m in self.model.feats.named_parameters():
            m.requires_grad_ = False
        for n, m in self.model.head_layer.named_parameters():
            m.requires_grad_ = False
        for n, m in self.model.fetc.named_parameters():
            m.requires_grad_ = False
        for n, m in self.model.exactly.named_parameters():
            m.requires_grad_ = True
        for n, m in self.model.tail_layer.named_parameters():
            m.requires_grad_ = False

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

        self.test_loader = torch.utils.data.DataLoader(
            self.testset, 
            batch_size = 1, 
            shuffle=False,  
            num_workers=cfg['workers'],
            pin_memory=True) 
        
        print("--Run initial test drive...")
        image , label = self.trainset.__getitem__(0)
        image = image.view(1, image.shape[0], image.shape[1], image.shape[2])
        print("image.shape: ",image.shape)
        self.model.eval()
        with torch.no_grad():
            self.model.forward(image)
        # self.criterion = losses.TripletMarginLoss(margin=0.1)
        self.criterion = BoneLoss()
        # self.miner = miners.MultiSimilarityMiner(epsilon=0.1)
        self.ece_loss = ECELoss()
        self.optimizer   = torch.optim.Adam(
            self.model.parameters(), lr = self.cfg['lr'])

        self.scheduler   = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[5, 10, 15, 20], gamma=0.5)

        self.ex_num = self.model.n
        self.num_class = self.cfg['num_class']
        self.proj_size = self.cfg['contra']['projection']
        self.distance = distances.LpDistance()

        for n, m in enumerate(self.model.exactly):
            m.nn = NN()

    def metric_train(self):
        for n, m in enumerate(self.model.exactly):
            m.cros_path = False
            m.proj_path = True
            m.near_path = False
            m.cnts = torch.zeros(self.num_class).to(self.device)
            m.embs = torch.zeros(self.num_class, self.proj_size).to(self.device)

        train_tbar = tqdm(self.train_loader)

        self.model.train()
        for i, data in enumerate(train_tbar):
            train_loss = torch.zeros([self.ex_num]).to(self.device)
            input = data[0].to(self.device)
            label = data[1].to(self.device)

            pred = self.model.forward(input)

            for n, m in enumerate(self.model.exactly):
                train_loss[n] = self.criterion(
                    m.proj, label, temperature = m.temperature)

                for y in range(self.num_class):
                    split = m.proj[label == y].detach().cuda()
                    m.cnts[y] += split.shape[0]
                    m.embs[y] += torch.sum(split, dim=0).cuda()

            self.optimizer.zero_grad()
            loss = torch.sum(train_loss)
            loss.backward()
            self.optimizer.step()

        for n, m in enumerate(self.model.exactly):
            X = F.normalize(m.embs, dim=1)
            Y = torch.linspace(0, self.num_class-1, self.num_class)
            m.nn.train(X, Y)

    def metric_valid(self, epoch):
        total = 0
        self.model.eval()
        val_tbar = tqdm(self.val_loader)
        corrects = torch.zeros(self.model.n).to(self.device) 
        
        for n, m in enumerate(self.model.exactly):
            m.cros_path = False
            m.proj_path = True
            m.near_path = False
        
        val_ece_loss = torch.zeros([self.ex_num]).to(self.device)

        with torch.no_grad():
            for i, (input, label) in enumerate(val_tbar):
                input = input.to(self.device)
                label = label.to(self.device)
                pred = self.model.forward(input)
                total += label.size(0)

                for n, m in enumerate(self.model.exactly):
                    proj = m.proj.detach().to(self.device) 
                    # pred = m.nn.predict(proj).int().to(self.device) 
                    dist = self.distance(proj, m.nn.train_pts)
                    self.logits = torch.div(-dist, m.temperature)
                    val_ece_loss[n] = self.ece_loss(self.logits, label).item()

                    self.logits = F.softmax(self.logits, dim=1)
                    conf, pred = torch.max(self.logits, dim=1)
                    corrects[n] += pred.eq(label).sum().item()

            for n, m in enumerate(self.model.exactly):
                acc = 100.* corrects[n] / total
                print("Output acc, temperature, ece_loss: {:.4f}, {:.4f}, {:.4f}".format(
                    acc, m.temperature.item(), val_ece_loss[n]/total))
                
        torch.cuda.empty_cache()


    def metric_cali(self, epoch):
        total = 0
        self.model.eval()
        val_tbar = tqdm(self.val_loader)
        corrects = torch.zeros(self.model.n).to(self.device) 
        labelz = torch.zeros(0)
        for n, m in enumerate(self.model.exactly):
            m.cros_path = False
            m.proj_path = True
            m.near_path = False
            m.logits    = torch.zeros(0)
            m.scaled    = torch.zeros(0)
        
        val_ece_loss = torch.zeros([self.ex_num]).to(self.device)

        with torch.no_grad():
            for i, (input, label) in enumerate(val_tbar):
                input = input.to(self.device)
                label = label.to(self.device)
                pred = self.model.forward(input)
                total += label.size(0)

                ## get ground truth labels
                labelz = torch.cat((labelz, label.detach().cpu()), dim=0)

                for n, m in enumerate(self.model.exactly):
                    proj = m.proj.detach().to(self.device) 
                    # pred = m.nn.predict(proj).int().to(self.device) 
                    dist = self.distance(proj, m.nn.train_pts)

                    ## get original logits
                    logits = F.softmax(- dist, dim=1).detach().cpu()
                    m.logits = torch.cat((m.logits, logits), dim=0)

                    ## get scaled logits
                    self.logits = torch.div(-dist, m.temperature)
                    val_ece_loss[n] = self.ece_loss(self.logits, label).item()
                    self.logits = F.softmax(self.logits, dim=1)
                    m.scaled = torch.cat((m.scaled, self.logits.detach().cpu()),dim=0)

                    conf, pred = torch.max(self.logits, dim=1)
                    corrects[n] += pred.eq(label).sum().item()

            for n, m in enumerate(self.model.exactly):
                acc = 100.* corrects[n] / total
                print("Output acc, temperature, ece_loss: {:.4f}, {:.4f}, {:.4f}".format(
                    acc, m.temperature.item(), val_ece_loss[n]/total))
                
                label_np = labelz.numpy()
                logit_np = m.logits.numpy()
                scale_np = m.scaled.numpy()
                conf_hist = visualization.ConfidenceHistogram()


        torch.cuda.empty_cache()


    def metric_test(self):
        self.model.eval()
        acc , total = 0 , 0
        test_tbar = tqdm(self.test_loader)

        for n, m in enumerate(self.model.exactly):
            m.cros_path = False
            m.proj_path = True
            m.near_path = True

        with torch.no_grad():
            for (i, data) in enumerate(test_tbar):
                input = data[0].to(self.device)
                label = data[1].to(self.device)
                total += input.shape[0]
                pred = self.model.forward(input)
                _ , pred = torch.max(pred, 1)
                acc += pred.eq(label).sum().item()
                test_tbar.set_description("total: {}, correct:{}".format(total, acc))
            print("accuracy: ", acc/ total)


