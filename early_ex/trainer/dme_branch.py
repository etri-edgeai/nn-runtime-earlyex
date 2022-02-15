import torch
from . import Trainer
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from ..model.branch import Branch, Gate
import torch.nn.functional as F
from early_ex import visualization

import pytorch_metric_learning
import pytorch_metric_learning.utils.logging_presets as logging_presets
from pytorch_metric_learning import losses, miners, samplers, testers, trainers
from pytorch_metric_learning.utils import common_functions
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

class DMEBranchTrainer(Trainer):
    def __init__(self, model, cfg) -> None:
        super().__init__(model, cfg)
        
    def branch_init(self):
        print("--Run initial test drive...")
        image , label = self.trainset.__getitem__(0)
        image = image.view(1, image.shape[0], image.shape[1], image.shape[2])
        print("image.shape: ",image.shape)
        self.model.eval()
        self.model.to(self.device)
        self.model.forward(image.to(self.device))

        self.criterion = losses.TripletMarginLoss(margin=0.1)
        self.miner = miners.MultiSimilarityMiner(epsilon=0.1)
        # self.sampler = samplers.MPerClassSampler(self.trainset.targets, m=4, length_before_new_iter=len(train_dataset))
        self.optimizer   = torch.optim.Adam(self.model.parameters(), lr = self.cfg['lr'])
        self.scheduler   = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[5, 10, 15, 20], gamma=0.5)
        print('criterion: Triplet Margin loss')
        print('optimizer: Adam, lr: ', self.cfg['lr'])
        print('scheduler = multistep LR')

    def metric_train(self, epoch):
        print("--Freeze Head and Body Layer...")
        for n, m in self.model.feats.named_parameters():
            m.requires_grad_ = False
        for n, m in self.model.head_layer.named_parameters():
            m.requires_grad_ = False
        for n, m in self.model.fetc.named_parameters():
            m.requires_grad_ = False

        for n, m in self.model.exactly.named_parameters():
            m.requires_grad_ = True
        for n, m in self.model.tail_layer.named_parameters():
            m.requires_grad_ = True

        self.model.train()
        ex_num = self.model.n
        print("Epoch " + str(epoch) + ':') 

        for n in range(ex_num):
            self.model.exactly[n].requires_grad = True
            self.model.exactly[n].cross = False
            self.model.exactly[n].projectt = True
            self.model.exactly[n].gate = False

        for i, data in enumerate(tqdm(self.train_loader)):
            train_loss = torch.zeros([ex_num + 1]).to(self.device)
            input = data[0].to(self.device)
            label = data[1].to(self.device)

            pred = self.model.forward(input)

            for n in range(ex_num):
                m = self.model.exactly[n]
                hard_pairs = self.miner(m.proj, label)
                train_loss[n] = self.criterion(m.proj, label, hard_pairs)

            self.optimizer.zero_grad()
            loss = torch.sum(train_loss)
            loss.backward()
            self.optimizer.step()

    def branch_valid(self, epoch, use_temp = False, test=False):
        print("--Branch Validation...")
        self.model.eval()
        ex_num = self.model.n
        total = 0
        corrects = []
        val_loss = []
        preds   = []

        # initialize list
        for n in range(ex_num + 1):
            corrects.append(0.0)
            val_loss.append(0.0)
            preds.append(torch.zeros((0)))

        with torch.no_grad():
            tbar = tqdm(self.val_loader)
            for i, data in enumerate(tbar):
                input = data[0].to(self.device)
                label = data[1].to(self.device)
                predf = self.model.forward(input)
                total += label.size(0)
                # get branches

                for n in range(ex_num):
                    m = self.model.exactly[n]
                    val_loss[n] += self.criterion(m.proj, label).item()

        for n in range (ex_num):  
            # get branch
            m = self.model.exactly[n]

            # calculate branch accuracy
            acc = 100.* corrects[n] / total
            val = val_loss[n] / total
            print("Branch: {}, val_loss: {:.4f}".format(n, val))
            np.set_printoptions(suppress=True)
        return None
