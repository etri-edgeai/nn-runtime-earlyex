# -*-coding:UTF-8-*-
import argparse
import torch
from utils.dataset import get_dataset, get_dataloader
from model import get_backbone
from utils.exnet import ExNet
from tqdm import tqdm
import torch.nn as nn
import pandas as pd
class Trainer(object):
    def __init__(self, cfg):
        self.cfg            = cfg
        self.batch_size     = cfg['batch_size']
        self.best           = 0
        self.device         = torch.device(cfg['device'])

        self.train_loader, self.val_loader, self.test_loader, self.trainset, self.testset = get_dataloader(cfg)
        self.backbone =  get_backbone(cfg).cuda()    
        self.model = ExNet(backbone=self.backbone, num_class = cfg['num_class'])

    def backbone_training(self):
        self.criterion   = nn.CrossEntropyLoss()
        self.optimizer   = torch.optim.Adam(self.model.backbone.parameters(), lr = self.cfg['lr'])
        self.scheduler   = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.cfg['backbone_training']['milestone'], gamma=0.5)

        print("Trainer backbone model...")
        try:
            for epoch in range(1, self.cfg['backbone_training']['epochs']):
                self.backbone_train_step(epoch)
                self.scheduler.step()
                self.backbone_validation(epoch)
        except KeyboardInterrupt:
            print("Skipping baseline training")

    def backbone_train_step(self, epoch):
        print("Epoch " + str(epoch) + ':') 
        tbar = tqdm(self.train_loader)
        train_loss = 0.0
        for i, data in enumerate(tbar): 
            input = data[0].to(self.device)
            label = data[1].to(self.device)

            self.optimizer.zero_grad()
            pred = self.model.forward(input)
            pred = pred[0]
            loss = self.criterion(pred, label)
            _, pred = torch.max(pred, 1)
            train_loss += float(loss.item())
            loss.backward()
            self.optimizer.step()
            tbar.set_description('Train loss: %.6f' % (train_loss / ((i + 1) * self.batch_size)))

    def backbone_validation(self, epoch, branch=None):
        tbar = tqdm(self.val_loader, desc='\r')
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
                pred = pred[0]
                conf = pred[1]
                loss = self.criterion(pred, label)
                _, pred = torch.max(pred, 1)

                val_loss += loss.item()
                correct += pred.eq(label).sum().item()
                tbar.set_description('Val loss: %.6f' % (val_loss / ((i + 1)*self.batch_size)))

            acc = 100.*correct/total
            print("epoch: ",epoch, " accuracy: ",acc)
            
            if epoch == 0:
                self.best = acc

            if acc > self.best and epoch > 1:
           
                print("best model saved at ",acc ,", ", self.cfg['backbone_path'])
                torch.save(self.model.backbone.state_dict(), self.cfg['backbone_path'])
                self.best = acc

    def branch_init(self, cfg):
        print("--Freeze Backbone Model...")
        # for n, m in self.model.backbone.named_parameters():
        #    m.requires_grad = False

        print("--Replace Activations into ExACT...")
        self.model.replace(self.model.backbone, cfg)
        print("--Run initial test drive...")

        image , label = self.trainset.__getitem__(0)
        image = image.view(1, image.shape[0], image.shape[1], image.shape[2])
        self.model.backbone.cuda()
        self.model.forward(image.cuda())
                
        self.criterion   = nn.CrossEntropyLoss()
        self.optimizer   = torch.optim.Adam(self.model.backbone.parameters(), lr = cfg['lr'])
        self.scheduler   = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[20, 40], gamma=0.5)

    def branch_tuning(self, epoch):
        print("Epoch " + str(epoch) + ':') 
        tbar = tqdm(self.train_loader)
        for i, data in enumerate(tbar):
            self.optimizer.zero_grad()
            losses = torch.zeros([self.model.ex_num+1]).cuda()
            predd = torch.zeros([self.model.ex_num+1]).cuda()
            input = data[0].to(self.device)
            label = data[1].to(self.device)
            pred, conf = self.model.forward(input)
            
            #t_loss = self.criterion(pred, label)
            
            for n in range(0, self.model.ex_num):
                mm = n
                m = self.model.exactly[mm]
                los = self.criterion(m.pred, label)
                losses[mm] += los
                m.loss = los
            loss = torch.sum(losses)
            #loss += t_loss 
            loss.backward()
            self.optimizer.step()

    def threshold_tuning(self, epoch,  time_list, acc_list):
        self.model.t_tuning = True
        self.model.test_mode = True
        train_loss=0

        self.criterion   = nn.CrossEntropyLoss()
        self.optimizer   = torch.optim.Adam(self.model.backbone.parameters(), lr = self.cfg.lr)
        self.scheduler   = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[10, 20], gamma=0.5)
        corrects = []
        confs   = []

        print("epoch" + str(epoch)+ ":")
        tbar = tqdm(self.train_loader)
        for i, data in enumerate(tbar):
            input = data[0].to(self.device)
            label = data[1].to(self.device)
            self.optimizer.zero_grad()
            pred = self.model.forward(input)
            
            pred = pred[0]
            loss = self.criterion(pred, label)
            _, pred = torch.max(pred, 1)

            loss.backward()
            self.optimizer.step()

    def branch_valid(self):
        tbar = tqdm(self.val_loader)
        total = 0
        cnt = 0
        output_correct= 0        
        corrects = []
        confs   = []
        ent_list = []
        acc_list = []
        val_loss = []
        get_confs = []
        min_ent = []
        max_ent = []
        std_ent = []
        losses = 0
        for n in range(0, self.model.ex_num):
            corrects.append(0.0) 
            confs.append(0.0)
            val_loss.append(0.0)
            get_confs.append(torch.zeros((0)).cuda())
        with torch.no_grad():
            for i, data in enumerate(tbar):
                cnt += 1
                input = data[0].to(self.device)
                label = data[1].to(self.device)

                output = self.model.forward(input)
                total += label.size(0)
              
                out = output[0]
                _ , out = torch.max(out,1)
                output_correct += out.eq(label).sum().item()

                for n in range(0, self.model.ex_num):
                    
                    m = self.model.exactly[n]
                    val_loss[n] +=(self.criterion(m.pred, label).item())
                    _, pred = torch.max(m.pred, 1)
                    
                    corrects[n] += pred.eq(label).sum().item()
                    confs[n] += m.sum_conf.item() 
                    get_confs[n] = torch.cat((get_confs[n], m.conf), dim=0)

            for n in range (0, self.model.ex_num):
                m = self.model.exactly[n]
                c = corrects[n]
                acc = 100.*c/total
                conf = confs[n] / total
                v_loss = val_loss[n] / total

                ent_list.append(conf)
                acc_list.append(acc)   
                max = torch.max(get_confs[n])
                min = torch.min(get_confs[n])
                std = torch.std(get_confs[n])
                x_np = get_confs[n].cpu().numpy() 
                x_df = pd.DataFrame(x_np)
                name ="get_confs["+str(n)+"].csv"
                #x_df.to_csv(name)
                print("gate: {}, acc: {:.2f}, v_loss: {:.6f}, mean_ent: {:.2f},  max_ent: {:.2f}, min_ent: {:.2f}, std_end: {:.2f}".format(n, acc, v_loss, conf, max , min, std))
            accc = 100.*output_correct/total
            print("output: ",accc)
            acc_list.append(accc)
            ent_list.append(0.0)

        return ent_list , acc_list

    def test(self, end=None):
        self.model.consensus = False
        self.model.backbone.eval()
        tbar = tqdm(self.test_loader, desc='\r')
        val_loss = 0.0
        correct = 0
        total = 0
        pred_correct = 0
        cnt = 0
        with torch.no_grad():
            for i, data in enumerate(tbar):
                cnt +=1
                if end != None:
                    if cnt == end:
                        break 
                input = data[0].to(self.device)
                label = data[1].to(self.device)
                
                total += label.size(0)

                predd = self.model.forward(input)
                pred = predd[0]
                conf = predd[1]
                _, pred = torch.max(pred, 1)
                correct += pred.eq(label).sum().item()
                tbar.set_description('correct: {:} total: {:} acc: {:.2f}'.format(correct , total , correct/total))
            acc = 100.*correct/total
            print("test accuracy: ",acc)

    def consensus_test(self, end=None,threshold=None):
        self.model.backbone.eval()
        self.model.consensus = True
        self.model.threshold = 0.50
        if threshold!=None:
            self.model.threshold = threshold
        tbar = tqdm(self.test_loader, desc='\r')
        correct = 0
        total = 0
        with torch.no_grad():
            for i, data in enumerate(tbar):
                input = data[0].to(self.device)
                label = data[1]
                
                total += label.size(0)
                self.model.forward(input)
                pred = self.model._pred
                correct += pred.eq(label).sum().item()
                tbar.set_description('correct: {:} total: {:} acc: {:.2f}'.format(correct , total , correct/total))
            acc = 100.*correct/total
            print("test accuracy: ",acc)

    def eenet_test(self, end=None):
        self.model.eenet = True

        tbar = tqdm(self.test_loader, desc='\r')
        val_loss = 0.0
        correct = 0
        total = 0
        pred_correct = 0
        cnt = 0
        with torch.no_grad():
            for i, data in enumerate(tbar):
                cnt +=1
                if end != None:
                    if cnt == end:
                        break 
                input = data[0].to(self.device)
                label = data[1].to(self.device)
                
                total += label.size(0)

                predd = self.model.forward(input)
                pred = predd[0]
                conf = predd[1]
                _, pred = torch.max(pred, 1)
                correct += pred.eq(label).sum().item()
                tbar.set_description('correct: {:} total: {:} acc: {:.2f}'.format(correct , total , correct/total))
            acc = 100.*correct/total
            print("test accuracy: ",acc)