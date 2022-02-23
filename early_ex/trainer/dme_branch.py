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
        self.ccriterion = nn.CrossEntropyLoss()
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
            m.points = torch.zeros(0, self.proj_size + 1)

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
                embs = torch.cat((m.proj.detach().cpu(), label.view(-1,1).detach().cpu()),dim=1)
                m.points = torch.cat((m.points, embs),dim=0)

                
            self.optimizer.zero_grad()
            loss = torch.sum(train_loss)
            loss.backward()
            self.optimizer.step()

        for n, m in enumerate(self.model.exactly):
            std = torch.zeros(self.num_class, self.proj_size)
            mean = torch.zeros(self.num_class, self.proj_size)

            for y in range(self.num_class):
                vect = m.points[m.points[:,-1] == y][:,:-1]
                mean[y] = torch.mean(vect, dim=0)
                std[y] = torch.std(vect, dim=0)
            
            X = F.normalize(mean.to(self.device), dim=1)
            Y = torch.linspace(0, self.num_class-1, self.num_class)
            m.nn.set(X, Y)

    def metric_valid(self, epoch):
        total = 0
        self.model.eval()
        val_tbar = tqdm(self.val_loader)
        corrects = torch.zeros(self.model.n).to(self.device) 
        far = torch.zeros(self.model.n, 1)

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
                    ## Original NN predict
                    pred = m.nn.predict(proj).int()
                    
                    ## distance
                    #dist = self.distance(proj, m.nn.train_pts)
                    #self.logits = torch.div(-dist, m.temperature)
                    #val_ece_loss[n] = self.ece_loss(self.logits, label).item()
                    #self.logits = F.softmax(self.logits, dim=1)                   
                    #conf, pred = torch.max(self.logits, dim=1)

                    ## distance based on normal distribution
                    # nn, d = proj.size(0), proj.size(1)
                    # mm = m.mean.size(0)
                    # x = proj.unsqueeze(1).expand(n2n, mm, d)
                    # y = m.mean.unsqueeze(0).expand(nn, mm, d)
                    # s = m.std.unsqueeze(0).expand(nn, mm, d)
                    # logits = torch.mean(torch.abs(x - y) / s, dim=2)
                    # logits = F.softmax(-logits / m.temperature.cpu(), dim=1)
                    # conf, pred = torch.max(logits, dim=1)
                    
                    #self.logits = torch.div(-1 , dist)
                    #self.logits = F.softmax(self.logits, dim=1)
                    #conf, pred = torch.max(self.logits, dim=1)
                    #print(conf.shape)
                    #print(conf) 
                    corrects[n] += pred.eq(label.cpu()).sum().item()

            for n, m in enumerate(self.model.exactly):
                acc = 100.* corrects[n] / total
                print("Output acc, temperature: {:.4f}, {:.4f}".format(
                    acc, m.temperature.item()))
                
        torch.cuda.empty_cache()

    def metric_visualize(self):
        total = 0
        self.model.eval()
        val_tbar = tqdm(self.val_loader)
        corrects = torch.zeros(self.model.n)
        labels = torch.zeros(0)
        val_ece_loss = torch.zeros([self.ex_num]).to(self.device)

        for n, m in enumerate(self.model.exactly):
            m.cros_path = False
            m.proj_path = True
            m.near_path = False
            m.logits = torch.zeros(0)
            m.scaled = torch.zeros(0)
            m.soft_logits = torch.zeros(0)
            m.soft_scaled = torch.zeros(0)
        
        with torch.no_grad():
            for i, (input,label) in enumerate(val_tbar):
                input = input.to(self.device)
                label = label.to(self.device)
                pred = self.model.forward(input)
                total += label.size(0)
                labels = torch.cat((labels, label.detach().cpu()), dim=0)
                for n, m in enumerate(self.model.exactly):
                    proj = m.proj.detach().cpu()
                    # nn, dd = proj.size(0), proj.size(1)
                    # mm = m.mean.size(0)
                    # proj = m.proj.detach().cpu()

                    ## distance ver 1
                    dist = distance_matrix(proj, m.nn.train_pts.cpu())
                
                    ## distance ver 2
                    # proj = proj.unsqueeze(1).expand(nn, mm, dd)
                    # mean = m.mean.unsqueeze(0).expand(nn, mm, dd)
                    # stdd = m.std.unsqueeze(0).expand(nn, mm, dd)
                    # logits = torch.mean(torch.abs(proj - mean)/ stdd, dim=2)
                    # logits = F.softmax(-logits / m.temperature.cpu(), dim=1)
                    # logits = -logits / m.temperature.cpu()

                    logits = - dist
                    conf, pred = torch.max(F.softmax(logits, dim=1), dim=1)
                    
                    m.logits = torch.cat((m.logits, logits), dim=0)
                    corrects[n] += pred.eq(label.detach().cpu()).sum().item()

            for n, m in enumerate(self.model.exactly):
                m.temperature.requires_grad = True
                optimizer = torch.optim.LBFGS([m.temperature],lr=0.001, max_iter=100)
                def eval():
                    optimizer.zero_grad()
                    m.scaled = torch.div(m.logits, m.temperature.cpu())
                    # m.scaled = torch.cat((m.scaled, scaled.cpu()),dim=0)
                    loss = self.ccriterion(m.scaled , labels.long())
                    loss.backward(retain_graph = True)
                    return loss
                optimizer.step(eval)
                acc = 100.* corrects[n] / total
                print("Output acc, temperature, ece_loss: {:.4f}, {:.4f}, {:.4f}".format(
                    acc, m.temperature.item(), val_ece_loss[n]))

                labels_np = labels.numpy()
                logits_np = m.logits.numpy()
                scaled_np = m.scaled.numpy()
                
                m.soft_logits = F.softmax(m.logits, dim=1)
                m.soft_scaled = F.softmax(m.scaled, dim=1)
                soft_logits_np = m.soft_logits.numpy()
                soft_scaled_np = m.soft_scaled.numpy()

                conf_hist = visualization.ConfidenceHistogram()
                plt1 = conf_hist.plot(scaled_np, labels_np,title="Conf. After #"+str(n))
                name = self.cfg['csv_dir'] + 'Confidence_after_'+str(n)+'.png'
                plt1.savefig(name,bbox_inches='tight')

                plt2 = conf_hist.plot(logits_np, labels_np,title="Conf. Before #"+str(n))
                name = self.cfg['csv_dir'] + 'Confidence_before_'+str(n)+'.png'
                plt2.savefig(name,bbox_inches='tight')

                rel_diagram = visualization.ReliabilityDiagram()
                plt3 = rel_diagram.plot(logits_np,labels_np,title="Reliability Before #"+str(n))
                name = self.cfg['csv_dir'] + 'Reliable_before_'+str(n)+'.png'
                plt3.savefig(name,bbox_inches='tight')

                rel_diagram = visualization.ReliabilityDiagram()
                plt4 = rel_diagram.plot(scaled_np,labels_np,title="Reliability After #"+str(n))
                name = self.cfg['csv_dir'] + 'Reliable_after_'+str(n)+'.png'
                plt4.savefig(name,bbox_inches='tight')

                name = self.cfg['csv_dir'] + 'Confusion Matrix'+str(n)+'.png'
                conf_matrix = visualization.confused(soft_scaled_np, labels_np, self.num_class, name)
                
                name = self.cfg['csv_dir'] + 'RoC_{}'.format(n)
                m.threshold = visualization.roc_curved(soft_scaled_np, labels_np, self.num_class, name).item() + 0.05
                print("Threshold #{} has been set to {:.4f}.".format(n, m.threshold))
    def metric_threshold(self):
        total = 0
        self.model.eval()
        test_tbar = tqdm(self.train_loader)
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

    def metric_cali(self):
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
                acc, m.temperature.item(), val_ece_loss[n]))

            m.temperature.requires_grad = True
            # optimizer = torch.optim.LBFGS([m.temperature],lr=0.01, max_iter=100)
            optimizer = torch.optim.SGD([m.temperature],lr=0.01)
            def eval():
                optimizer.zero_grad()
                logits = torch.div(- m.dist, m.temperature.cpu())
                scaled = F.softmax(logits, dim=1)
                # m.scaled = torch.cat((m.scaled, scaled.cpu()),dim=0)
                loss = self.ccriterion(scaled , labelz.long())
                loss.backward(retain_graph = True)
                return loss
            optimizer.step(eval)

            labels_np = labelz.numpy()
            logits_np = F.softmax(- m.dist, dim=1).detach().cpu().numpy()
            scaled_np = F.softmax(torch.div(- m.dist, m.temperature.cpu()),dim=1).detach().numpy()

            conf_hist = visualization.ConfidenceHistogram()
            plt_test = conf_hist.plot(scaled_np, labels_np,title="Conf. After #"+str(n))
            name = self.cfg['csv_dir'] + 'conf_histogram_test_after_'+str(n)+'.png'
            plt_test.savefig(name,bbox_inches='tight')

            plt_test = conf_hist.plot(logits_np, labels_np,title="Conf. Before #"+str(n))
            name = self.cfg['csv_dir'] + 'conf_histogram_test_before_'+str(n)+'.png'
            plt_test.savefig(name,bbox_inches='tight')

            rel_diagram = visualization.ReliabilityDiagram()
            plt_test_2 = rel_diagram.plot(logits_np,labels_np,title="Reliability Before #"+str(n))
            name = self.cfg['csv_dir'] + 'rel_diagram_test_before_'+str(n)+'.png'
            plt_test_2.savefig(name,bbox_inches='tight')

            rel_diagram = visualization.ReliabilityDiagram()
            plt_test_2 = rel_diagram.plot(scaled_np,labels_np,title="Reliability After #"+str(n))
            name = self.cfg['csv_dir'] + 'rel_diagram_test_after_'+str(n)+'.png'
            plt_test_2.savefig(name,bbox_inches='tight')

        torch.cuda.empty_cache()




        torch.cuda.empty_cache()

    def metric_test(self):
        self.model.eval()
        acc , total = 0 , 0
        test_tbar = tqdm(self.test_loader)
        
        self.device = self.cfg['test_device']

        self.model.to(self.device)

        for n, m in enumerate(self.model.exactly):
            m.cros_path = False
            m.proj_path = True
            m.near_path = True
            m.nn.train_pts = m.nn.train_pts.cpu()

        self.model.exit_count = torch.zeros(self.model.n+1, dtype=torch.int)
        # self.model.exactly[0].threshold = 0.9
        # self.model.exactly[0].threshold = 0.9        
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

        print(self.model.exit_count)

