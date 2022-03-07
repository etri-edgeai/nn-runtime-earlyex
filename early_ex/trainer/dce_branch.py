import torch
from . import Trainer
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from early_ex import visualization
from early_ex.utils import *
from early_ex.loss import *


class DCEBranchTrainer(Trainer):
    def __init__(self, model, cfg) -> None:
        super().__init__(model, cfg)
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
        
        # print("--Run initial test drive...")
        # image , label = self.trainset.__getitem__(0)
        # image = image.view(1, image.shape[0], image.shape[1], image.shape[2])
        # print("image.shape: ",image.shape)
        # self.model.eval()
        # with torch.no_grad():
        #     self.model.forward(image)

        self.criterion = nn.CrossEntropyLoss()
        self.ece_loss  = ECELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr = self.cfg['lr'])

        self.scheduler   = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[5, 10, 15, 20], gamma=0.5)

        self.ex_num = self.model.n
        self.num_class = self.cfg['num_class']
        self.proj_size = self.cfg['contra']['projection']
        # self.distance = distances.LpDistance()

        # for n, m in enumerate(self.model.exactly):
        #     m.nn = NN()

    def branch_train(self, epoch):
        self.model.train()
        ex_num = self.model.n
        print("Epoch " + str(epoch) + ':') 
        for n, m in enumerate(self.model.exactly):
            m.cros_path = True
            m.pred_path = False
            m.proj_path = False
            m.near_path = False

        for i, data in enumerate(tqdm(self.train_loader)):
            losses = torch.zeros([ex_num + 1]).to(self.device)
            input = data[0].to(self.device)
            label = data[1].to(self.device)

            pred = self.model.forward(input)

            for n in range(ex_num):
                m = self.model.exactly[n]
                losses[n] = self.criterion(m.logits, label)  
            losses[ex_num] = self.criterion(pred, label)
            self.optimizer.zero_grad()
            loss = torch.sum(losses)
            loss.backward()
            self.optimizer.step()

    def branch_visualize(self, epoch, use_temp = False, test=False):
        print("--Branch Validation...")
        self.model.eval()
        ex_num = self.model.n
        total = 0
        val_loss = torch.zeros(ex_num + 1)
        preds   = []
        labels = torch.zeros(0)

        # initialize list
        for n, m in enumerate(self.model.exactly):
            m.cros_path = True
            m.pred_path = False
            m.proj_path = False
            m.near_path = False

            m.logitss = torch.zeros(0)
            m.scaled = torch.zeros(0)
            m.soft_logits = torch.zeros(0)
            m.soft_scaled = torch.zeros(0)
            m.corrects = 0

        with torch.no_grad():
            val_tbar = tqdm(self.val_loader)
            for i, (input,label) in enumerate(val_tbar):
                input = input.to(self.device)
                label = label
                total += label.size(0)
                predf = self.model.forward(input)
                labels = torch.cat((labels, label), dim=0)

                for n, m in enumerate(self.model.exactly):
                    logits = m.logits.detach().cpu()
                    m.logitss = torch.cat((m.logitss, logits), dim=0)
                    conf, pred = torch.max(logits, dim=1)
                    m.corrects += pred.eq(label).sum().item()


            for n, m in enumerate(self.model.exactly):
                # calculate branch accuracy
                acc = 100.* m.corrects / total
                print("Branch: {}, acc: {:.2f}".format(n, acc))
                np.set_printoptions(suppress=True)
                m.temperature.requires_grad = True
                optimizer = torch.optim.LBFGS([m.temperature],lr=0.005, max_iter=100)
                def eval():
                    optimizer.zero_grad()
                    m.scaled = torch.div(m.logitss, m.temperature.cpu())
                    loss = self.criterion(m.scaled , labels.long())
                    loss.backward(retain_graph = True)
                    return loss
                optimizer.step(eval)
            

                labels_np = labels.numpy()
                logits_np = m.logitss.numpy()
                scaled_np = m.scaled.numpy()
                # print("scaled_np:",scaled_np)
                m.soft_logits = F.softmax(m.logitss, dim=1)
                m.soft_scaled = F.softmax(m.scaled, dim=1)
                soft_logits_np = m.soft_logits.numpy()
                soft_scaled_np = m.soft_scaled.numpy()
                # print("soft_scaled:",soft_scaled_np)

                conf_hist = visualization.ConfidenceHistogram()
                plt1 = conf_hist.plot(
                    scaled_np, labels_np,title="Conf. After #"+str(n))
                name = self.cfg['csv_dir'] + 'Confidence_after_'+str(n)+'.png'
                plt1.savefig(name,bbox_inches='tight')

                plt2 = conf_hist.plot(
                    logits_np, labels_np,title="Conf. Before #"+str(n))
                name = self.cfg['csv_dir'] + 'Confidence_before_'+str(n)+'.png'
                plt2.savefig(name,bbox_inches='tight')

                rel_diagram = visualization.ReliabilityDiagram()
                plt3 = rel_diagram.plot(
                    logits_np,labels_np,title="Reliability Before #"+str(n))
                name = self.cfg['csv_dir'] + 'Reliable_before_'+str(n)+'.png'
                plt3.savefig(name,bbox_inches='tight')

                rel_diagram = visualization.ReliabilityDiagram()
                plt4 = rel_diagram.plot(
                    scaled_np,labels_np,title="Reliability After #"+str(n))
                name = self.cfg['csv_dir'] + 'Reliable_after_'+str(n)+'.png'
                plt4.savefig(name,bbox_inches='tight')

                name = self.cfg['csv_dir'] + 'Confusion Matrix'+str(n)+'.png'
                conf_matrix = visualization.confused(
                    soft_scaled_np, labels_np, self.num_class, name)
                
                name = self.cfg['csv_dir'] + 'RoC_{}'.format(n)
                m.threshold = visualization.roc_curved3(
                    soft_scaled_np, 
                    labels_np, 
                    self.num_class, 
                    name,
                    n, 
                    self.model.n).item()
                print("Threshold #{} has been set to {:.4f}.".format(
                    n, m.threshold))

        torch.cuda.empty_cache() 
        return None

    def branch_calibrate(self):
        print("--Branch Calibration...")
        criterion = nn.CrossEntropyLoss()
        ece_criterion = ECELoss()
        temp_list = []

        labels = torch.zeros(0)
        logits  = []
        scaled = []
        ex_num = self.model.ex_num

        for n in range(ex_num):
            self.model.exactly[n].temp = False
            self.model.exactly[n].temperature.requires_grad=True
            scaled.append(torch.zeros((0)))        
            logits.append(torch.zeros((0)))

        tbar = tqdm(self.val_loader)
        for i, data in enumerate(tbar):       
            input = data[0].to(self.device)
            label = data[1].to(self.device)
            self.model.forward(input)
            labels = torch.cat(
                (labels.to(self.device) ,label), dim=0)

            for n in range(ex_num):
                m = self.model.exactly[n]
                logits[n] = torch.cat((
                    torch.tensor(logits[n]).to(self.device),  
                    m.logits
                    ), dim=0)

        labels = torch.tensor(labels).long()

        for n in tqdm(range(ex_num)):
            m = self.model.exactly[n]
            m.temperature.requires_grad=True

            optimizer = torch.optim.LBFGS(
                [m.temperature], 
                lr=0.1, 
                max_iter=1000)
            print(logits[n].shape)
            print(labels.shape)

            def eval():
                optimizer.zero_grad()
                m.temp = False
                scaled[n] = logits[n] / m.temperature
                loss = criterion(scaled[n] , labels)
                loss.backward(retain_graph = True)
                return loss
            optimizer.step(eval)
  
        soft=[]
        for n in range(self.model.ex_num):
            self.model.exactly[n].temp = True
            soft.append(torch.zeros((0)))
            soft[n] = F.softmax(scaled[n], dim=1)
            soft_np =  soft[n].cpu().detach().numpy()
            
            logits_np = logits[n].cpu().detach().numpy()
            labels_np = labels.cpu().detach().numpy()
            scaled_np = scaled[n].cpu().detach().numpy()
            conf_hist = visualization.ConfidenceHistogram()

            plt_test = conf_hist.plot(
                scaled_np, labels_np,title="Conf. After #"+str(n))
            name = self.cfg['csv_dir'] + self.cfg['path']['mode']\
                + 'conf_histogram_test_after_'+str(n)+'.png'
            plt_test.savefig(name, bbox_inches='tight')

            plt_test = conf_hist.plot(
                logits_np, labels_np,title="Conf. Before #"+str(n))
            name = self.cfg['csv_dir'] + self.cfg['path']['mode']\
                + 'conf_histogram_test_before_'+str(n)+'.png'
            plt_test.savefig(name,bbox_inches='tight')


            rel_diagram = visualization.ReliabilityDiagram()
            plt_test_2 = rel_diagram.plot(
                logits_np,labels_np,title="Reliability Before #"+str(n))
            name = self.cfg['csv_dir'] + self.cfg['path']['mode']\
                + 'rel_diagram_test_before_'+str(n)+'.png'
            plt_test_2.savefig(name,bbox_inches='tight')

            rel_diagram = visualization.ReliabilityDiagram()
            plt_test_2 = rel_diagram.plot(
                scaled_np,labels_np,title="Reliability After #"+str(n))
            name = self.cfg['csv_dir'] + self.cfg['path']['mode']\
                + 'rel_diagram_test_after_'+str(n)+'.png'
            plt_test_2.savefig(name,bbox_inches='tight')

        print("<<<BEFORE Calibration>>>")
        for n in range (0, self.model.ex_num):
            m.temp = True
            print("[{}] NLL Loss: {:.4f}, ECE Loss: {:.4f}".format(
                n, 
                criterion(logits[n], labels).item(), 
                ece_criterion(logits[n],labels).item())
                )

        print("<<<AFTER Calibration>>>")
        for n in range (0, self.model.ex_num):
            print("[{}] NLL Loss: {:.4f}, ECE Loss: {:.4f}".format(
                n, 
                criterion(scaled[n], labels).item(), 
                ece_criterion(scaled[n],labels).item())
                )

        torch.save(self.model.backbone.state_dict(), self.cfg['model_path'])
        print("model saved...")

    
    def branch_test(self):
        self.model.eval()
        acc , total = 0 , 0
        test_tbar = tqdm(self.test_loader)
        self.device = self.cfg['test_device']
        self.model.to(self.device)

        for n, m in enumerate(self.model.exactly):
            m.cros_path = True
            m.pred_path = True
            m.proj_path = False
            m.near_path = False
            m.corrects = 0
        self.model.exit_count = torch.zeros(self.model.n+1, dtype=torch.int)

        with torch.no_grad():
            for (i, data) in enumerate(test_tbar):
                input = data[0].to(self.device)
                label = data[1]
                total += input.shape[0]
                pred = self.model.forward(input)
                _ , pred = torch.max(pred, 1)
                pred = pred.to(self.device)
                acc += pred.eq(label).sum().item()
                test_tbar.set_description("total: {}, correct:{}".format(total, acc))
            print("accuracy: ", acc/ total)
        print(self.model.exit_count)
