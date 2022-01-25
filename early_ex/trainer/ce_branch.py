import torch
from . import Trainer
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from ..model.branch import Branch, Gate
import torch.nn.functional as F
from metric_ex import visualization

class ECELoss(nn.Module):
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(
                bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(
                    avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece

class CEBranchTrainer(Trainer):
    def __init__(self, model, cfg) -> None:
        super().__init__(model, cfg)
        
    def branch_init(self):
        print("--Freeze Backbone Model...")
        for n, m in self.model.backbone.named_parameters():
           m.requires_grad = False

        print("--Replace Activations into ExACT...")
        self.exit_insertion(self.model.backbone)
        print("--Run initial test drive...")
        image , label = self.trainset.__getitem__(0)
        image = image.view(1, image.shape[0], image.shape[1], image.shape[2])
        print("image.shape: ",image.shape)
        
        self.model.backbone.to(self.device)
        self.model.forward(image.to(self.device))
        # self.model.forward(image)
                
        self.optimizer   = torch.optim.Adam(self.model.backbone.parameters(), lr = self.cfg['lr'])
        self.scheduler   = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[20, 40], gamma=0.5)

    def exit_insertion(self, model):
        # print(model)

        for n, module in model.named_children():
            if len(list(module.children())) > 0:
                self.exit_insertion(module)
    
            if isinstance(module, nn.ReLU):
            # if isinstance(module, nn.Conv2d):
                old = getattr(model, n)
                new = nn.Sequential(
                    # Gate(id=self.model.ex_num),
                    old,
                    Branch(
                        id = self.model.ex_num, 
                        num_class = self.model.num_class, 
                        cfg = self.cfg
                        )
                    )
                new[1].cross = True
                new[1].temp  = False
                setattr(model, n, new)
                self.model.exactly.append(new[1])
                self.model.ex_num += 1      
            if isinstance(module, nn.Linear):
                break

    def branch_training(self, epochs = 10):
        print('branch')
        self.train_tbar = tqdm(self.train_loader)
        try:
            for epoch in range(0, epochs):
                self.branch_train(epoch)
                self.scheduler.step()
                self.branch_valid(epoch)

        except KeyboardInterrupt:
            print("Skipping branch training")

    def branch_train(self, epoch):
        self.model.backbone.train()

        print("Epoch " + str(epoch) + ':') 
        ex_num = self.model.ex_num
        for n in range(ex_num):
            self.model.exactly[n].requires_grad=True
            self.model.exactly[n].cross= True
        tbar = tqdm(self.train_loader)
        for i, data in enumerate(tbar):
            losses = torch.zeros([self.model.ex_num+1]).to(self.device)
            input = data[0].to(self.device)
            label = data[1].to(self.device)

            pred = self.model.forward(input)

            for n in range(ex_num):
                m = self.model.exactly[n]
                losses[n] = self.criterion(m.pred, label)  

            self.optimizer.zero_grad()
            loss = torch.sum(losses)
            loss.backward()
            self.optimizer.step()

    def branch_valid(self, epoch, use_temp = False, test=False):
        print("--Branch Validation...")
        # self.model.backbone.eval()
        tbar = tqdm(self.val_loader)

        total = 0
        cnt = 0
            
        corrects = []
        val_loss = []
        confidence  = []
        # logits = []
        preds   = []
        # initialize list
        for n in range(0, self.model.ex_num):
            corrects.append(0.0)
            val_loss.append(0.0)
            # confidence.append(torch.zeros((0)))
            # logits.append(torch.zeros((0)))
            preds.append(torch.zeros((0)))

        with torch.no_grad():
            for i, data in enumerate(tbar):
                input = data[0].to(self.device)
                label = data[1].to(self.device)
                self.model.forward(input)
                total += label.size(0)
                # get branches

                for n in range(0, self.model.ex_num):
                    m = self.model.exactly[n]
                    val_loss[n] += self.criterion(m.pred, label).item()
                    conf, pred = torch.max(m.pred, 1)
                    corrects[n] += pred.eq(label).sum().item()
                    # confidence[n] = torch.cat((confidence[n],   conf.cpu().detach())  , dim=0)
                    # logits[n]   = torch.cat((logits[n],     m.logits.cpu().detach())   , dim=0)
                    # preds[n]    = torch.cat((preds[n],     pred.cpu().detach())     , dim=0)
            
                tbar.set_description('total: {}, c0: {}, c4:{}'.format(total, corrects[0], corrects[-1]))


        for n in range (0, self.model.ex_num):  
            # get branch
            m = self.model.exactly[n]

            # calculate branch accuracy
            acc = 100.* corrects[n] / total
            val = val_loss[n] / total
            print("gate: {}, acc: {:.2f}, val_loss: {:.4f}".format(n, acc, val))
            np.set_printoptions(suppress=True)

            # calculate branch entropy
            typee = "confidence_"
            # if self.model.use_temp:
            #     typee = "scaled_confidence_"
            # x_df = pd.DataFrame(confidence[n].cpu().detach().numpy().round(2))
            # name = self.cfg['csv_dir']+typee+str(n)+".csv"
            # x_df.to_csv(name)

        torch.cuda.empty_cache() 
        return None

    def branch_calibrate(self):
        print("--Branch Calibration...")
        for n, m in self.model.backbone.named_parameters():
            m.requires_grad = True
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
            # ece_losses = torch.zeros([self.model.ex_num+1]).to(self.device)
            # losses = torch.zeros([self.model.ex_num+1]).to(self.device)

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

    
    def open_time_test(self):
        self.model.backbone.eval()
        for i in range(0, self.model.ex_num):
            m = self.model.exactly[i]
            m.gate = True
            m.threshold = 0.9
            m.temp = True

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