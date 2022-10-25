"""Trainer function used to implement LazyNet function"""
import os
from argparse import ArgumentParser
import torch
from torch import nn
from torch.nn import functional as F
from utils import read_yaml, save_checkpoint, FocalLoss
from model.lazy import LazyNet
from data import get_dataset
from tqdm import tqdm
from torch import optim

def parse_args():
    """Parse Argument"""
    parser = ArgumentParser(description='Edge Lazy Inference Prototype')
    parser.add_argument('-c', type=str, default="./cfgs/base.yml")
    return parser.parse_args()

class Trainer():
    """Trainer Class containing train, validation, calibration"""
    def __init__(self, cfg):
        """Init config"""
        self.cfg = cfg
        self.device = torch.device(cfg['device'])
        
        # Get Dataset
        loaders = get_dataset(cfg)
        self.train_loader = loaders[0]
        self.val_loader = loaders[1]
        self.test_loader = loaders[2]

        # Set Model
        self.model = LazyNet(cfg).to(self.device)
        # self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.criterion = FocalLoss().to(self.device)

        self.k_criterion = nn.KLDivLoss(
            reduction="batchmean", log_target=True)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr = cfg['lr'])
        self.best_pred = 0

    def train(self):
        """Train function"""
        self.model.train()
        # Start Training
        for e in range(1, self.cfg['epochs']+1):
            pbar = tqdm(self.train_loader)
            for i, (images, labels) in enumerate(pbar):

                skip_losses = torch.zeros(self.model.lazy_num+1).to(self.device)
                images = images.to(self.device)
                labels = labels.type(torch.LongTensor).to(self.device)

                vals, logits = self.model(images)
                self.optimizer.zero_grad()

                spac_loss = self.criterion(logits, labels)

                for ii, v in enumerate(vals):
                    skip_losses[ii] = self.criterion(v, labels)
                skip_loss = skip_losses.sum()

                # k_loss = self.k_criterion(logits, vals[-1])

                loss = spac_loss + skip_loss
                # loss = spac_loss
                loss.backward()
                self.optimizer.step()
                sk_list = skip_losses.cpu().float().detach().tolist()
                new_list = [round(x ,2) for x in sk_list]
                pbar.set_description(
                    "[{}-{}/{}]loss:{:.1f},{}".format(
                        self.cfg['mode'], e, self.cfg['epochs'],
                        spac_loss, new_list))

            if e % self.cfg['val_epoch'] == 0:
                self.valid(epoch=e)
            if e % self.cfg['save_epoch'] == 0:
                save_checkpoint(self.model, self.cfg, e)

    def valid(self, epoch=0):
        """Validation function for testing LazyNet"""
        self.model.eval()
        pbar = tqdm(self.val_loader)
        # Start validation
        total = 0
        sk_corrects = torch.zeros(self.model.lazy_num+1).to(self.device)
        sp_correct = torch.zeros(1).to(self.device)
        for i, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.type(torch.LongTensor).to(self.device)
            self.optimizer.zero_grad()

            with torch.no_grad():
                sk_logits, sp_logit = self.model(images)
            sp_prob = F.softmax(sp_logit, dim=1)
            conf, sp_pred = torch.max(sp_prob, dim=1)
            sp_correct += sp_pred.eq(labels).sum().item()

            for iter, sk_logit in enumerate(sk_logits):
                sk_prob = F.softmax(sk_logit, dim=1)
                sk_conf, sk_pred = torch.max(sk_prob, dim=1)
                sk_corrects[iter] += sk_pred.eq(labels).sum().int().item()
            total += images.shape[0]

            sk_list = sk_corrects.cpu().detach().tolist()
            new_list = [round(x * 100 / total,2) for x in sk_list]
            

def main():
    """Main function for trainer"""
    args = parse_args()
    cfg = read_yaml(args.c)
    trainer = Trainer(cfg)
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("terminating training")
        save_checkpoint(trainer.model, cfg, 0)

if __name__ == '__main__':
    """Main function"""
    torch.cuda.empty_cache()
    main()

###https://madebyollin.github.io/convnet-calculator/