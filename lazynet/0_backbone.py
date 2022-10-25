from argparse import ArgumentParser
import torch
from torch import nn
from torch.nn import functional as F
from utils import read_yaml, save_checkpoint
from model.backbone import get_backbone
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
        # Init config
        self.cfg = cfg
        self.device = torch.device(cfg['device'])
        
        # Get Dataset
        loaders = get_dataset(cfg)
        self.train_loader = loaders[0]
        self.val_loader = loaders[1]
        self.test_loader = loaders[2]

        # Set Model
        self.model = get_backbone(cfg).to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr = cfg['lr'])
        self.best_pred = 0

    def train(self):
        self.model.train()
        # Start Training
        for e in range(1, self.cfg['epochs']+1):
            pbar = tqdm(self.train_loader)
            for i, (images, labels) in enumerate(pbar):
                images = images.to(self.device)
                labels = labels.type(torch.LongTensor).to(self.device)
                pred = self.model(images)
                self.optimizer.zero_grad()
                loss = self.criterion(pred, labels)
                loss.backward()
                self.optimizer.step()
                pbar.set_description(
                    "[{}-{}/{}]loss:{:.2f}".format(
                        self.cfg['mode'], e, self.cfg['epochs'], loss))

            if e % self.cfg['val_epoch'] == 0:
                self.valid(epoch=e)
            if e % self.cfg['save_epoch'] == 0:
                save_checkpoint(self.model, self.cfg, e)

    def valid(self, epoch=0):
        self.model.eval()
        pbar = tqdm(self.val_loader)
        # Start validation
        total = 0
        correct = torch.zeros(1).to(self.device)
        for i, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.type(torch.LongTensor).to(self.device)
            self.optimizer.zero_grad()

            with torch.no_grad():
                pred = self.model(images)

            conf, pred = torch.max(pred, dim=1)
            correct += pred.eq(labels).sum().item()
            total += images.shape[0]
            pbar.set_description(
                "[{}-{}/{}]acc:{:.1f}%".format(
                    self.cfg['mode'], epoch, self.cfg['epochs'], 
                     correct.item() * 100 / total))

def main():
    """Main function"""
    args = parse_args()
    cfg = read_yaml(args.c)
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()

###https://madebyollin.github.io/convnet-calculator/