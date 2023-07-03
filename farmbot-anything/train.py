import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from argparse import ArgumentParser
from fame.utils import read_yaml
def parse_args():
    """Parse Argument"""
    parser = ArgumentParser(description='Farmbot-Anything Model on Edge')
    parser.add_argument('-c', type=str, default="./config/base.yml")
    return parser.parse_args()

class Trainer():
    """Trainer Class containing train, validation, calibration"""
    def __init__(self, cfg):
        """Init config"""
        self.cfg = cfg
        self.device = torch.device(cfg['device'])
        return self
        

    def train(self):
        """Train function"""
        self.model.train()
        # Start Training
        return self

if __name__ == '__main__':
    """Main function"""
    torch.cuda.empty_cache()
    args = parse_args()
    cfg = read_yaml(args.c)
    trainer = Trainer(cfg)
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("terminating training")
        # save_checkpoint(trainer.model, cfg, 0)
