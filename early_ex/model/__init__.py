from .branch import Branch, EarlyExitError
import torch.nn as nn
import torch
class Model(nn.Module):
    def __init__(self, backbone, cfg):
        super(Model, self).__init__()
        self.num_class = cfg['num_class']
        self.backbone  = backbone
        self.exactly = []
        self.exnames = []
        self.ex_num = 0
        self.temp = False

    def start_count(self):
        self.count = []

    def forward(self, x):
        try:     
            pred = self.backbone(x)
        except EarlyExitError as b:
            return b.pred
        else:
            return pred
