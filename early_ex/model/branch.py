import torch
import torch.nn as nn
import torch.nn.functional as F
# from early_ex.utils import AverageMeter
from early_ex.utils import *
from pytorch_metric_learning import distances


class Branch(nn.Module):
    def __init__( 
        self, num_class=10, id=0, cfg=None):

        super(Branch, self).__init__()
        # Common configs
        self.id = id
        self.branch_uninitialized = True
        self.num_class = num_class
        self.cfg = cfg
        self.temp = False
        self.device    = cfg['device']

        self.chan_size = cfg['branch']['channel']
        self.img_size  = cfg['branch']['size']
        # self.feat_size = cfg['branch']['feature']
        self.flat_size = self.chan_size * self.img_size * self.img_size

        self.proj_size = cfg['contra']['projection']
        self.repr_size = cfg['contra']['representation']
        self.hidd_size = cfg['contra']['hidden']
        
        self.exit = False
        self.gate = False
        self.cros_path = False
        self.proj_path = False
        self.near_path = False
        self.threshold = 0.8
        self.temperature = nn.Parameter(
            torch.Tensor([1.0]), requires_grad=True)

        self.distance = distances.LpDistance()

        self.nn = NN()

    def branch_init(self, input):
        batch, channel, width, height = input.shape
        print('feature map: ', input.shape)
        self.branch_uninitialized = False
        self.transform = nn.Sequential(
            nn.Conv2d(in_channels=channel, 
                        out_channels=self.chan_size, 
                        kernel_size = 1, 
                        bias = False
                        ),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((self.img_size, self.img_size)),
            nn.Flatten(),
            nn.Linear(self.flat_size, self.repr_size),
            nn.ReLU(),
            )

        self.project = nn.Sequential(
            nn.Linear(self.repr_size, self.proj_size),
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.repr_size, self.num_class)
        )
        
    def forward(self, x):
        if self.branch_uninitialized:
            self.branch_init(x)

        self.repr = self.transform(x)

        if self.cros_path:
            self.logits = self.classifier(self.repr)
            if self.temp:
                self.pred = F.softmax(self.logits / self.temperature, dim=1)
            else:
                self.pred = F.softmax(self.logits, dim=1) 
            self.conf, _ = torch.max(self.pred, dim=1)
            if self.conf.item() > self.threshold:
                self.exit = True

        if self.proj_path:
            middle = self.project(self.repr)
            self.proj = F.normalize(middle)
            
            if self.near_path:
                dist = self.nn.dist(self.proj)
                # print("dist: ", dist)
                logits = - dist
                logits = torch.div(logits, self.temperature)
                self.logits = F.softmax(logits, dim=1)
                # print("logits: ",self.logits)
                self.conf, _  = torch.max(self.logits, dim=1)
                # print(self.conf.item())
                if self.conf.item() > self.threshold:
                    print("{:.2f} > {:.2f}".format(self.conf.item(), self.threshold))
                    self.exit = True
        return x
