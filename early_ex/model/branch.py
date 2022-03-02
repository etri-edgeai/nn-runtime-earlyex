from re import M
import torch
import torch.nn as nn
import torch.nn.functional as F
# from early_ex.utils import AverageMeter
from early_ex.utils import *


class Branch(nn.Module):
    def __init__( 
        self, 
        num_class=10, 
        id=0, 
        cfg=None,
        input=None):

        super().__init__()
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
        
        self.exit = False
        self.gate = False
        self.cros_path = False
        self.proj_path = False
        self.near_path = False
        self.threshold = 0.8
        self.temperature = nn.Parameter(
            torch.Tensor([1.0]), requires_grad=True)

        X0 = torch.zeros(self.cfg['num_class'], self.proj_size)
        Y0 = torch.linspace(0, self.num_class-1, self.num_class)

        X = torch.autograd.Variable(X0, requires_grad=False)
        Y = torch.autograd.Variable(Y0, requires_grad=False)

        self.nn = NN(X=X, Y=Y, p=2)

        batch, channel, width, height = input.shape

        
        self.transform = nn.Sequential(
            nn.Conv2d(
                in_channels=channel, out_channels=self.chan_size, 
                kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=self.chan_size ),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((self.img_size, self.img_size)),
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(self.flat_size, self.repr_size),
            nn.ReLU()
        )

        # self.transform = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((self.img_size, self.img_size)),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=channel, 
        #                 out_channels=self.chan_size, 
        #                 kernel_size = 1, 
        #                 bias = False
        #                 ),
        #     nn.ReLU(),
        #     nn.Flatten(),
        #     nn.Linear(self.flat_size, self.repr_size),
        #     nn.Sigmoid(),
        #     )
        
        self.project = nn.Sequential(
            nn.Linear(self.repr_size, self.proj_size),
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.repr_size, self.num_class)
        )
        

    #def branch_init(self, input):
    #    batch, channel, width, height = input.shape
    #    print('feature map: ', input.shape)
    #    self.branch_uninitialized = False
    #    self.transform = nn.Sequential(
    #        nn.Conv2d(in_channels=channel, 
    #                    out_channels=self.chan_size, 
    #                    kernel_size = 1, 
    #                    bias = False
    #                    ),
    #        nn.ReLU(),
    #        nn.AdaptiveAvgPool2d((self.img_size, self.img_size)),
    #        nn.Flatten(),
    #        nn.Linear(self.flat_size, self.repr_size),
    #        nn.ReLU(),
    #        )
    #    self.project = nn.Sequential(
    #        nn.Linear(self.repr_size, self.proj_size),
    #    )
    #
    #    self.classifier = nn.Sequential(
    #        nn.Linear(self.repr_size, self.num_class)
    #    )

    def forward(self, x):
        #if self.branch_uninitialized:
        #    self.branch_init(x)

        self.repr = self.transform(x)

        if self.cros_path:
            logits = self.classifier(self.repr)
            scaled = F.softmax(logits / self.temperature, dim=1)
            conf, _ = torch.max(scaled, dim=1)
            if conf.item() > self.threshold:
                self.exit = True
                return scaled

        if self.proj_path:
            repp = self.project(self.repr)
            proj = F.normalize(repp)
            if self.near_path:
                scaled = F.softmax(
                    torch.div(
                        - self.nn.dist(proj), self.temperature), dim=1)
                # print("logits: ",self.logits)
                conf, _  = torch.max(scaled, dim=1)
                # print(self.conf.item())
                if conf.item() > self.threshold:
                    # print("{:.2f} > {:.2f}".format(self.conf.item(), self.threshold))
                    self.exit = True
                    return scaled
            else:
                self.proj = proj.clone()

        return None



