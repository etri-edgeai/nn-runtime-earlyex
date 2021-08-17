
from numpy.lib.function_base import _calculate_shapes
import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import Conv2d
from operator import attrgetter
from torch.autograd import Variable
import sys
from torch.distributions import Categorical



class ExAct(nn.Module):
    def __init__(self,activation="relu", num_class=10, id=0, exnet=None ,cfg=None ):
        super(ExAct, self).__init__()
        self.id = id
        self.branch = False
        self.branch_uninitialized = True
        self.num_class = num_class
        self.exnet = exnet

        self.activation = nn.ReLU()    
        self.hidden = cfg['branch']['hidden']
        self.size =   cfg['branch']['size'] 
        self.expansion = self.size * self.size
        
        self.tim = 1.0
        self.acc = 1.0
        self.loss = 0.0

        self.sigmoid = nn.Sigmoid()

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden * self.expansion, self.num_class),
            nn.Softmax(dim=1)
            )

        self.threshold = 0.016


    def t_prepare(self, acc):
        self.acc = acc

    def exbranch_setup(self, input):

        batch, channel, width, height = input.shape
        print(input.shape)
        self.refit = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=self.hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.hidden)
            ,nn.AdaptiveAvgPool2d((self.size, self.size))
        ).cuda()
        
    
    def forward(self, x , open=None):
        x = self.activation(x)
    
        if self.branch:
            if self.branch_uninitialized:
                self.exbranch_setup(x)
                self.branch_uninitialized = False

            x0 = self.refit(x)
            x0 = x0.view(x0.size(0), -1)
            self.pred = self.classifier(x0)
            self.p = self.pred
            
            self.conf = -1 * torch.nansum((self.p) * torch.log(self.p),dim=1) / self.num_class
            self.sum_conf = torch.sum(self.conf) 
            self.mean_conf = torch.mean(self.conf)
            if self.exnet.test_mode:
                assert x.shape[0] < 2
                if self.conf[0] < self.threshold:
                    #print("gate: ",self.id," entropy: {:.2f}",self.id, self.conf[0])
                    if not ((self.id + 1) is self.exnet.ex_num):
                        self.exnet.set_exit(self.id + 1 , True)    
        return x

