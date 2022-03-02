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
        self.hidd_size = cfg['contra']['hidden']
        
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
            self.proj = proj.clone()
            if self.near_path:
                dist = self.nn.dist(proj)
                logits = - dist
                scaled = torch.div(logits, self.temperature)
                scaled = F.softmax(scaled, dim=1)
                # print("logits: ",self.logits)
                conf, _  = torch.max(scaled, dim=1)
                # print(self.conf.item())
                if conf.item() > self.threshold:
                    # print("{:.2f} > {:.2f}".format(self.conf.item(), self.threshold))
                    self.exit = True
                    return scaled
        return None




# @torch.jit.script
# def distance_matrix(x, y): #pairwise distance of vectors
#     # y = x if type(y) == type(None) else y
#     p = 2
#     n, d = x.size(0), x.size(1)
#     m = y.size(0)
#     x = x.unsqueeze(1).expand(n, m, d)
#     y = y.unsqueeze(0).expand(n, m, d)
#     dist = torch.pow(x - y, p).sum(2)
#     return dist

# class NN(nn.Module):
#     def __init__(self, X = None, Y = None, p = 2):
#         super(NN, self).__init__()
#         self.p = p

#     def set(self, X, Y):
#         self.train_pts = torch.autograd.Variable(X, requires_grad=False)
#         self.train_label = torch.autograd.Variable(Y, requires_grad=False)

#     def dist(self, x):
#         return distance_matrix(x, self.train_pts) ** (1/2)

#     def forward(self, x):
#         return self.predict(x)

#     def predict(self, x):
#         #if type(self.train_pts) == type(None) or type(self.train_label) == type(None):
#         #    name = self.__class__.__name__
#         #    raise RuntimeError(f"{name} wasn't trained. Need to execute {name}.train() first")
        
#         dist = distance_matrix(x, self.train_pts) ** (1/self.p)
#         labels = torch.argmin(dist, dim=1)
#         return self.train_label[labels]