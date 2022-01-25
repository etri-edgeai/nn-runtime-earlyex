import torch
import torch.nn as nn
import torch.nn.functional as F
from early_ex.utils import AverageMeter

class EarlyExitError(Exception):
    def __init__ (self, pred):
        self.pred = pred

class Gate(nn.Module):
    def __init__(self, id):
        super(Gate, self).__init__()
        self.exit = False
        self.id = id

    def forward(self, x):
        if self.exit:
            raise EarlyExitError(self.id)
        return x

class Branch(nn.Module):
    def __init__( 
        self, num_class=10, id=0, cfg=None):

        super(Branch, self).__init__()
        self.id = id
        self.branch_uninitialized = True
        self.num_class = num_class
        self.cfg = cfg
        self.channel = cfg['branch']['channel']
        self.size =   cfg['branch']['size']
        self.feature = cfg['branch']['feature'] 
        self.activation = nn.ReLU()
        self.temp = False
        self.d_yes = AverageMeter()
        self.d_no = AverageMeter()
        self.knn_gate = False
        self.exit = False
        self.gate = False
        self.cross = True
        self.projectt = True
        self.threshold = 0.9
        self.temperature = nn.Parameter(
            torch.Tensor([1.5]), 
            requires_grad=True
            )

    def branch_init(self, input):
        batch, channel, width, height = input.shape
        print(input.shape)
        self.shape = self.channel * self.size * self.size
        self.representation = self.cfg['contra']['representation']
        self.projection = self.cfg['contra']['projection']
        self.hidden = self.cfg['contra']['hidden']

        self.transform = nn.Sequential(
            nn.Conv2d(in_channels=channel, 
                        out_channels=self.channel, 
                        kernel_size=1, 
                        bias=False
                        ),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((self.size, self.size)),
            nn.Flatten(),
            nn.Linear(self.shape, self.representation),
            nn.ReLU(),
            ).to(self.cfg['device'])


        middle = self.transform(input)

        self.project = nn.Sequential(
            nn.Linear(middle.shape[1], self.projection),
        ).to(self.cfg['device'])

        self.classifier = nn.Sequential(
            nn.Linear(middle.shape[1], self.num_class)
        ).to(self.cfg['device'])
        
        output = self.classifier(middle)

    def forward(self, x):
        if self.branch_uninitialized:
            self.branch_init(x)
            self.branch_uninitialized = False

        # self.repr = x.view(x.shape[0], -1)
        self.repr = self.transform(x)        
        if self.cross:
            self.logits = self.classifier(self.repr)
            if self.temp:
                self.pred = F.softmax(self.logits/self.temperature, dim=1)
            else:
                self.pred = F.softmax(self.logits, dim=1) 

            self.conf, _ = torch.max(self.pred, 1)

            if self.gate:
                if self.conf.item() > self.threshold:
                    raise EarlyExitError(self.pred)

        if self.projectt:
            self.proj = F.normalize(self.project(self.repr))
        if self.knn_gate:
            # self.pred = self.knn.forward(self.proj)
            # p = self.knn.predict(self.repr.cpu().detach())
            self.pred, conf, count = self.knn.predict(self.repr.cpu().detach())
            if self.gate:            
                if conf[0].item() > self.threshold:
                    self.exit=True

        return x
