import torch
import torch.nn as nn
import torch.nn.functional as F
from operator import attrgetter
from utils.exact import Gate, ExAct, ExException
from copy import deepcopy
import warnings

class ExNet(object):
    def __init__(self, backbone, num_class=10):
        super(ExNet, self).__init__()
        self.num_class = num_class
        self.backbone  = backbone

        self.gates = []
        self.exactly = []
        self.exnames = []
        self.test_mode = False
        self.t_tuning = False
        self.theshold = 0.5
        self._pred = None
        self.flip = True
        self.ex_num = 0

        self.pref = 0
        self.conf = 0
        self.consensus = False
        self.eenet = False

    def test_drive(self, img):
        pred = self.forward(img)
        return pred
    
    def get_exlist(self):
        for n, module in self.backbone.named_modules():
            if isinstance(module, ExAct):
                self.exlist.append(str(n))
        return self.exlist

    def set_exit(self, id, exit):
        idd = id
        if id == self.ex_num:
            idd = id-1
        self.gates[idd].exit = exit

    def set_branch(self, id, branch):
        self.exactly[id].branch = branch

    def replace(self, model):
        
        for n, module in model.named_children():
            if len(list(module.children())) > 0:
                self.replace(module)
    
            if isinstance(module, nn.ReLU):
                old = getattr(model, n)
                new = nn.Sequential(
                    Gate(id=self.ex_num, exnet=self),
                    ExAct(id=self.ex_num, num_class=self.num_class, exnet=self),
                    ).cuda()     
                setattr(model, n, new)

                new[1].branch = True
                self.exnames.append(module)
                self.gates.append(new[0])
                self.exactly.append(new[1])
                self.ex_num += 1       

            if isinstance(module, nn.Linear):
                break

    def too(self, device):
        for n, module in self.backbone.named_modules():
            module.to(device)
    
    def forward(self, x):
        result_pred = None
        result_conf = None

        try:
            x = self.backbone(x)
            result_pred = x
            result_conf = torch.tensor([1.0],requires_grad=False)
        except ExException as e:
            results = e.id-1
            result_pred = self.exactly[results].pred 
            result_conf = self.exactly[results].conf
        finally:
            assert result_pred != None
            return result_pred, result_conf
