import torch.nn as nn
import torch
from early_ex.model import Model
from early_ex.model.branch import Branch
import torch.nn.functional as F
import copy



class DevourModel(Model):
    def __init__(self, cfg, N=3):
        super(Model, self).__init__()
        self.cfg = cfg
        self.num_class = cfg['num_class']
        self.img_size = cfg['img_size']
        self.head_layer = nn.Sequential()
        self.feats = nn.ModuleList([])
        self.fetc = nn.ModuleList([])
        self.exactly = nn.ModuleList([])
        self.tail_layer = nn.Sequential()
        self.gate  = []
        self.n = N
        self.name = "resnet"

    def start_count(self):
        self.count = []


    def forward_init(self):
        x = torch.randn(3, 3, 1000, 1000)
        print("0. Generating input shape:",x.shape)
        x = self.head_layer(x)
        print("1. After head: {}".format(x.shape))
        for i in range(self.n):
            x = self.feats[i].forward(x)
            k = i+2
            print("{}. After Feat: {}".format(k, x.shape))
            self.exactly[i].forward(x)
        
        for i in range(len(self.fetc)):
            k +=1
            x = self.fetc[i].forward(x)
            print("{}. After Fetc: {}".format(k, x.shape))
        
        b, c, w, h = x.shape
        print("X. Input to Tail layer: ", x.shape)
        features = self.cfg['branch']['feature']
        dropout = 0.5
        for m in self.tail_list:
            name = str(type(m).__name__)
            if "Dropout" in name:
                dropout = m.p
            if "Linear" in name:
                features = m.in_features
                break
        return nn.Sequential(
            nn.Conv2d(
                in_channels=c, out_channels=features, 
                kernel_size=1, bias=False),
            nn.BatchNorm2d(features),
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Dropout(dropout),
            nn.Flatten(),
            nn.Linear(features, self.cfg['num_class']))

    def hunt(self, module):
        for n, m in module.named_children():
            print(n, ' ', type(m).__name__)

    def bite(self, module, start=0, end=0):
        result = []
        counter = 0
        assert end >= start
        print("start: {}, end: {}".format(start, end))
        # print("----------------------------")
        for n, m in module.named_children():
            name = type(m).__name__
            if counter >= start and counter <= end:
                if name == "Linear":
                    result.append(nn.Flatten())
                result.append(copy.deepcopy(m))
                print("idx: *\t",counter, "\t",name)
            else:
                print("idx: \t",counter,"\t",name)
            counter += 1

        return result
    
    def forward(self, x):
        x = self.head_layer(x)
        for i in range(self.n):
            x = self.feats[i].forward(x)
            self.exactly[i].forward(x)
            if self.exactly[i].conf[0] > self.exactly[i].threshold and self.gate[i]:
                return self.exactly[i].pred
        
        for i in range(len(self.fetc)):
            x = self.fetc[i].forward(x)
        x = self.tail_layer(x)      
        return x  

    def devour(self, backbone, name='resnet'):
        self.head_list = []
        self.body_list = []
        self.tail_list = []
        
        ### bite model based on types
        print("----------------------------------------")
        print("Scouting Module...")

        if 'efficientnet' in name:
            e = len(backbone.features)
            head, head_start, head_end = (backbone.features,    0, 0)
            body, body_start, body_end = (backbone.features,    1, e) 
            tail, tail_start, tail_end = (backbone,             1, 2)

        if 'mobilenet' in name:
            e = len(backbone.features)
            head, head_start, head_end = (backbone.features,    0, 0)
            body, body_start, body_end = (backbone.features,    1, e) 
            tail, tail_start, tail_end = (backbone.classifier,  0, 2)

        if 'resnet' in name:
            head, head_start, head_end = (backbone, 0, 3)
            body, body_start, body_end = (backbone, 4, 7) 
            tail, tail_start, tail_end = (backbone, 8, 9)

        if 'inception' in name:
            head, head_start, head_end = (backbone,  0,  6)
            body, body_start, body_end = (backbone,  7, 14) 
            tail, tail_start, tail_end = (backbone, 16, 21)
        
        if 'vgg' in name:
            e = len(backbone.features)
            head, head_start, head_end = (backbone.features,    0, 6)
            body, body_start, body_end = (backbone.features,    7, e) 
            tail, tail_start, tail_end = (backbone.classifier,  0, 6)
        
        if 'regnet' in name:
            e = len(backbone)
            head, head_start, head_end = (backbone,  0, 0)
            body, body_start, body_end = (backbone.trunk_output,  0, 3) 
            tail, tail_start, tail_end = (backbone,  2, 3)

        self.hunt(backbone)
        print("----------------------------------------")

        print('<Head_layer>')
        self.head_list = self.bite(head, start=head_start, end=head_end)
        print("----------------------------------------")

        print('<Body_layer>')
        self.body_list = self.bite(body, start=body_start, end=body_end)
        print("----------------------------------------")

        print('<Tail Layer')
        self.tail_list = self.bite(tail, start=tail_start, end=tail_end)

        self.construct()
        del self.head_list
        del self.body_list
        del self.tail_list
        
    def construct(self):
        ## Create head layers
        self.head_layer = nn.Sequential(*self.head_list)
        
        ## Create body layers
        print('---------------------------------------------------')

        print("split feats={}, using N={} early exits"
              .format(len(self.body_list), self.n))
        N = self.n
        if self.n == 1:
            N += 1
        div = len(self.body_list) / N
        div = int(div)
        print("divide size:",div)
        split_list = lambda test_list, x: [test_list[i:i+x] for i in range(0, len(test_list), x)]
        final_list = split_list(self.body_list, div)
        print("Constructing head-body-tail layers with early exits")
        print("<head layer>")
        print('     || ')
        ## feats layers are body layers with early exits
        
        for x in range(self.n):
            for y in range(len(final_list[x])):
                if y < len(final_list[x])-1:
                    print('[feat layer]')
                else:
                    print('[feat layer] -> [exit #{}]'
                        .format(x))
            print('     || ')
            self.feats.append(nn.Sequential(*final_list[x]))
            self.exactly.append(Branch(cfg=self.cfg))
            self.gate.append(False)
            self.temp = False

        ## fetc layers are extra body layers without early exits
        for x in range(self.n, len(final_list)):
            for y in range(len(final_list[x])):
                print('[fetc layer]')
            self.fetc.append(nn.Sequential(*final_list[x]))
        print('     || ')
        print("<tail layer>")
        print('---------------------------------------------------')
        len(self.feats) , len(self.fetc)
        
        ##create tail layer
        print("creating tail layer")
        self.tail_layer = self.forward_init()
        print("Model Set Complete!")