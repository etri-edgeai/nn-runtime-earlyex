import torch.nn as nn
import torch
from early_ex.model import Model
# from early_ex.model.branch import Branch
import torch.nn.functional as F
import copy
    
class Branch(nn.Module):
    def __init__( 
        self, cfg=None):

        super(Branch, self).__init__()
        self.branch_uninitialized = True
        self.num_class = cfg['num_class']
        self.cfg = cfg
        self.channel = cfg['branch']['channel']
        self.size =   cfg['branch']['size']
        self.feature = cfg['branch']['feature'] 
        self.temp = False
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
        self.branch_uninitialized = False
        batch, channel, width, height = input.shape
        # print(input.shape)
        self.shape = self.channel * self.size * self.size
        self.representation = self.cfg['contra']['representation']
        self.projection = self.cfg['contra']['projection']
        self.hidden = self.cfg['contra']['hidden']

        self.transform = nn.Sequential(
            nn.Conv2d(in_channels=channel, 
                        out_channels=self.channel, 
                        kernel_size=1, 
                        bias=False),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((self.size, self.size)),
            nn.Flatten(),
            nn.Linear(self.shape, self.representation),
            nn.ReLU(),
            )

        self.project = nn.Sequential(
            nn.Linear(self.representation, self.projection))

        self.classifier = nn.Sequential(
            nn.Linear(self.representation, self.num_class))
        
    def forward(self, x):
        if self.branch_uninitialized:
            self.branch_init(x)

        self.repr = self.transform(x)
        if self.cross:
            self.logits = self.classifier(self.repr)
            if self.temp:
                self.pred = F.softmax(self.logits/self.temperature, dim=1)
            else:
                self.pred = F.softmax(self.logits, dim=1) 
            self.conf, _ = torch.max(self.pred, 1)
        return x

class DevourModel(Model):
    def __init__(self, cfg, N=3):
        super(Model, self).__init__()
        self.cfg = cfg
        self.num_class = cfg['num_class']
        self.img_size = cfg['img_size']
        self.head_layer = nn.Sequential()
        self.feats = nn.ModuleList([])
        self.exfeats = nn.ModuleList([])
        self.exits = nn.ModuleList([])
        self.tail_layer = nn.Sequential()
        self.gate  = []
        self.n = N
        self.name = "resnet"

    def start_count(self):
        self.count = []
        
    def forward_init(self):
        
        x = torch.randn(1, 3, 1000, 1000)
        print("0. Generating input shape:",x.shape)
        x = self.head_layer(x)
        print("1. After head: ", x.shape)
        for i in range(self.n):
            x = self.feats[i].forward(x)
            k = i+2
            print("{}. After Feat: {}".format(k, x.shape))
            self.exits[i].forward(x)
        
        for i in range(len(self.exfeats)):
            k +=1
            x = self.exfeats[i].forward(x)
            print("{}. After Fetc: {}".format(k, x.shape))
        
        b, c, w, h = x.shape
        print("X. Input to Tail layer: ", x.shape)
        features = 100
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
        print("Hunting Module...")
        for n, m in module.named_children():
            print(n, ' ', type(m).__name__)

    def bite(self, module, start=0, end=0):
        result = []
        counter = 0
        assert end >= start
        print("Biting Module: ",type(module).__name__)
        print("start: {}, end: {}".format(start, end))
        print("Check \t | \t Module name")
        print("----------------------------")
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
            self.exits[i].forward(x)
            if self.exits[i].conf[0] > self.exits[i].threshold and self.gate[i]:
                    return self.exits[i].pred
        
        for i in range(len(self.exfeats)):
            x = self.exfeats[i].forward(x)
        x = self.tail_layer(x)      
        return x  

    def devour(self, backbone, name='resnet'):
        self.head_list = []
        self.body_list = []
        self.tail_list = []
        
        print("Devouring Module...")
        ### bite model based on types
        if 'efficientnet' in name:
            self.hunt(backbone)
            self.head_list = self.bite(backbone.features, start=0, end=0)
            end = len(backbone.features)
            self.body_list = self.bite(backbone.features, start=1, end=end)
            self.tail_list = self.bite(backbone         , start=1, end=2)

        if 'mobilenet' in name:
            self.hunt(backbone)
            self.head_list = self.bite(backbone.features, start=0, end=0)
            end = len(backbone.features)
            self.body_list = self.bite(backbone.features, start=1, end=end)
            self.tail_list = self.bite(backbone.classifier, start=0, end=2)

        if 'resnet' in name:
            self.hunt(backbone)
            self.head_list = self.bite(backbone, start=0, end=3)
            self.body_list = self.bite(backbone, start=4, end=7)
            self.tail_list = self.bite(backbone, start=8, end=9)

        if 'inception' in name:
            self.hunt(backbone)
            self.head_list = self.bite(backbone, start=0, end=6)
            self.body_list = self.bite(backbone, start=7, end=14)
            self.tail_list = self.bite(backbone, start=16, end=21)
            
        if 'vgg' in name:
            end = len(backbone.features)
            self.head_list = self.bite(backbone.features, start=0, end=2)
            self.body_list = self.bite(backbone.features, start=3, end=end)
            self.tail_list = self.bite(backbone.classifier, start=0, end=6)
            
            
        self.construct()
        del self.head_list
        del self.body_list
        del self.tail_list
        
    def construct(self):
        ## Create head layers
        self.head_layer = nn.Sequential(*self.head_list)
        
        ## Create body layers
        print("split feats={}, using N={} early exits"
              .format(len(self.body_list), self.n))
        
        div = len(self.body_list) / (self.n)
        div = int(div)
        print("divide size:",div)
        split_list = lambda test_list, x: [test_list[i:i+x] for i in range(0, len(test_list), x)]

        final_list = split_list(self.body_list, div)
        print("Constructing head-body-tail layers with early exits")
        print('---------------------------------------------------')
        print("<head layer>")
        print('     || ')
        ## feats layers are body layers with early exits
        for x in range(self.n):
            print('[feat layer #{}](size={}) -> [exit] #{}'
                  .format(x, len(final_list[x]), x))
            print('     || ')
            self.feats.append(nn.Sequential(*final_list[x]))
            self.exits.append(Branch(cfg=self.cfg))
            self.gate.append(False)
            self.temp = False

        ## fetc layers are extra body layers without early exits
        for x in range(self.n, len(final_list)):
            print('[fetc layer #{}](size={})'
                  .format(x, len(final_list[x])))
            print('     || ')
            self.exfeats.append(nn.Sequential(*final_list[x]))
        print("<tail layer>")
        print('---------------------------------------------------')
        len(self.feats) , len(self.exfeats)
        
        ##create tail layer
        print("creating tail layer")
        self.tail_layer = self.forward_init()
        print("Model Set Complete!")