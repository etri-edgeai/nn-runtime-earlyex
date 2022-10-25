"""My version of Lazy Network"""
import torch
import torch.fx
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
from model.backbone import get_backbone, get_plan, get_modulelist
from model.ffm import FeatureFusion
from model.cbrelu import CBRelu
from model.skips import Skips
from model.spatial import Spatial
import copy

class LazyNet(nn.Module):
    """load backbone into model"""
    def __init__(self, cfg):
        """initialization function for lazy"""
        super().__init__()
        # Init configuration file
        self.cfg = cfg

        # Data config
        self.num_class  = cfg['num_class']
        self.img_size   = cfg['img_size']
        self.device     = cfg['device']

        # Backbone Config
        self.bb_name    = cfg['backbone']
        self.lazy_num   = cfg['lazy_num'] + 1
        backbone        = get_backbone(cfg)
        plan            = get_plan(backbone, cfg['backbone'])
        
        head_list       = get_modulelist(
            plan['head'], start=plan['head_start'], end=plan['head_end'])

        mid_list       = get_modulelist(
            plan['mid'], start=plan['mid_start'], end=plan['mid_end'])

        tail_list       = get_modulelist(
            plan['tail'], start=plan['tail_start'], end=plan['tail_end'])

        self.construct(backbone, head_list, mid_list, tail_list)

        # Train Mode set
        self.train_mode = True
        self.temperature= nn.Parameter(torch.Tensor([1]), requires_grad=False)
        self.exit_list = torch.zeros(self.lazy_num+2)

    def construct(self, backbone, head_list, mid_list, tail_list):
        """construct lazy parts using plan"""
        id = 0
        N = self.lazy_num
        input = torch.randn(1, 3, self.cfg['img_size'], self.cfg['img_size'])

        # Create Head layers
        self.head_layer = nn.Sequential(*head_list)
        self.feats_layers = nn.ModuleList([])
        self.skip_layers  = nn.ModuleList([])

        input = self.head_layer(input)
        
        # Set Spatial Analysis
        self.spatial    = Spatial(input, self.cfg).to(self.device)

        # Create Mid Layers
        print('---------------------------------------------------')
        print("split feats={}, using N={} Lazy Entries"
              .format(len(mid_list), N-1))
        div = len(mid_list) / N
        div = int(div)
        print("divide size:", div)
        split_list = lambda test_list, \
            x: [test_list[i:i+x] for i in range(0, len(test_list), x)]
        final_list = split_list(mid_list, div)
        print("Constructing head-body-tail layers with lazy entries")
        print("<head layer>")
        print('     || ')

        for x in range(N):              
            for y in range(len(final_list[x])):
                if y < len(final_list[x])-1:
                    print('[feat layer]')
                else:
                    print('[feat layer] -> [lazy #{}]'.format(x))
            print('     || ')
            self.feats_layers.append(nn.Sequential(*final_list[x]))
            input = self.feats_layers[x].forward(input)
            self.skip_layers.append(Skips(cfg=self.cfg, input=input, id=id))
            id += 1

        # Create Fetc layers
        self.fetc_layers = nn.ModuleList([])
        x += 1
        for y in range(x, len(final_list)):
            self.fetc_layers.append(nn.Sequential(*final_list[x]))
            print('[fetc layer]')
        
        for fetc in self.fetc_layers:
            input = fetc(input)
        
        print('     || ')
        print("<tail layer>")
        print('---------------------------------------------------')
        print("Model Set Complete!")
        
        self.ff_layer = FeatureFusion(input.shape, reduction=2)

        self.tail_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                self.cfg['__fc_features__'], self.cfg['num_class'], bias=True))

    def train_forward(self, input):
        """Forward Train function"""
        # Head Inference
        x = self.head_layer(input)

        # Spatial Analysis
        repr, logits = self.spatial(x)

        # Mid Inference
        ss = []
        for i, (feat, skip) in enumerate(
            zip(self.feats_layers, self.skip_layers)):
            x = feat(x)
            s = skip(x)
            ss.append(s)
        
        # Fetc Inference
        for fetc in self.fetc_layers:
            x = fetc(x)

        # Feature Fusion
        vals = []
        for s_iter in ss:
            val = self.ff_layer(s_iter, repr)
            val = self.tail_layers(val)
            vals.append(val)
        
        val = self.ff_layer(x, repr)
        val = self.tail_layers(val)
        vals.append(val)
        return vals, logits

    def test_forward(self, input):
        # Head Inference
        x = self.head_layer(input)

        # Spatial Analysis
        repr, logits = self.spatial(x)
        scaled = F.softmax(
            torch.div(logits, self.temperature), dim=1)
        conf, pred = torch.max(scaled, dim=1)

        skip_idx = 0
        
        try:
            thresholds = self.cfg['thresholds']

        except:
            thresholds = [0.85, 0.65, 0, 0, 0]
            # thresholds = [0, 0, 0, 0, 0]
            # thresholds = [1, 1, 1, 1, 0]
            # thresholds = [0, 1, 1, 1, 1]
            # thresholds = [1, 0, 1, 1, 1]
            
        if conf > thresholds[0]:
            self.exit_list[0] += 1
            return scaled

        for n in range(self.lazy_num+1):
            if conf < thresholds[n] and conf > thresholds[n+1]:
                self.exit_list[n+1] += 1
                skip_idx = n


        # Mid Inference
        for i, (feat, skip) in enumerate(
            zip(self.feats_layers, self.skip_layers)):
            x = feat(x)
            if i == skip_idx:
                s = skip(x)
                break

        val = []
        if skip_idx < 3:
            val = self.ff_layer(s, repr)
            val = self.tail_layers(val)

        # Fetc Inference
        if skip_idx >= 3:
            for fetc in self.fetc_layers:
                x = fetc(x)
            val = self.ff_layer(x, repr)
            val = self.tail_layers(val)
        return val

    def forward(self, input):
        """Forward function"""
        if self.train_mode:
            vals, logits = self.train_forward(input)
            return vals, logits
        else:
            pred = self.test_forward(input)
            return pred
