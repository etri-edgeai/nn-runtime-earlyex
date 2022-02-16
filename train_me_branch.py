import yaml
import argparse
from early_ex.utils import *
from early_ex.model.devour import DevourModel
from early_ex.model.backbone import get_backbone
from early_ex.trainer.dme_branch import DMEBranchTrainer
# from early_ex.trainer.ce_branch import DCEBranchTrainer
from tqdm import tqdm

print("Devour & Branch Trainer v0.9")

cfg = config("./configs/base.yml")
backbone = get_backbone(cfg)

model = DevourModel(cfg, N=cfg['num_exits'])
model.devour(backbone, cfg['backbone'])
self = DMEBranchTrainer(model, cfg)

model = model.to(cfg['device'])
    
for epoch in range(30): 
    self.metric_train()
    self.metric_valid(epoch)
# self.metric_test()

