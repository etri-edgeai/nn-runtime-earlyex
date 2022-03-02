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

model = DevourModel(cfg, N=cfg['num_exits'], backbone=backbone)
self = DMEBranchTrainer(model, cfg)

model = model.to(cfg['device'])

try:
    for epoch in range(10): 
        print("epoch: ",epoch)
        self.metric_train()
        # self.metric_valid(epoch)
        self.metric_visualize()
except KeyboardInterrupt:
    print("terminate train")

self.metric_visualize()
self.metric_test2()

model_scripted = torch.jit.script(model) # Export to TorchScript
model_scripted.save('./checkpoints/model_scripted.pt') # Save




# self.metric_cali()

