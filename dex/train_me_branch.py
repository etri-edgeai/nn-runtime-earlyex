import yaml
import argparse
from early_ex.utils import *
from early_ex.model import Model
from early_ex.model.devour import DevourModel
from early_ex.model.backbone import get_backbone
from early_ex.trainer.dme_branch import DMEBranchTrainer
from early_ex.trainer.backbone import BackboneTrainer
from tqdm import tqdm

def main():
    print("Devour & Branch Trainer v0.9")
    cfg = config("./early_ex/configs/base.yml")
    backbone = get_backbone(cfg)
    model = DevourModel(cfg, N=cfg['num_exits'], backbone=backbone)
    self = DMEBranchTrainer(model, cfg)
    model = model.to(cfg['device'])
    torch.cuda.synchronize()
    try:
        
        for epoch in range(20): 
            print("epoch: ",epoch)
            self.metric_train2()
            # self.metric_valid(epoch)
            self.metric_visualize()
    except KeyboardInterrupt:
        print("terminate train")

    self.metric_visualize()
    self.metric_test()
    model_scripted = torch.jit.script(model) # Export to TorchScript
    model_scripted.save('./model.pt') # Save

    model_scripted = torch.jit.script(backbone)
    model_scripted.save('./backbone.pt')
if __name__ == "__main__":
    main()