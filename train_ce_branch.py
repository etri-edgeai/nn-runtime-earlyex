import yaml
import argparse
from early_ex.utils import *
from early_ex.model.devour import DevourModel
from early_ex.model.backbone import get_backbone
from early_ex.trainer.dce_branch import DCEBranchTrainer

def main():
    print("Devour & Branch Trainer v0.9")

    cfg = config("./early_ex/configs/base.yml")
    backbone = get_backbone(cfg)

    model = DevourModel(cfg, N=cfg['num_exits'], backbone=backbone)
    trainer = DCEBranchTrainer(model, cfg)
    model = model.to(cfg['device'])

    try:
        for epoch in range(10):
            trainer.branch_train(epoch)
            trainer.scheduler.step()
            trainer.branch_visualize(epoch)
    except KeyboardInterrupt:
        print("terminating backbone training")
    trainer.branch_visualize(epoch)
    trainer.branch_test()
        
    model_scripted = torch.jit.script(model) # Export to TorchScript
    model_scripted.save('./checkpoints/model_scripted.pt') # Save


if __name__ == "__main__":
    main()
