import yaml
import argparse
from early_ex.utils import *
from early_ex.model.devour import DevourModel
from early_ex.model.backbone import get_backbone
from early_ex.trainer.dce_branch import DCEBranchTrainer

def main():
    print("Devour & Branch Trainer v0.9")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, 
        default = "./configs/base.yml")
    args = parser.parse_args()
    cfg = config(args.config)
    backbone = get_backbone(cfg)

    model = DevourModel(cfg, N=cfg['num_exits'])
    model.devour(backbone, cfg['backbone'])

    trainer = DCEBranchTrainer(model, cfg)

    trainer.trainset, trainer.testset = get_dataset(cfg)

    trainer.train_loader = torch.utils.data.DataLoader(
        trainer.trainset, 
        batch_size=cfg['batch_size'], 
        shuffle=True,  
        num_workers=cfg['workers'],
        pin_memory=True) 

    trainer.val_loader = torch.utils.data.DataLoader(
        trainer.testset, 
        batch_size=cfg['batch_size'], 
        shuffle=False,  
        num_workers=cfg['workers'],
        pin_memory=True) 

    trainer.branch_init()
    try:
        for epoch in range(30):
            trainer.branch_train(epoch)
            trainer.scheduler.step()
            trainer.branch_valid(epoch)
    except KeyboardInterrupt:
        print("terminating backbone training")



if __name__ == "__main__":
    main()
