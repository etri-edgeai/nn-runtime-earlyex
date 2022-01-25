import yaml
import argparse
from early_ex.utils import *
from early_ex.model.devour import DevourModel
from early_ex.model.backbone import get_backbone

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
if __name__ == "__main__":
    main()
