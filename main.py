import argparse
import time
import yaml
import os
from src.trainer.brtrainer import BranchTrainer
from src.utils import config, log_init

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default = "./configs/test.yml")
    args = parser.parse_args()
    cfg = config(args.config)
    trainer = BranchTrainer(cfg)

    # trainer.backbone_training()
    # trainer.backbone_validation(0)

    trainer.branch_init()
    trainer.branch_training()

    log = log_init(trainer, cfg)
    # measure_time(cfg,trainer,cfg['timed']['sample'] , log)

    trainer.set_gates(cfg)

    # test_gates(cfg,trainer,cfg['timed']['sample'],log)

    print(log)
    


    if not os.path.isdir(cfg['performance_path']):
        print('The directory is not present. Creating a new one..')
        os.mkdir(cfg['performance_path'])
    else:
        print('The directory is present.')

    with open(cfg['performance_path']+ cfg['backbone']+'_'+ cfg['dataset']+'_result.yaml', 'w') as file:
        yaml.dump(log, file)
if __name__ == "__main__":
    main()
