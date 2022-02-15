# import yaml
# import argparse
# from early_ex.utils import *
# from early_ex.model.devour import LazyDevourModel
# from early_ex.model.backbone import get_backbone
# from early_ex.trainer.dce_branch import DCEBranchTrainer
# from tqdm import tqdm
# def main():
#     print("Lazy Entry Trainer v0.9")

#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         '--config', type=str, 
#         default = "./configs/base.yml")
#     args = parser.parse_args()
#     cfg = config(args.config)
#     backbone = get_backbone(cfg)

#     model = LazyDevourModel(cfg, N=cfg['num_exits'])
#     model.devour(backbone, cfg['backbone'])

#     trainer = DCEBranchTrainer(model, cfg)

#     trainer.trainset, trainer.testset = get_dataset(cfg)

#     trainer.train_loader = torch.utils.data.DataLoader(
#         trainer.trainset, 
#         batch_size=cfg['batch_size'], 
#         shuffle=True,  
#         num_workers=cfg['workers'],
#         pin_memory=True) 

#     trainer.val_loader = torch.utils.data.DataLoader(
#         trainer.testset, 
#         batch_size=cfg['batch_size'], 
#         shuffle=False,  
#         num_workers=cfg['workers'],
#         pin_memory=True) 

#     trainer.branch_init()
#     try:
        
#         for epoch in range(30):
#             train_tbar = tqdm(trainer.train_loader)
#             trainer.model.train()

#             for i, data in enumerate(train_tbar):
#                 input = data[0].to(trainer.device)
#                 label = data[1].to(trainer.device)

#                 # Forward function. Here where it gets nasty...
#                 x = trainer.model.head_layer(input)



#     except KeyboardInterrupt:
#         print("terminating backbone training")



# if __name__ == "__main__":
#     main()
