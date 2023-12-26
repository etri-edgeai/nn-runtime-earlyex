import json
import os
import sys
from argparse import ArgumentParser
import time
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import yaml
from pytorch3d.loss import chamfer_distance
from torch.autograd import Variable
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import horovod.torch as hvd
from modules.mask_rcnn import maskrcnn_resnet50_fpn_v2, MaskRCNN
cwd = os.getcwd()
sys.path.append(cwd)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True
from modules.data.rendered import ShapeNetRenderedDataset, custom_collate_fn
from modules.e3rf import E3RFnet
from torch.utils.data.distributed import DistributedSampler
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    for name, module in model.named_modules():
        num_params = sum(p.numel() for p in module.parameters())
        if num_params > 0:
            print(f"{name:30s} {num_params / 1e6:.2f}M")

class Trainer():
    """Trainer Class containing train, validation, calibration"""
    def __init__(self, cfg):
        """Init config"""
        self.cfg = cfg
        self.device = hvd.rank()
        # self.device = torch.device("cuda:2")
        self.dataset = ShapeNetRenderedDataset(self.cfg, "train")
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=8, shuffle=False, 
            num_workers=0, collate_fn=custom_collate_fn,drop_last=True,
            sampler=DistributedSampler(
                self.dataset, num_replicas=hvd.size(), rank=hvd.rank())
            
            )
        self.val_dataset = ShapeNetRenderedDataset(self.cfg, "val")
        self.val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=1, shuffle=False, 
            num_workers=0, collate_fn=custom_collate_fn,
            sampler=DistributedSampler(
                self.val_dataset, num_replicas=hvd.size(), rank=hvd.rank())
            )

        self.model = E3RFnet(self.cfg,  num_class=49, device=self.device)
        self.model.train()
        self.model.training=True

        try:
            self.model.load_state_dict(torch.load(cfg['e3rf_checkpoints']))
            # self.model = torch.load(self.cfg['e3rf_checkpoints'])
        except:
            print("No E3RF(RCNN) checkpoint found, training from scratch...")            


        self.epoch = self.cfg['epochs']


        print("optimizer setup")

    def train(self):
        """
        Args:
            model: model to train
            train_loader: training data loader
            optimizer: optimizer
            epoch: current epoch
            device: cuda or cpu
        """
        # torch.cuda.empty_cache()
        self.model.to(self.device)
        self.model.train()
        self.model.training = True
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.0001)

        optimizer = hvd.DistributedOptimizer(
            optimizer, 
            named_parameters=self.model.named_parameters())

        hvd.broadcast_parameters(
            self.model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        for epoch in range(self.epoch):
            # pbar = tqdm(self.dataloader)
            total_loss = 0.0
            num_batches = 0 
            total_cls_loss = 0.0
            total_box_loss = 0.0
            total_mask_loss = 0.0
            total_rpn_box_loss = 0.0
            total_objectness_loss = 0.0

            for i, (inputs, targets) in enumerate(self.dataloader):
                start_time = time.time()
                optimizer.zero_grad()

                losses, det, _ = self.model(inputs, targets)

                loss = losses['loss_classifier'] + losses['loss_box_reg'] \
                    +  losses['loss_objectness'] + losses['loss_rpn_box_reg'] \
                    +  losses['loss_mask'] 
                    # + losses['pcd_loss'] + 5.0 *losses['loss_mask']
                # pcd_preds, detections, loss = self.model(imgs, depth, targets)
                num_batches += 1
                total_loss += loss.item()
                total_box_loss += losses['loss_box_reg'].item()
                total_cls_loss += losses['loss_classifier'].item()
                total_mask_loss += losses['loss_mask'].item()
                total_rpn_box_loss += losses['loss_rpn_box_reg'].item()
                total_objectness_loss += losses['loss_objectness'].item()
                
                # total_loss += loss
                loss.backward()
                optimizer.synchronize()
                optimizer.step()
                end_time = time.time()
                elapsed_time = end_time - start_time
                if hvd.rank() == 0 and i % 10 == 0:
                    total_iterations = len(self.dataloader)
                    remaining_iterations = total_iterations - i
                    remaining_time = remaining_iterations * elapsed_time
                    remaining_minutes = int(remaining_time // 60)
                    remaining_seconds = int(remaining_time % 60)
                    total_time = total_iterations * elapsed_time
                    total_minutes = int(total_time // 60)
                    total_seconds = int(total_time % 60)
                    print(f"[epoch: {epoch}], [{i}/{total_iterations}]")
                    print(f"Elapsed time: {int(elapsed_time * i //60)} minutes, {int(elapsed_time * i % 60)} seconds")
                    print(f"Estimated total time: {total_minutes} minutes {total_seconds} seconds")
                    print(f"Estimated remaining time: {remaining_minutes} minutes {remaining_seconds} seconds")
                # if i > 30:
                #     break
            avg_loss = total_loss / num_batches
            avg_cls_loss = total_cls_loss / num_batches
            avg_box_loss = total_box_loss / num_batches
            avg_mask_loss = total_mask_loss / num_batches
            avg_rpn_box_loss = total_rpn_box_loss / num_batches
            avg_objectness_loss = total_objectness_loss / num_batches

            print(f"[{epoch}/{self.cfg['epochs']}]Loss: {avg_loss:.4f}")
            print(f"cls_loss: {avg_cls_loss:.4f}")
            print(f"box_loss: {avg_box_loss:.4f}")
            print(f"mask_loss: {avg_mask_loss:.4f}")
            print(f"rpn_box_loss: {avg_rpn_box_loss:.4f}")
            print(f"objectness_loss: {avg_objectness_loss:.4f}")

            if epoch % 5 == 0 and hvd.rank() == 0:
                torch.save(self.model.state_dict(),  cfg['e3rf_checkpoints'])
                print("Model saved!")
            # if epoch % 2 == 0 and hvd.rank() == 0:
            #     self.valid()
            #     self.model.train()
            

    # def valid(self):
    #     # self.model.eval()
    #     #
    #     self.model.to(self.device)
        
    #     print("Validating...")
    #     pbar = tqdm(self.val_dataloader)
    #     total_loss = 0.0
    #     for i, (inputs, targets) in enumerate(pbar):
    #         start_time = time.time()

    #         with torch.no_grad():
    #             losses, detections, pcd_preds = self.model(inputs, targets)

    #     # avg_loss = total_loss / len(self.val_dataloader)
    #     # print(f"Validation Loss: {avg_loss:.4f}")
    #     torch.cuda.ipc_collect()


def main(cfg, rank):
    trainer = Trainer(cfg)    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("terminating training")
    finally:
        torch.save(trainer.model.state_dict(), cfg['e3rf_checkpoints'])

if __name__ == '__main__':
    """Main function"""
    hvd.init()
    torch.cuda.empty_cache()
    torch.cuda.set_device(hvd.local_rank())

    mp.set_start_method('spawn')
    # Load configuration from YAML file
    with open("./config/e3rf.yml", 'r') as f:
        cfg = yaml.safe_load(f)
    main(cfg, rank=hvd.rank())

