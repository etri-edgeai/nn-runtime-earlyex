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
# import horovod.torch as hvd

cwd = os.getcwd()
sys.path.append(cwd)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from modules.data.srendered import ShapeNetRenderedDataset, custom_collate_fn
from modules.model import FAMENet
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
    def __init__(self, cfg, device):
        """Init config"""
        self.cfg = cfg
        self.device = device
        # self.device = torch.device("cuda:2")
        self.dataset = ShapeNetRenderedDataset(self.cfg, "train")
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=cfg['4_batch_size'], shuffle=False, 
            num_workers=cfg['4_num_workers'], collate_fn=custom_collate_fn,
            # sampler=DistributedSampler(
            #     self.dataset, num_replicas=hvd.size(), rank=hvd.rank())
            )
        # self.val_dataset = ShapeNetRenderedDataset(self.cfg, "val")
        # self.val_dataloader = torch.utils.data.DataLoader(
        #     self.val_dataset, batch_size=1, shuffle=False, 
        #     num_workers=8, collate_fn=custom_collate_fn,
        #     sampler=DistributedSampler(
        #         self.val_dataset, num_replicas=hvd.size(), rank=hvd.rank())
        #     )

        self.model = FAMENet(self.cfg,  device = self.device, num_class=49)
        # count_parameters(self.model)
        try:
            self.model.load_state_dict( torch.load(cfg['4_se3rf_checkpoints']))
            self.model.to(self.device)
        except:
            print("No FAMENet checkpoint found, training from scratch...")            

        self.epoch = self.cfg['4_epochs']


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
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.cfg['4_learning_rate'])

        # optimizer = hvd.DistributedOptimizer(
        #     optimizer, 
        #     named_parameters=self.model.named_parameters())

        # hvd.broadcast_parameters(
        #     self.model.state_dict(), root_rank=0)
        # hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        for epoch in range(self.epoch):
            # pbar = tqdm(self.dataloader)
            total_loss = 0.0
            total_pcd_loss = 0.0
            total_inst_loss = 0.0
            total_cate_loss = 0.0
            num_batches = 0 
            # for i, (images,  data) in enumerate(pbar):
            for i, (images, data) in enumerate(self.dataloader):
                start_time = time.time()
                imgs = torch.stack([
                    img.to(self.device) for img in images], dim=0)
                depth = torch.stack([
                    d.to(self.device) for d in data['depth']], dim=0)
                pcd = torch.stack([
                    p.to(self.device) for p in data['pcd']],dim=0)
                # apcd = torch.stack([
                #     p.to(self.device) for p in data['apcd']],dim=0)
                apcd = [torch.stack([
                    p.to(self.device) for p in a],dim=0) \
                        for a in data['apcd']]
                masks = [torch.stack(m).to(self.device) \
                         for m in data['masks']]
                bboxes = [torch.stack(b).to(self.device) \
                        for b in data['boxes']]
                labels = [torch.stack(l).to(self.device) \
                        for l in data['labels']]
                imgs = imgs.squeeze(0)
                # print(pcd.shape, apcd.shape)
                los = self.model(imgs, depth, apcd, masks, bboxes, labels)
                num_batches += 1
                loss = los['pcd_loss'] + los['inst_loss'] + los['cate_loss']
                try:
                    total_pcd_loss += los['pcd_loss'].item() 
                except AttributeError:
                    total_pcd_loss += 0.0
                total_inst_loss += los['inst_loss'].item()
                total_cate_loss += los['cate_loss'].item()
                total_loss += loss.item()
                
                optimizer.zero_grad()
                # scaler.scale(loss).backward()
                # scaler.step(self.optimizer)
                # scaler.update()
                loss.backward()
                optimizer.step()
                end_time = time.time()
                elapsed_time = end_time - start_time
                # if hvd.rank() == 0 and i % 50 == 0:
                if i % 50 == 0:
                    total_iterations = len(self.dataloader)
                    remaining_iterations = total_iterations - i
                    remaining_time = remaining_iterations * elapsed_time
                    remaining_minutes = int(remaining_time // 60)
                    remaining_seconds = int(remaining_time % 60)
                    total_time = total_iterations * elapsed_time
                    total_minutes = int(total_time // 60)
                    total_seconds = int(total_time % 60)
                    print(f"[epoch: {epoch}], [{i}/{total_iterations}]")
                    print(f"Elapsed time: {elapsed_time:.2f} seconds")
                    print(f"Estimated total time: {total_minutes} minutes {total_seconds} seconds")
                    print(f"Estimated remaining time: {remaining_minutes} minutes {remaining_seconds} seconds")
            avg_loss = total_loss / num_batches
            print(f"[{epoch}/{self.cfg['epochs']}]Loss: {avg_loss:.4f}")
            print(f"pcd_loss: {total_pcd_loss / num_batches:.4f}")
            print(f"inst_loss: {total_inst_loss / num_batches:.4f}")
            print(f"cate_loss: {total_cate_loss / num_batches:.4f}")
            # if epoch % 5 == 0 and hvd.rank() == 0:
            if epoch % 5 == 0:
                torch.save(self.model.state_dict(),  cfg['4_se3rf_checkpoints'])
                print("Model saved!")


def main(cfg, rank):
    trainer = Trainer(cfg, rank)    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("terminating training")
    finally:
        torch.save(trainer.model.state_dict(), cfg['4_se3rf_checkpoints'])

if __name__ == '__main__':
    """Main function"""
    # hvd.init()
    torch.cuda.empty_cache()
    # torch.cuda.set_device(hvd.local_rank())

    mp.set_start_method('spawn')
    # Load configuration from YAML file
    with open("./config/config.yml", 'r') as f:
        cfg = yaml.safe_load(f)
    main(cfg,
         rank = torch.device("cuda:0")
        #  rank=hvd.rank()
         )

