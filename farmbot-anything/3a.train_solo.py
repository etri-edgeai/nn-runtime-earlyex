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
from torch.autograd import Variable
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import horovod.torch as hvd
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

cwd = os.getcwd()
sys.path.append(cwd)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True
# torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = False
from modules.data.srendered import ShapeNetRenderedDataset, custom_collate_fn
from modules.solov1 import SOLOV1


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
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

        # Load dataset        
        self.dataset = ShapeNetRenderedDataset(self.cfg, "train")
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=cfg['3_batch_size'], shuffle=False, 
            num_workers=cfg['3_num_workers'],  collate_fn=custom_collate_fn,
            sampler=DistributedSampler(
                self.dataset, num_replicas=hvd.size(), rank=hvd.rank()))

        # Find total number of classes
        jsonpath = cfg['0_train_json']
        with open(jsonpath, 'r') as f:
            self.coco_annotations = json.load(f)
        self.coco_categories = self.coco_annotations['categories']
        num_classes = len(self.coco_categories) + 1 
        
        # Load model
        self.model = SOLOV1(num_classes=num_classes, device=self.device)
        try:
            print("Loading SOLOv1 checkpoint...")
            self.model.load_state_dict(
                torch.load(self.cfg['3_solo_checkpoints']))
        except:
            print("No SOLOv1 checkpoint found, training from scratch...")
        

    def train(self):
        # Setup model
        self.model.train()
        self.model.to(self.device)

        # Setup optimizer
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.cfg['3_learning_rate'])
        optimizer = hvd.DistributedOptimizer(
            optimizer, 
            named_parameters=self.model.named_parameters())

        # Broadcast parameters from rank 0 to all other processes
        hvd.broadcast_parameters(
            self.model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        # Start train loop
        for epoch in range(self.cfg['3_epochs']):
            total_loss = 0.0
            inst_loss = 0.0
            cate_loss = 0.0
            num_batches = 0
            for i, (images, data) in enumerate(self.dataloader):
                start_time = time.time()
                num_batches += 1

                # Load data
                imgs = torch.stack([
                    img.to(self.device) for img in images], dim=0).squeeze(0)
                depth = torch.stack([
                    d.to(self.device) for d in data['depth']], dim=0)

                # Load Target
                pcd = torch.stack([
                    p.to(self.device) for p in data['pcd']],dim=0)
                masks = [torch.stack(m).to(self.device) \
                         for m in data['masks']]
                bboxes = [torch.stack(b).to(self.device) \
                        for b in data['boxes']]
                labels = [torch.stack(l).to(self.device) \
                        for l in data['labels']]

                # Forward pass
                losses = self.model(img=imgs, gt_bboxes=bboxes, \
                                    gt_labels=labels, gt_masks=masks)
                # Backward pass
                loss = losses['loss_ins'] + losses['loss_cate']
                total_loss += loss.item()
                inst_loss += losses['loss_ins'].item()
                cate_loss += losses['loss_cate'].item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Print statistics
                end_time = time.time()
                elapsed_time = end_time - start_time
                if hvd.rank() == 0 and i % 50 == 0:
                    total_iterations = len(self.dataloader)
                    remaining_iterations = total_iterations - i
                    remaining_time = remaining_iterations * elapsed_time
                    r_min = int(remaining_time // 60)
                    r_sec = int(remaining_time % 60)
                    total_time = total_iterations * elapsed_time
                    total_min = int(total_time // 60)
                    total_sec = int(total_time % 60)
                    print(f"[epoch: {epoch}], [{i}/{total_iterations}]")
                    print(f"Elapsed time: {elapsed_time:.2f} seconds")
                    print(
                        f"Estimated total time: {total_min}:{total_sec}")
                    print(
                        f"Estimated remaining time: {r_min}:{r_sec}")

            # Print epoch statistics
            avg_loss = total_loss / num_batches
            avg_inst_loss = inst_loss / num_batches
            avg_cate_loss = cate_loss / num_batches
            print(f"[{epoch}/{self.cfg['3_epochs']}]Loss: {avg_loss:.4f}")
            print(f"Instance Loss: {avg_inst_loss:.4f}")
            print(f"Category Loss: {avg_cate_loss:.4f}")

            # Save checkpoint
            if epoch % 5 == 0 and hvd.rank()==0:
                print("Saving checkpoint...")
                torch.save(
                    self.model.state_dict(), self.cfg['3_solo_checkpoints'])

def main(cfg, rank):
    trainer = Trainer(cfg)
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("terminating training")
    

if __name__ == '__main__':
    """Main function"""
    hvd.init()
    torch.cuda.empty_cache()
    torch.cuda.set_device(hvd.local_rank())
    mp.set_start_method('spawn')
    # Load configuration from YAML file
    with open("./config/config.yml", 'r') as f:
        cfg = yaml.safe_load(f)
    main(cfg, rank=hvd.rank())    
